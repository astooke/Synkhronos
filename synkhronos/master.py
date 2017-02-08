
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything exposed to the user.
"""
# import ipdb
# import gtimer as gt

import pickle
import numpy as np
import multiprocessing as mp
import theano
from threading import BrokenBarrierError
import atexit

from .variables import struct, Inputs, Shareds, Outputs, SynkFunction
from .common import use_gpu
from .common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                    ALL_GATHER, GATHER, CPU_COMM, AVG_ALIASES, SCATTER)
from .util import (get_n_gpu, build_sync, check_collect, check_op,
                  check_func_scatter, get_worker_reduce_ops,
                  check_shared_var, check_scatter_sources, get_shared_IDs)


# Globals  (only functions exposed to user will use via global access)
g = struct(
    # State
    forked=False,
    distributed=False,
    closed=False,
    # Multiprocessing
    sync=None,
    processes=list(),
    # Theano
    inputs=Inputs(),
    shareds=Shareds(),
    outputs=Outputs(),
    # GPU
    synk_functions=list(),
    n_gpu=None,
    gpu_comm=None,
    master_rank=None,
)


###############################################################################
#                                                                             #
#                           Building Functions.                               #
#                                                                             #
###############################################################################


class Function(SynkFunction):
    """ TODO: Function docstring """

    _n_gpu = None
    _master_rank = None

    def __init__(self, shared_IDs, output_IDs, g_inputs, g_outputs,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_IDs = shared_IDs
        self._output_IDs = output_IDs
        self._name = self._theano_function.name
        self._build_output_subset_shmem()

        # For streamlining some helper operations.
        self._input_names = [g_inputs.names[i] for i in self._input_IDs]
        self._input_vars = [g_inputs.vars[i] for i in self._input_IDs]
        self._output_to_cpu = [g_outputs.to_cpu[i] for i in self._output_IDs]
        self._output_avg_funcs = [g_outputs.avg_funcs[i] for i in self._output_IDs]
        self._n_outputs = len(output_IDs)
        self._previous_batch_size = None
        self._my_idx = [0, 0]
        self._previous_output_subset = None
        self._n_inputs = len(self._input_IDs)

    @property
    def name(self):
        return self._name

    @property
    def output_to_cpu(self):
        return self._output_to_cpu

    ###########################################################################
    #                       User callables (use globals g directly)           #

    def __call__(self, *args, **kwargs):
        """
        This needs to:
        1. Share input data.
        2. Signal to workers to start and what to do.
        3. Call the local theano function on data.
        4. Collect result from workers and return it.

        NOTE: Barriers happen INSIDE master function call.
        """
        if not g.distributed:
            raise RuntimeError("Synkhronos functions have not been distributed "
                "to workers, can only call Theano function.")
        if g.closed:
            raise RuntimeError("Synkhronos already closed, can only call "
                "Theano function.")
        return_shmems = kwargs.pop("return_shmems", False)
        output_subset = kwargs.pop("output_subset", None)
        input_datas = self._order_inputs(g.inputs, args, kwargs)
        input_shmems = self._update_shmems(g.inputs, input_datas)
        output_set = self._share_output_subset(output_subset)
        g.sync.exec_ID.value = FUNCTION
        g.sync.func_ID.value = self._ID
        g.sync.barriers.exec_in.wait()
        my_inputs = self._get_my_inputs(g.inputs)
        my_results = self._call_theano_function(my_inputs, output_subset)  # always a list

        results = self._collect_results(g.gpu_comm, my_results, output_set)  # always returns list
        exec_out_check(g.sync)
        if return_shmems:
            results.append(input_shmems)  # append list of results with tuple of shmems
        if len(results) == 1:
            results = results[0]
        return results

    def get_input_shmems(self, *args, **kwargs):
        """ doctstring """
        if not g.distributed or g.closed:
            raise RuntimeError("Cannot call this method on inactive synkhronos "
                "function.")
        if not args and not kwargs:  # (simply gather existing)
            input_shmems = list()
            for input_ID in self._input_IDs:
                input_shmems.append(g.inputs.shmems[input_ID])
        else:  # (make new ones according to input datas)
            input_datas = self._order_inputs(g.inputs, args, kwargs)
            input_shmems = self._update_shmems(g.inputs, input_datas)
        return input_shmems

    def as_theano(self, *args, **kwargs):
        """ Use this to get outputs back on CPU as originally built. """
        results = self._theano_function(*args, **kwargs)
        if not isinstance(results, list):
            results = [results]
        output_subset = kwargs.pop("output_subset", None)
        if output_subset is None:
            for idx, to_cpu in enumerate(self._output_to_cpu):
                if to_cpu:
                    results[idx] = np.asarray(results[idx])
        else:
            for idx_r, idx in enumerate(output_subset):
                if self._outputs_to_cpu[idx]:
                    results[idx_r] = np.asarray(results[idx_r])
        if len(results) == 1:
            results = results[0]
        return results

    ###########################################################################
    #                     Helpers (not for user)                              #

    def _order_inputs(self, g_inputs, args, kwargs):
        """ Includes basic datatype and ndims checking. """
        if len(args) + len(kwargs) != self._n_inputs:
            raise TypeError("Incorrect number of inputs to synkhronos function.")
        ordered_inputs = list(args)
        if kwargs:
            ordered_inputs += [None] * len(kwargs)
            for key, input_data in kwargs.iteritems():
                if key in self._input_names:
                    idx = self._input_names.index(key)
                elif key in self._input_vars:
                    idx = self._input_vars.index(key)
                else:
                    raise ValueError("Input passed as keyword arg not found "
                        "in inputs (vars or names) of function: ", key)
                if ordered_inputs[idx] is None:
                    ordered_inputs[idx] = input_data
                else:
                    raise ValueError("Received duplicate input args/kwargs: ",
                        key)
        input_datas = g_inputs.check_inputs(self._input_IDs, ordered_inputs)
        return input_datas

    def _share_output_subset(self, output_subset):
        if output_subset != self._previous_output_subset:
            if output_subset is None:
                self._output_subset_shmem[:] = True
            else:
                if not isinstance(output_subset, list):
                    raise TypeError("Optional param output_subset must be a "
                        "list of ints.")
                for idx in output_subset:
                    if not isinstance(idx, int):
                        raise TypeError("Optional param output_subset must a "
                            "list of ints.")
                    if idx < 0 or idx > self._n_outputs - 1:
                        raise ValueError("Output_subset entry out of range.")
                self._output_subset_shmem[:] = False
                for idx in output_subset:
                    self._output_subset_shmem[idx] = True
            self._previous_output_subset = output_subset
        output_set = [i for i, x in enumerate(self._output_subset_shmem) if x]
        return output_set

    def _update_shmems(self, g_inputs, input_datas):
        self._update_batch_size(g_inputs, input_datas)
        shmems = list()
        for input_data, input_ID in zip(input_datas, self._input_IDs):
            shmems.append(g_inputs.update_shmem(input_ID, input_data))
        return shmems

    def _update_batch_size(self, g_inputs, input_datas):
        if not any(self._inputs_scatter):
            return  # (all inputs broadcast, no data parallel)
        b_size = None
        for input_data, scatter in zip(input_datas, self._inputs_scatter):
            if scatter:
                b_size = input_data.shape[0] if b_size is None else b_size
                if input_data.shape[0] != b_size:
                    raise ValueError("Scatter Inputs of different batch sizes "
                        "(using 0-th index).")
        if b_size != self._previous_batch_size:
            assign_idx = np.ceil(
                np.linspace(0, b_size, self._n_gpu + 1)).astype(int)
            g_inputs.sync.assign_idx[self._ID][:] = assign_idx
            self._my_idx = (assign_idx[self._master_rank],
                            assign_idx[self._master_rank + 1])
            self._previous_batch_size = b_size

    def _get_my_inputs(self, g_inputs):
        s_idx = self._my_idx[0]
        e_idx = self._my_idx[1]
        my_inputs = list()
        for input_ID, scatter in zip(self._input_IDs, self._inputs_scatter):
            if scatter:
                my_inputs.append(g_inputs.shmems[input_ID][s_idx:e_idx])
            else:
                max_idx = g_inputs.sync.max_idx[input_ID]
                my_inputs.append(g_inputs.shmems[input_ID][:max_idx])
        return my_inputs

    def _collect_results(self, gpu_comm, my_results, output_set):
        results = list()
        for idx, r in zip(output_set, my_results):
            mode = self._collect_modes[idx]
            op = self._reduce_ops[idx]
            if mode == "reduce":
                op = "sum" if op in AVG_ALIASES else op
                gpu_comm.reduce(r, op=op, dest=r)  # (in-place)
            elif mode == "gather":
                r = gpu_comm.all_gather(r)
            elif mode is not None:
                raise RuntimeError("Unrecognized collect mode in master "
                    "function: ", mode)
            results.append(r)
        for idx_r, idx in enumerate(output_set):
            mode = self._collect_modes[idx]
            op = self._reduce_ops[idx]
            if mode == "reduce" and op in AVG_ALIASES:
                results[idx_r] = self._output_avg_funcs[idx](results[idx_r])
            if self._output_to_cpu[idx]:
                results[idx_r] = np.array(results[idx_r])
        return results


def function(inputs, outputs=None,
             collect_modes="reduce", reduce_ops="avg",
             broadcast_inputs=None, scatter_inputs=None,
             **kwargs):
    """
    Use when creating a Theano function instead of ``theano.function``.

    ``collect_modes`` and ``reduce_ops`` can be single strings or lists which
    determine how each output is handled.  ``[None]`` is a valid entry, which
    results in no communication from workers to master.  (In the future, this
    will only be the default behavior for the function, but will be possible to
    overrule when calling.)

    Use either ``broadcast_inputs`` or ``scatter_inputs`` to list the variables
    in either category; the remainder will do the opposite.  If nothing is
    provided, all inputs are scattered.  Scattering is performed by dividing the
    data evenly along the 0-th dimension.

    Inputs and outputs need not be variables transferred to the GPU by the user.
    Internally, synkhronos will apply these transfers so that all outputs remain
    on their respective worker GPU, so that data is collected to the master GPU
    via GPU-comms.  In the end, the outputs will be returned to the CPU in the
    master process only.  If the user provides any outputs already appended
    with a transfer to remain on the GPU, they will be left there in the master.

    Args:
        inputs (var): as ``inputs`` in ``theano.function``
        outputs (None, optional): as ``outputs`` in ``theano.function``
        collect_modes (str, list, optional): default behaviors; "gather" or "reduce"
        reduce_ops (str, list, optional): default behaviors; e.g. "sum", "prod", "min", "max", "avg"
        broadcast_inputs (None, optional): list of vars or names of inputs to broadcast
        scatter_inputs (None, optional): list of vars or names of inputs to scatter
        **kwargs (TYPE): passed directly to ``Theano.function``

    Raises:
        RuntimeError: If not yet forked or if already distributed.

    Returns:
        SynkFunction: Callable like a Theano function.
    """
    if not g.forked:
        raise RuntimeError("Must fork before making functions for GPU.")
    if g.distributed:
        raise RuntimeError("Cannot make new functions after distributing.")

    inputs_scatter = check_func_scatter(inputs, broadcast_inputs, scatter_inputs)
    collect_modes, reduce_ops = check_collect(outputs, collect_modes, reduce_ops)
    gpu_outputs, output_IDs = g.outputs.register(outputs)
    theano_function = theano.function(inputs, gpu_outputs, **kwargs)
    input_IDs = g.inputs.register_func(theano_function)
    shared_IDs = g.shareds.register_func(theano_function)
    synk_function = Function(ID=len(g.synk_functions),  # Fcn can ID itself
                             theano_function=theano_function,
                             input_IDs=input_IDs,
                             shared_IDs=shared_IDs,
                             output_IDs=output_IDs,
                             inputs_scatter=inputs_scatter,
                             collect_modes=collect_modes,
                             reduce_ops=reduce_ops,
                             g_inputs=g.inputs,
                             g_outputs=g.outputs,
                             )
    g.synk_functions.append(synk_function)
    return synk_function


###############################################################################
#                                                                             #
#                      GPU Collectives.                                       #
#                                                                             #
###############################################################################


def gpu_comm_prep(comm_ID, functions=None, shared_vars=None,
                  has_op=False, op=None):
    """ Not called by user but using direct globals access to streamline. """
    if not g.distributed:
        raise RuntimeError("Synk functions not yet distributed-- \
            cannot call comm functions.")
    if g.closed:
        raise RuntimeError("synk already closed--cannot call comm \
            functions.")
    g.sync.exec_ID.value = GPU_COMM
    g.sync.comm_ID.value = comm_ID
    shared_IDs = get_shared_IDs(g.shareds, functions, shared_vars)
    n_shared = len(shared_IDs)
    g.shareds.sync.shared_IDs[:n_shared] = shared_IDs
    g.sync.n_shared.value = n_shared
    if has_op:
        op_ID = check_op(op)
        avg = op in AVG_ALIASES
        op = "sum" if avg else op
        g.sync.op_ID.value = op_ID
        return shared_IDs, op, avg
    else:
        return shared_IDs


###############################################################################
#                       User functions                                        #


def broadcast(shared_vars=None, functions=None):
    """GPU-comm: broadcast values from master to workers.

    In all multi-variable GPU-comm functions, the default behavior if no
    variables and no functions are provided is to call the operation on all
    shared variables in the session.

    Args:
        shared_vars (None, optional): names or vars to be broadcast
        functions (None, optional): functions to have all shared vars broadcast
    """
    shared_IDs = gpu_comm_prep(BROADCAST, functions, shared_vars)
    g.sync.barriers.exec_in.wait()
    for shared_ID in shared_IDs:
        src = g.shareds.gpuarrays[shared_ID]
        g.gpu_comm.broadcast(src)
    exec_out_check(g.sync)


def gather(shared_vars=None, functions=None, dest=None, nd_up=None):
    """GPU-comm: gather values from workers into master.

    (Calls all_gather, but results are ignored in workers--can't have new shared
    variables in them.)

    Args:
        shared_vars (None, optional): names or vars to gather
        functions (None, optional): functions to have all shared vars gathered
        dest (None, optional): GPU-array to write result (only if one var)
        nd_up (None, optional): Number of additional dimensions in result

    Raises:
        ValueError: Description

    Returns:
        List of GPUArrays, if no destination provided.
    """
    shared_IDs = gpu_comm_prep(GATHER, functions, shared_vars)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying destination, can only gather one var.")
    g.sync.barriers.exec_in.wait()
    results = list()
    for shared_ID in shared_IDs:
        src = g.shareds.gpuarrays[shared_ID]
        r = g.gpu_comm.all_gather(src, dest=dest, nd_up=nd_up)
        results.append(r)
    exec_out_check(g.sync)
    if dest is None:
        return results


def reduce(shared_vars=None, functions=None, op="avg", in_place=True, dest=None):
    """GPU-comm: workers reduce values to master only.

    Can only use destination when reducing a single shared variable.  In-place
    will overwrite the values of all shared variables involed (in the master
    only), otherwise will return new GPU-arrays.

    Args:
        shared_vars (None, optional): names or vars to be reduced
        functions (None, optional): functions to have all shared vars reduced
        op (str, optional): e.g. "sum, prod, min, max, avg"
        in_place (bool, optional): overwrite result into shared var source
        dest (None, optional): GPU-array to write result (only if one var)

    Raises:
        ValueError: If infeasible inputs.

    Returns:
        List of GPUArrays, if no destination provided and not in-place.
    """
    shared_IDs, op, avg = gpu_comm_prep(REDUCE, functions, shared_vars, True, op)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying desination, can only reduce one var.")
    if avg and (not in_place or dest is not None):
        raise ValueError("Can only use 'average' op with in-place reduce "
            "(requires None dest).")
    g.sync.barriers.exec_in.wait()
    results = list()
    for shared_ID in shared_IDs:
        src = g.shareds.gpuarrays[shared_ID]
        dest = src if dest is None and in_place else dest
        results.append(g.gpu_comm.reduce(src, op, dest))
    if avg:
        for shared_ID in shared_IDs:
            g.shareds.avg_funcs[shared_ID]()
    exec_out_check(g.sync)
    if not in_place and dest is None:  # (otherwise results will be Nones)
        return results


def all_reduce(shared_vars=None, functions=None, op="avg"):
    """GPU-comm: master and workers all reduce values, in-place only.

    Args:
        shared_vars (None, optional): names or vars to be reduced
        functions (None, optional): functions to have all shared vars reduced
        op (str, optional): e.g. "sum, prod, min, max, avg"
    """
    shared_IDs, op, avg = \
        gpu_comm_prep(ALL_REDUCE, functions, shared_vars, True, op)
    g.sync.barriers.exec_in.wait()
    for shared_ID in shared_IDs:
        src = g.shareds.gpuarrays[shared_ID]
        g.gpu_comm.all_reduce(src, op, src)
    if avg:
        for shared_ID in shared_IDs:
            g.shareds.avg_funcs[shared_ID]()
    exec_out_check(g.sync)


def all_gather(source, dest):
    """GPU-comm: master and workers all gather values into their local vars.

    Only one Theano shared variable can be used for the source, and another
    Theano shared variable of the right shape must already exist for use as the
    destination (since no new shared variables can be created in workers).

    Args:
        source (name or var): shared variable to be gathered
        dest (name or var): shared variable to receive values in
    """
    shared_IDs = gpu_comm_prep(ALL_GATHER, shared_vars=[source, dest])
    g.sync.barriers.exec_in.wait()
    src = g.shareds.gpuarrays[shared_IDs[0]]
    dest = g.shareds.gpuarrays[shared_IDs[1]]
    g.gpu_comm.all_gather(src, dest)
    exec_out_check(g.sync)


###############################################################################
#                                                                             #
#                         CPU-based Communications                            #
#                                                                             #
###############################################################################


def scatter(shared_var, sources):
    """CPU-comm: Scatter shared variable values across master and workers.

    This collective communication is performed on the CPU.  Values from arrays
    in `sources` are copied into synkhronos shared memory, and workers use these
    values to set their respective GPU variable values.  Must include entry for
    master.  All arrays must be the same shape as the Theano shared variable.

    TODO: return the shared memory used for this operation so user can write to
    it.

    Args:
        shared_var (name or var): Theano shared variable to have value set.
        sources (list): List of arrays to be used in shared_var.set_value().
    """
    shared_var, shared_ID = check_shared_var(g.shareds, shared_var)
    sources = check_scatter_sources(g.shareds, g.n_gpu, sources, shared_ID)
    if g.shareds.shmems[shared_ID] is None:
        g.shareds.build_shmems(shared_ID, g.n_gpu, g.master_rank)
    for rank, src in enumerate(sources):
        if rank == g.master_rank:
            shared_var.set_value(src)
        else:
            g.shareds.shmems[shared_ID][rank][:] = src
    g.sync.exec_ID.value = CPU_COMM
    g.sync.comm_id.value = SCATTER
    g.sync.shared_IDs[0] = shared_ID  # (can only to one per call)
    g.sync.barriers.exec_in.wait()
    exec_out_check(g.sync)


###############################################################################
#                                                                             #
#                       Initializing and Exiting.                             #
#                                                                             #
###############################################################################


def fork(n_gpu=None, master_rank=0):
    """Fork a python sub-process for each additional GPU and initialize.

    Call this function before building any Theano variables.  (Theano must be
    configured to ipmort to CPU only.)  Initializes one GPU on each process,
    including the master, and initializes GPU collective communications via
    pygpu & NVIDIA NCCL.

    Args:
        n_gpu (None, optional): Number of GPUs to use (default is all)
        master_rank (int, optional): default is 0

    Raises:
        RuntimeError: If already forked or fails to initialize.

    Returns:
        int: number of GPUs using.
    """
    if g.forked:
        raise RuntimeError("Only fork once.")
    from .worker import worker_exec

    n_gpu, master_rank = get_n_gpu(n_gpu, master_rank)
    sync = build_sync(n_gpu)

    for rank in [r for r in range(n_gpu) if r != master_rank]:
        args = (rank, n_gpu, master_rank, sync)
        g.processes.append(mp.Process(target=worker_exec, args=args))
    for p in g.processes:
        p.start()

    atexit.register(_close)

    gpu_comm = use_gpu(master_rank, n_gpu, sync)
    if not gpu_comm:
        raise RuntimeError("At least one synkhronos worker failed to "
            "initialize GPU during fork.")
    else:
        print("Synkhronos: " + str(n_gpu) + " GPUs succesfully initialized, "
            "master rank: " + str(master_rank))

    g.forked = True
    g.n_gpu = n_gpu
    g.master_rank = master_rank
    g.sync = sync
    g.gpu_comm = gpu_comm

    Function._n_gpu = n_gpu
    Function._master_rank = master_rank

    return n_gpu


def distribute():
    """Sets up theano functions from master on workers.

    Pickles all theano functions built with this package (i.e. using
    ``synkhronos.function()``) into one file, which workers unpickle.  Theano's
    behavior is to include all shared variable values in the file.  Workers are
    aware of correspondences among input and shared variables used in multiple
    functions, for efficient memory usage.  Functions are compiled in the master
    only.

    In the future, distribution will happen automatically, lazily at the time of
    any function call when it is necessary.  It will remain optional for the
    user to call, as it may be time-consuming.

    The pickle file is automatically deleted by a worker.

    Raises:
        RuntimeError: If not yet forked or if already distributed.
    """
    if not g.forked:
        raise RuntimeError("Need to fork before distributing functions.")
    if g.distributed:
        raise RuntimeError("Can distribute only once.")

    # Pickle all functions together in one list to preserve correspondences
    # among variables in different functions.
    g.shareds.set_avg_facs(g.n_gpu)
    pkl_functions = [sf.theano_function for sf in g.synk_functions]
    pkl_functions += g.shareds.avg_funcs
    with open(PKL_FILE, "wb") as f:
        pickle.dump(pkl_functions, f, pickle.HIGHEST_PROTOCOL)

    # Finishing building sync objects and writing function setup info.
    g.inputs.build_sync(len(g.synk_functions), g.n_gpu)
    g.shareds.build_sync()
    g.sync.n_user_fcns.value = len(g.synk_functions)
    g.sync.dict["collect_modes"] = [fn._collect_modes for fn in g.synk_functions]
    g.sync.dict["reduce_ops"] = get_worker_reduce_ops(g.synk_functions)
    g.sync.dict["inputs_scatter"] = [fn._inputs_scatter for fn in g.synk_functions]

    # Signal workers to receive.
    g.sync.distributed.value = True
    g.sync.barriers.distribute.wait()

    g.outputs.set_avg_facs(g.n_gpu)
    g.sync.barriers.distribute_out.wait()
    g.distributed = True


def close():
    """Close workers and join their processes.  Called automatically on exit.
    """
    if not g.forked:
        print("Warning: Calling close() before forking has no effect.")
    elif g.closed:
        print("Warning: Called close() after synkhronos already closed.")
    else:
        _close()


def _close():
    """ Called automatically on exit any time after fork. """
    if g.forked and not g.closed:
        # (try to get workers to exit quietly)
        if not g.sync.distributed.value:
            try:
                g.sync.barriers.distribute.wait(1)
            except BrokenBarrierError:
                pass
        else:
            g.sync.quit.value = True
            try:
                g.sync.barriers.exec_in.wait(1)
            except BrokenBarrierError:
                pass
        for p in g.processes:
            p.join()
        g.closed = True


def exec_out_check(sync):
    sync.barriers.exec_out.wait()
    if not sync.workers_OK.value:
        raise RuntimeError("Encountered worker error during execution loop.")
