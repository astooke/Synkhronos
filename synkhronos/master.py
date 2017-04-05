
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has (almost) everything exposed to the user.
"""
# import ipdb
import pickle
import numpy as np
import multiprocessing as mp
import theano
from threading import BrokenBarrierError
import atexit

from .function import FunctionHelpers
from .function import process_outputs, process_updates, process_givens
from .data import DataHelpers
from .scatterer import Scatterer
from .shareds_registry import SharedsRegistry

from .common import *
from .util import *
from .accumulators import Accumulators

CREATE = True

# Globals  (only functions exposed to user will use via global access)
g = struct(
    # State
    forked=False,
    distributed=False,
    closed=False,
    # Multiprocessing
    sync=None,
    processes=list(),
    scatterer=None,
    # Theano
    shareds=SharedsRegistry(),
    # GPU
    synk_functions=list(),
    accumulators=Accumulators(),
    gpu_comm=None,
    # ZMQ
    cpu_comm=None,
)


###############################################################################
#                                                                             #
#                      Function: user callable methods                        #
#                                                                             #
###############################################################################


class Function(FunctionHelpers):
    """ Class of instances returned by ``synkhronos.function()``.  """

    def __call__(self, *args, output_subset=None, batch=None, num_slices=1,
                 **kwargs):
        """ Callable as in Theano function.

        When called, Synkhronos functions:

            1. Share input data,
            2. Signal to workers to start and what to do,
            3. Call the local theano function on assigned data subset,
            4. Collect results from workers and return it.

        Theano function keyword argument ``output_subset`` is supported.

        Args:
            *args (data): Normal data inputs to Theano function
            **kwargs (data): Normal data inputs to Theano function

        Raises:
            RuntimeError: If not distributed or if synkhronos closed.
        """
        check_active()
        ordered_inputs = self._order_inputs(args, kwargs)
        self._share_input_data(g.scatterer, ordered_inputs, batch)
        self._update_f_info(g.sync.func, num_slices, output_subset)
        exct_in(FUNCTION, self._ID)
        my_results = self._run_function(g.scatterer, num_slices, output_subset)
        outputs = self._collect_results(g.gpu_comm, g.cpu_comm, my_results)
        exct_out()
        if not self._return_list and len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def as_theano(self, *args, **kwargs):
        """Call the function in the master process only, as normal Theano.

        This method will return outputs to the CPU if they were originally
        requested there, unlike using ``function.theano_function()``, which is
        built to hold all outputs on the GPU.

        Args:
            *args (data): Normal data inputs to the Theano function
            **kwargs (data): Normal data inputs to the Theano function
        """
        # FIXME:  possibly out of date.
        results = self._f(*args, **kwargs)
        if not isinstance(results, list):
            results = [results]
        o_subset = kwargs.pop("output_subset", None)
        output_set = range(self._n_output) if o_subset is None else o_subset
        for idx_r, idx_o in enumerate(output_set):
            if self._outputs.to_cpu[idx_o]:
                results[idx_r] = np.asarray(results[idx_r])
        if len(results) == 1:
            results = results[0]
        return results

    def build_inputs(self, *args, **kwargs):
        """ convenience method which internally calls synkhronos.data() for
        each input variable associated with this function; provide data inputs
        as if calling the Theano function.
        # TODO: move force_cast and oversize to function signature?
        """
        # FIXME: possibly out of date
        force_cast = kwargs.pop("force_cast", False)
        oversize = kwargs.pop("oversize", 1)
        scatter = kwargs.pop("scatter", True)
        inputs = self._order_inputs(args, kwargs)
        if not isinstance(scatter, (list, tuple)):
            scatter = [scatter] * self._n_input
        elif len(scatter) != self._n_input:
            raise ValueError("Scatter must be single boolean or list/tuple of "
                "length equal to the number of inputs.")
        synk_datas = list()
        for var, inpt, scat in zip(self._input.vars, inputs, scatter):
            synk_data = data(value=inpt,
                             dtype=var.dtype,
                             scatter=scat,
                             force_cast=force_cast,
                             oversize=oversize,
                             )
            synk_datas.append(synk_data)
        return tuple(synk_datas)


###############################################################################
#                                                                             #
#                        Building Functions                                   #
#                                                                             #
###############################################################################


def function(inputs, outputs=None, bcast_inputs=None, updates=None,
             givens=None, sliceable_shareds=None,
             **kwargs):
    """
    Use when creating a Theano function instead of ``theano.function``.

    ``collect_modes`` and ``reduce_ops`` can be single strings or lists which
    determine how each output is handled.  ``[None]`` is a valid entry, which
    results in no communication from workers to master.  (In the future, this
    will only be the default behavior for the function, but will be possible to
    overrule when calling.)

    All inputs are scattered evenly along the `0-th` dimension.  (Use a Theano
    shared variable for a broadcasted input.)

    Inputs and outputs need not be variables transferred to the GPU by the user.
    Internally, synkhronos will apply these transfers so that all outputs remain
    on their respective worker GPU, so that data is collected to the master GPU
    via GPU-comms.  In the end, the outputs will be returned to the CPU in the
    master process only.  If the user provides any outputs already appended
    with a transfer to remain on the GPU, they will be left there in the master.

    Args:
        inputs (var): as ``inputs`` in ``theano.function()``
        outputs (None, optional): as ``outputs`` in ``theano.function()``
        collect_modes (str, list, optional): default behaviors;
            "gather" or "reduce"
        reduce_ops (str, list, optional): default behaviors;
            "sum", "prod", "min", "max", "avg"
        **kwargs (TYPE): passed directly to ``theano.function()``

    Raises:
        RuntimeError: If not yet forked or if already distributed.

    Returns:
        Synkhronos.Function: Callable like a Theano function.
    """
    if not g.forked:
        raise RuntimeError("Must fork before making functions for GPU.")
    if g.distributed:
        raise RuntimeError("Cannot make new functions after distributing (for now).")

    if not isinstance(inputs, list):
        raise TypeError("Input 'inputs' must be list.")
    bcast_inputs = [] if bcast_inputs is None else bcast_inputs
    if not isinstance(bcast_inputs, list):
        raise TypeError("Input 'bcast_inputs' must be list if not None.")

    gpu_outputs, to_cpu, output_modes = \
        process_outputs(outputs)
    reg_updates, update_vars, slc_update_gpu_outs, update_modes = \
        process_updates(updates)
    reg_givens, slc_givens, slc_idx_inputs, slc_shareds = \
        process_givens(givens, sliceable_shareds)

    theano_function = theano.function(
        inputs=inputs + bcast_inputs,
        outputs=gpu_outputs,  # a list, so function always returns list
        updates=reg_updates,
        givens=reg_givens,
        **kwargs,
    )
    if len(update_outs) == 0 and sliced_shareds is None:
        sliced_function = theano_function
    else:
        sliced_function = theano.function(
            inputs=inputs + bcast_inputs + slc_idx_inputs,
            outputs=gpu_outputs + slc_update_gpu_outs,
            givens=slc_givens,
            **kwargs,
        )

    g.shareds.register_func(theano_function)
    synk_function = Function(ID=len(g.synk_functions),  # Fcn can ID itself
                             theano_function=theano_function,
                             sliced_function=sliced_function,
                             inputs=inputs,
                             bcast_inputs=bcast_inputs,
                             slc_shareds=slc_shareds,
                             update_vars=update_vars,
                             to_cpu=to_cpu,
                             collect_modes=outputs_modes + update_modes,
                             accumulators=g.accumulators,
                             return_list=isinstance(outputs, list),
                             )
    g.synk_functions.append(synk_function)
    return synk_function


###############################################################################
#                                                                             #
#                    Data Management (shared memory)                          #
#                                                                             #
###############################################################################


class Data(DataHelpers):
    """ User will hold some of these: required instead of numpy arrays as
    inputs to functions or to collective communications. """

    def set_value(self, input_data, force_cast=False, oversize=1):
        """ Change data values and length.
        If need be, reshape or reallocate shared memory.
        Oversize only applies to underlying shared memory.  Numpy wrapper will
        be of exact shape of 'input_data'.
        """
        input_data = self._condition_data(input_data, force_cast)
        self._update_array(g.sync.data, input_data.shape, oversize)
        self._data[:] = input_data

    def set_length(self, length, oversize=1):
        length = int(length)
        if length < 1:
            raise ValueError("Length must be a positive integer.")
        shape = list(self.shape)
        shape[0] = length
        self._update_array(g.sync.data, shape, oversize)

    def set_shape(self, shape, oversize=1):
        if len(shape) != self.ndim:
            raise ValueError("Cannot change number of dimensions.")
        self._update_array(g.sync.data, shape, oversize)

    def condition_data(self, input_data, force_cast=False):
        """ See resulting data would be used internally, or raise error. """
        return self._condition_data(input_data, force_cast)

    def free_memory(self):
        """ Removes references in master and workers
        (only way to shrink alloc_size) """
        self._free_shmem()
        g.sync.data.ID.value = self._ID
        self._signal(DATA_FREE)

    def _signal(self, sub_ID):
        exct_in(DATA, sub_ID)
        exct_out()


def data(var_or_arr=None, dtype=None, ndim=None, shape=None,
         scatter=True, minibatch=False,
         force_cast=False, oversize=1, name=None):
    """ Returns a Data object, which is the only type that synkhronos
    functions can receive for Theano inputs.
    """
    if var_or_arr is not None:
        try:
            dtype = var_or_arr.dtype
            ndim = var_or_arr.ndim
        except AttributeError as exc:
            raise exc("Input 'var_or_arr' must have dtype and ndim attributes.")
    elif dtype is None or (ndim is None and shape is None):
        raise TypeError("Must provide either 1) variable or array, or 2) dtype "
            "and either ndim or shape.")
    if shape is not None:
        if ndim is not None:
            if ndim != len(shape):
                raise ValueError("Received inconsistent shape and ndim values.")
        ndim = len(shape)
    synk_data = Data(len(g.scatterer), dtype, ndim, scatter, minibatch, name)
    g.scatterer.append(synk_data)
    g.sync.data.dtype.value = DTYPES.index(dtype)
    g.sync.data.ndim.value = ndim
    g.sync.data.scatter.value = scatter
    exct_in(DATA, DATA_CREATE)  # init in worker, eagerly
    exct_out()  # (must finish before allocating)
    if isinstance(var_or_arr, np.ndarray):
        synk_data.set_value(var_or_arr, force_cast, oversize)
    elif shape is not None:
        synk_data.set_shape(shape, oversize)
    return synk_data


###############################################################################
#                                                                             #
#                      GPU Collectives.                                       #
#                                                                             #
###############################################################################


def gpu_comm_prep(sync_comm, g_shareds, shared_vars, op_ID=None):
    check_active()
    shared_IDs = g_shareds.get_IDs(shared_vars)
    n_shared = len(shared_IDs)
    sync_comm.vars[:n_shared] = shared_IDs
    sync_comm.n_shared.value = n_shared
    if op_ID is not None:
        sync_comm.op.value = op_ID
    return shared_IDs


###############################################################################
#                       User functions                                        #


def broadcast(shared_vars):
    """GPU-comm: broadcast values from master to workers.

    In all multi-variable GPU-comm functions, the default behavior if no
    variables and no functions are provided is to call the operation on all
    shared variables in the session.

    Args:
        shared_vars (None, optional): names or vars to be broadcast
        functions (None, optional): functions to have all shared vars broadcast
    """
    shared_IDs = gpu_comm_prep(g.sync.comm, g.shareds, shared_vars)
    exct_in(GPU_COMM, GPU_BROADCAST)
    for shared_ID in shared_IDs:
        src = g.shareds.get_array(shared_ID)
        g.gpu_comm.broadcast(src)
    exct_out()


def gather(shared_vars, dest=None, nd_up=None):
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
    shared_IDs = gpu_comm_prep(g.sync.comm, g.shareds, shared_vars)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying destination, can only gather one var.")
    exct_in(GPU_COMM, GPU_GATHER)
    results = list()
    for shared_ID in shared_IDs:
        src = g.shareds.get_array(shared_ID)
        r = g.gpu_comm.all_gather(src, dest=dest, nd_up=nd_up)
        results.append(r)
    exct_out()
    if dest is None:
        return results


def reduce(shared_vars, op="avg", in_place=True, dest=None):
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
    op, avg, op_ID = get_op_and_avg(op)
    shared_IDs = gpu_comm_prep(g.sync.comm, g.shareds, shared_vars, op_ID)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying desination, can only reduce one var.")
    if avg and (not in_place or dest is not None):
        raise ValueError("Can only use 'average' op with in-place reduce "
            "(requires None dest).")
    exct_in(GPU_COMM, GPU_REDUCE)
    results = list()
    for shared_ID in shared_IDs:
        src = g.shareds.get_array(shared_ID)
        dest = src if dest is None and in_place else dest
        results.append(g.gpu_comm.reduce(src, op, dest))
    if avg:
        g.shareds.call_avg_fs(shared_IDs)
    exct_out()
    if not in_place and dest is None:  # (otherwise results will be Nones)
        return results


def all_reduce(shared_vars, op="avg"):
    """GPU-comm: master and workers all reduce values, in-place only.

    Args:
        shared_vars (None, optional): names or vars to be reduced
        functions (None, optional): functions to have all shared vars reduced
        op (str, optional): e.g. "sum, prod, min, max, avg"
    """
    op, avg, op_ID = get_op_and_avg(op)
    shared_IDs = gpu_comm_prep(g.sync.comm, g.shareds, shared_vars, op_ID)
    exct_in(GPU_COMM, GPU_ALL_REDUCE)
    for shared_ID in shared_IDs:
        src = g.shareds.get_array(shared_ID)
        g.gpu_comm.all_reduce(src, op, src)
    if avg:
        g.shareds.call_avg_fs(shared_IDs)
    exct_out()


def all_gather(source, dest):
    """GPU-comm: master and workers all gather values into their local vars.

    Only one Theano shared variable can be used for the source, and another
    Theano shared variable of the right shape must already exist for use as the
    destination (since no new shared variables can be created in workers).

    Args:
        source (name or var): shared variable to be gathered
        dest (name or var): shared variable to receive values in
    """
    shared_IDs = gpu_comm_prep(g.sync.comm, g.shareds, [source, dest])
    exct_in(GPU_COMM, GPU_ALL_GATHER)
    src = g.shareds.get_array(shared_IDs[0])
    dest = g.shareds.get_array(shared_IDs[1])
    g.gpu_comm.all_gather(src, dest)
    exct_out()


###############################################################################
#                                                                             #
#                         CPU-based Communications                            #
#                                                                             #
###############################################################################


def scatter(vars_and_data, batch=None):
    """Scatter data and push to master and worker GPU Theano shared variables.

    Input `shared_vars_data` can be either a dictionary, a list, or a single
    variable/name.  If a dictionary, the input is used as in `set_shmems`; the
    data is used to update the shared memory before workers use the values.
    Otherwise, the input is used to determine which Theano shared variables to
    scatter over existing data in shared memory.

    Input parameter `batch` behaves as for function calls; it can limit the
    scatter effect over some subset of the allocated shared memory.


    Args:
        shared_vars_data (TYPE): Shared variables to scatter, optionally with data
        batch (None, optional): int, slice, or list of requested indices

    Raises:
        ValueError: If no input data and shared memory does not exist yet.
    """
    check_active()
    variables = tuple(vars_and_data.keys())  # (establish an order)
    synk_datas = [vars_and_data[var] for var in variables]
    shared_vars = g.shareds.get_vars(variables)
    shared_IDs = g.shareds.get_IDs(shared_vars)
    check_synk_inputs(synk_datas, shared_vars)
    g.scatterer.assign_inputs(synk_datas, batch)
    n_shared = len(shared_IDs)
    g.sync.comm.n_shared.value = n_shared
    g.sync.comm.vars[:n_shared] = shared_IDs
    exct_in(CPU_COMM, SCATTER)
    my_inputs, scatter = g.scatterer.get_my_inputs(n_shared)  # (as in workers)
    if not all(scatter):
        raise TypeError("Must use synk data set to scatter.")
    for var, my_input in zip(shared_vars, my_inputs):
        var.set_value(my_input)
    exct_out()


###############################################################################
#                                                                             #
#                       Initializing and Exiting.                             #
#                                                                             #
###############################################################################


def fork(n_parallel=None, use_gpu=True, master_rank=0, n_in_out=20, n_shared=100, max_dim=10):
    """Fork a python sub-process for each additional GPU and initialize.

    Call this function before building any Theano variables.  (Theano must be
    configured to ipmort to CPU only.)  Initializes one GPU on each process,
    including the master, and initializes GPU collective communications via
    pygpu & NVIDIA NCCL.

    Args:
        n_parallel (None, optional): Number of GPUs to use (default is all)
        master_rank (int, optional): default is 0

    Raises:
        RuntimeError: If already forked or fails to initialize.

    Returns:
        int: number of GPUs using.
    """
    if g.forked:
        raise RuntimeError("Only fork once.")
    from .worker import worker_exct, profiling_worker

    if use_gpu:
        n_parallel = get_n_gpu(n_parallel, master_rank)
    else:
        raise NotImplementedError
    g.shareds.set_n_parallel(n_parallel)
    g.sync = build_sync(n_parallel, n_in_out, n_shared, max_dim)
    g.scatterer = Scatterer(n_parallel, n_in_out, master_rank)

    for rank in [r for r in range(n_parallel) if r != master_rank]:
        args = (rank, n_parallel, master_rank, g.sync, g.scatterer.sync)
        g.processes.append(mp.Process(target=worker_exct, args=args))
    for p in g.processes:
        p.start()

    atexit.register(_close)

    g.cpu_comm = init_cpu_comm(n_parallel, master_rank, g.sync.init)

    if use_gpu:
        g.gpu_comm = init_gpu(master_rank, n_parallel, g.sync)
        if not g.gpu_comm:
            raise RuntimeError("At least one synkhronos worker failed to "
                "initialize GPU during fork.")
        else:
            print("Synkhronos: " + str(n_parallel) + " GPUs succesfully initialized, "
                "master rank: " + str(master_rank))


    g.forked = True
    Function._inv_n = 1 / n_parallel
    return n_parallel


def distribute():
    """Sets up theano functions from master on workers.

    Pickles all theano functions built with this package (i.e. using
    ``synkhronos.function()``) into one file, which workers unpickle.  Theano's
    behavior is to include all shared variable values in the file.  Workers are
    aware of correspondences among input and shared variables used across
    multiple functions.

    In the future, distribution will happen automatically, lazily at the time of
    any function call when it is necessary.  It will remain optional for the
    user to call, as it may be time-consuming.

    The pickle file is automatically deleted by a worker.

    Raises:
        RuntimeError: If not yet forked or if already distributed.
    """
    if not g.forked:
        raise RuntimeError("Need to fork before distributing functions.")

    # Pickle all functions together into one file to preserve correspondences
    # among variables in different functions.
    print("Synkhronos distributing functions...")
    distribution = [sf._get_distro_info for sf in g.synk_functions]
    with open(PKL_FILE, "wb") as f:
        pickle.dump(distribution, f, pickle.HIGHEST_PROTOCOL)
    exct_in(DISTRIBUTE)
    exct_out()
    print("...distribution complete.")
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


###############################################################################
#                           Execution Helpers                                 #


def _close():
    """ Called automatically on exit any time after fork. """
    if g.forked and not g.closed:
        # (try to get workers to exit quietly)
        g.sync.exct.quit.value = True
        try:
            g.sync.exct.barrier_in.wait(1)
        except BrokenBarrierError:
            pass
        for p in g.processes:
            p.join()
        g.closed = True


def check_active():
    if not g.distributed or g.closed:
        raise RuntimeError("Cannot call this function on inactive synkhronos.")


def exct_in(exct_ID, sub_ID=0):
    g.sync.exct.ID.value = exct_ID
    g.sync.exct.sub_ID.value = sub_ID
    g.sync.exct.barrier_in.wait()


def exct_out():
    g.sync.exct.barrier_out.wait()
    if not g.sync.exct.workers_OK.value:
        raise RuntimeError("Encountered worker error during execution loop.")

