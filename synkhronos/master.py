
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
                    ALL_GATHER, GATHER, CPU_COMM, AVG_ALIASES, SCATTER,
                    SCAT_IDXS_TAG)
from .util import (get_n_gpu, build_sync, check_collect, check_op,
                   get_worker_reduce_ops, check_batch_types, get_shared_IDs,
                   _assign_scat_idxs)
from .shmemarray import NpShmemArray


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
    inputs=Inputs(True),
    shareds=Shareds(True),
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
    """ Class of instances returned by ``synkhronos.function()``.  """

    _create = True
    _n_gpu = None
    _rank = None  # (is also master_rank)

    def __init__(self, shared_IDs, output_IDs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_IDs = shared_IDs
        self._output_IDs = output_IDs
        self._name = self._theano_function.name
        self._build_sync()

        # For streamlining some helper operations.
        # (Using actual globals here so it doesn't look like they are being
        # attached when instantiating a function.)
        self._input_names = [g.inputs.names[i] for i in self._input_IDs]
        self._input_vars = [g.inputs.vars[i] for i in self._input_IDs]
        self._output_to_cpu = [g.outputs.to_cpu[i] for i in self._output_IDs]
        self._output_avg_funcs = [g.outputs.avg_funcs[i] for i in self._output_IDs]
        self._previous_batch_size = None
        self._my_slice = None
        self._previous_output_subset = None
        self._n_inputs = len(self._input_IDs)
        self._previous_space_start = None
        self._previous_space_end = None

    @property
    def name(self):
        """ Read-only: will be same name as underlying Theano function. """
        return self._name

    @property
    def output_to_cpu(self):
        """ Ready-only: lists whether outputs are returned to CPU. """
        return self._output_to_cpu

    ###########################################################################
    #                       User callables (use globals g directly)           #

    def __call__(self, *args, **kwargs):
        """ Callable as in Theano function.

        When called, Synkhronos functions:

            1. Share input data,
            2. Signal to workers to start and what to do,
            3. Call the local theano function on assigned data subset,
            4. Collect results from workers and return it.

        If an input array is the input shared memory array used internally for
        that variable (as returned by the ``get_input_shmems()`` method), this
        function will recognize that and will use the shared memories directly,
        avoiding the additional memory copy.  To use only a subset of the total
        batch size (`0-th` dimension) allocated in the shared memory, pass a
        contiguous slice of the shared memory array starting at the beginning,
        e.g. ``shmem[:new_batch_size]``.

        Theano function keyword argument ``output_subset`` is supported.

        Args:
            *args (data): Normal data inputs to Theano function
            **kwargs (data): Normal data inputs to Theano function

        Raises:
            RuntimeError: If not distributed or if synkhronos closed.
        """
        check_active()
        output_subset = kwargs.pop("output_subset", None)
        oversize = kwargs.pop("oversize", None)
        batch = kwargs.pop("batch", None)
        batch = check_batch_types(batch)
        organized_inputs = self._organize_inputs(g.inputs, args, kwargs)
        if organized_inputs:
            g.inputs.update_shmems(organized_inputs, oversize=oversize)
        assign_scat_idxs(g.sync, g.n_gpu, g.inputs, self._input_IDs, batch)
        output_set = self._share_output_subset(output_subset)
        g.sync.exec_ID.value = FUNCTION
        g.sync.func_ID.value = self._ID
        g.sync.barriers.exec_in.wait()
        my_inputs = self._get_my_inputs(g.sync, g.inputs)
        my_results = self._call_theano_function(my_inputs, output_subset)  # always a list

        results = self._collect_results(g.gpu_comm, my_results, output_set)  # always returns list
        exec_out_check(g.sync)
        if len(results) == 1:
            results = results[0]
        return results

    def set_input_shmems(self, *args, **kwargs):
        """Update internal shared memory arrays used for inputs.

        A full set of function arguments must be provided.  New shared memory
        will be allocated only if necessary.  This is like calling
        synkhronos.set_shmems() on all input variables associated with this
        function.

        Optional param 'oversize' can be used to allocate shared memory at up to
        2x the size (along `0-th` dimension) of the input values.  This only
        applies if the input values do not fit in the current allocation.

        Args:
            *args (data): Normal data inputs to the Theano function
            **kwargs (data): Normal data inputs to the Theano function
            oversize (None, optional): factor in the range [1, 2]

        Raises:
            RuntimeError: If functions not distributed or if synkhronos closed.

        Returns:
            Dict: Numpy-wrapped shared memory arrays under keys of input vars.
        """
        check_active()
        oversize = kwargs.pop('oversize', None)
        organized_inputs = self._organize_inputs(g.inputs, args, kwargs)
        shmems = g.inputs.update_shmems(organized_inputs, oversize=oversize)
        return shmems

    def get_input_shmems(self):
        """Get internal shared memory arrays used for inputs.

        This method returns the current share memory arrays used internally by
        the function to communicate input data to workers.  Each variable's
        memory is wrapped in a numpy array.  A list is return in the order of
        the function's inputs.

        Raises:
            RuntimeError: If synkhronos not distributed or closed.
        """
        check_active()
        shmems = list()
        for input_ID in self._input_IDs:
            shmems.append(g.inputs.shmems[input_ID])
        return shmems

    def as_theano(self, *args, **kwargs):
        """Call the function in the master process only, as normal Theano.

        This method will return outputs to the CPU if they were originally
        requested there, unlike using ``function.theano_function()``, which is
        built to hold all outputs on the GPU.

        Args:
            *args (data): Normal data inputs to the Theano function
            **kwargs (data): Normal data inputs to the Theano function
        """
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

    def _organize_inputs(self, g_inputs, args, kwargs):
        """ Returns a dict which can be used on variable class update_shmems()"""
        n_args = len(args) + len(kwargs)
        if n_args == 0:
            for input_ID in self._input_IDs:
                if g_inputs.shmems[input_ID] is None:
                    raise ValueError("Called synkhronos function with no input "
                        "data, but input shared memory does not yet exist.")
            return dict()
        elif n_args != self._n_inputs:
            raise TypeError("Incorrect number of inputs to synkhronos function "
                "(must supply all or none).")
        for var in kwargs.keys():
            if var not in self._input_vars and var not in self._input_names:
                raise ValueError("Unrecognized input for function: ", var)
        organized_inputs = kwargs
        for idx, input_data in enumerate(args):
            var = g_inputs.vars[self._input_IDs[idx]]
            name = g_inputs.names[self._input_IDs[idx]]
            if var in kwargs:
                raise ValueError("Redundant input: ", var)
            elif name is not None and name in kwargs:
                raise ValueError("Redundant input: ", name)
            else:
                organized_inputs[var] = input_data
        return organized_inputs  # (dict of the raw input data)

    def _share_output_subset(self, output_subset):
        # TODO: update either this or build_sync to make them match.
        if output_subset != self._previous_output_subset:
            if output_subset is None:
                self._sync.output_subset[:] = [True] * self._n_outputs
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
                self._sync.output_subset[:] = [False] * self._n_outputs
                for idx in output_subset:
                    self._sync.output_subset[idx] = True
            self._previous_output_subset = output_subset
        output_set = [i for i, x in enumerate(self._sync.output_subset) if x]
        return output_set

    def _get_my_inputs(self, sync, g_inputs):
        my_idxs = slice(*sync.scat.assign_idxs[self._rank:self._rank + 2])
        if sync.scat.use_idxs_arr.value:
            my_idxs = sync.scat.idxs_arr[my_idxs]
        my_inputs = list()
        for input_ID in self._input_IDs:
            my_inputs.append(g_inputs.shmems[input_ID][my_idxs])
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
        collect_modes (str, list, optional): default behaviors; "gather" or "reduce"
        reduce_ops (str, list, optional): default behaviors; e.g. "sum", "prod", "min", "max", "avg"
        **kwargs (TYPE): passed directly to ``theano.function()``

    Raises:
        RuntimeError: If not yet forked or if already distributed.

    Returns:
        SynkFunction: Callable like a Theano function.
    """
    if not g.forked:
        raise RuntimeError("Must fork before making functions for GPU.")
    if g.distributed:
        raise RuntimeError("Cannot make new functions after distributing.")

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
                             collect_modes=collect_modes,
                             reduce_ops=reduce_ops,
                             )
    g.synk_functions.append(synk_function)
    return synk_function


###############################################################################
#                                                                             #
#                        Shared Memory Management                             #
#                                                                             #
###############################################################################


def get_shmems(*variables):
    """Get the shared memory arrays currently in use for Theano variables.

    Can be used for input variables or Theano shared variables.

    The shared memory arrays are numpy-wrapped, so can be efficiently written to
    using slice notation, e.g. shmem[:] = new_data.

    Args:
        *variables (vars and/or names): Variables to be retrieved.

    Returns:
        Dict: shared memory arrays under keys provided.
    """
    check_active()
    if len(variables) == 1 and isinstance(variables[0], (list, tuple)):
        variables = variables[0]
    input_vars, shared_vars = segregate_input_shared(variables)
    input_shmems = g.inputs.get_shmems_vars(input_vars)
    shared_shmems = g.shareds.get_shmems_vars(shared_vars)
    shmems = dict()
    for shmem, var in zip(input_shmems, input_vars):
        shmems[var] = shmem
    for shmem, var in zip(shared_shmems, shared_vars):
        shmems[var] = shmem
    return shmems


def set_shmems(vars_data, oversize=None):
    """Update the shared memory arrays used for Theano variables.

    Inputs should be provided in a dictionary where each key is a Theano
    variable instance or name (shared or input) and the value is the data, to be
    interpreted as a single array to be scattered along the 0-th dimension.

    If the input data is a contiguous slice of the existing shared memory
    starting at its beginning (e.g. shmem[:n]), then this function does nothing.
    If the input data is new memory, and the previously allocated shared memory
    is already the exact shape in all dimensions past `0-th` and is large enough
    (in the `0-th dimension`), then the input data will be copied into the
    existing shared memory starting at its beginning.  Otherwise, a new shared
    memory array will be allocated.

    Optional param 'oversize' can be used to allocate shared memory at up to
    2x the size (along `0-th` dimension) of the input values.  This only
    applies if the input values do not fit in the current allocation.

    Args:
        vars_data (dict): variables as keys and data as values
        oversize (None, optional): factor in range [1, 2]

    Returns:
        Dict: Numpy-wrapped shared memory arrays under the keys provided.
    """
    check_active()
    all_vars = tuple(vars_data.keys())
    input_vars, shared_vars = segregate_input_shared(all_vars)
    shmems = g.inputs.update_shmems(vars_data, input_vars, oversize)
    shared_shmems = g.shareds.update_shmems(vars_data, shared_vars, oversize)
    for k, v in shared_shmems.items():
        shmems[k] = v
    return shmems


def free_shmems(*input_vars):
    """
    Will eliminate all internal references to a shared array (including in
    workers) so it can be freed.  Also useful to shrink shmem.

    Warning:
        Not Implemented.
    """
    check_active()
    raise NotImplementedError


###############################################################################
#                           Helpers                                           #


def segregate_input_shared(variables):
    input_vars = list()
    shared_vars = list()
    for idx, var in enumerate(variables):
        if g.inputs.is_member(var):
            input_vars.append(var)
        elif g.shareds.is_member(var):
            shared_vars.append(var)
        else:
            raise ValueError("Unrecognized variable instance or name: ", var)
    return input_vars, shared_vars


def assign_scat_idxs(sync, n_gpu, g_vars, var_IDs, batch):
    """ Used in functions and in scatter collective """
    sizes = [g_vars.sync.shapes[var_ID][0] for var_ID in var_IDs]
    sync.scat.assign_idxs[:] = _assign_scat_idxs(n_gpu, sizes, batch)
    if not isinstance(batch, (int, slice)):
        sync.scat.use_idxs_arr.value = True
        n_idxs = len(batch)
        if sync.scat.idxs_arr is None or n_idxs > sync.scat.idxs_arr.size:
            alloc_scat_idxs(sync, n_idxs)  # (will be oversized)
        sync.scat.idxs_arr[:n_idxs] = batch
    else:
        sync.scat.use_idxs_arr.value = False


def alloc_scat_idxs(sync, n_idxs):
    size = int(n_idxs * 1.1)  # (always some extra)
    sync.scat.idxs_tag.value += 1
    sync.scat.idxs_size.value = size
    tag = SCAT_IDXS_TAG + str(sync.scat.idxs_tag.value)
    sync.scat.idxs_arr = NpShmemArray('int32', size, tag)


###############################################################################
#                                                                             #
#                      GPU Collectives.                                       #
#                                                                             #
###############################################################################


def gpu_comm_prep(comm_ID, shared_vars=None, functions=None,
                  has_op=False, op=None):
    """ Not called by user but using direct globals access to streamline. """
    check_active()
    g.sync.exec_ID.value = GPU_COMM
    g.sync.comm_ID.value = comm_ID
    shared_IDs = get_shared_IDs(g.shareds, shared_vars, functions)
    shared_IDs = tuple(range(g.shareds.num)) if not shared_IDs else shared_IDs
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
    shared_IDs = gpu_comm_prep(BROADCAST, shared_vars, functions)
    g.sync.barriers.exec_in.wait()
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
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
    shared_IDs = gpu_comm_prep(GATHER, shared_vars, functions)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying destination, can only gather one var.")
    g.sync.barriers.exec_in.wait()
    results = list()
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
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
    shared_IDs, op, avg = gpu_comm_prep(REDUCE, shared_vars, functions, True, op)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying desination, can only reduce one var.")
    if avg and (not in_place or dest is not None):
        raise ValueError("Can only use 'average' op with in-place reduce "
            "(requires None dest).")
    g.sync.barriers.exec_in.wait()
    results = list()
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
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
        gpu_comm_prep(ALL_REDUCE, shared_vars, functions, True, op)
    g.sync.barriers.exec_in.wait()
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
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
    src = g.shareds.get_gpuarray(shared_IDs[0])
    dest = g.shareds.get_gpuarray(shared_IDs[1])
    g.gpu_comm.all_gather(src, dest)
    exec_out_check(g.sync)


###############################################################################
#                                                                             #
#                         CPU-based Communications                            #
#                                                                             #
###############################################################################


def scatter(shared_vars_data, batch=None):
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
    batch = check_batch_types(batch)
    shared_IDs = g.shareds.get_IDs(shared_vars_data)
    if isinstance(shared_vars_data, dict):
        g.shareds.update_shmems(shared_vars_data)
    else:
        for shared_ID, var in zip(shared_IDs, shared_vars_data):
            if g.shareds.shmems[shared_ID] is None:
                raise ValueError("Called scatter with no input data, but shared "
                    "memory does not exist yet for variable: ", var)
    assign_scat_idxs(g.sync, g.n_gpu, g.shareds, shared_IDs, batch)

    n_shared = len(shared_IDs)
    g.sync.exec_ID.value = CPU_COMM
    g.sync.comm_ID.value = SCATTER
    g.sync.n_shared.value = n_shared
    g.shareds.sync.shared_IDs[:n_shared] = shared_IDs
    g.sync.barriers.exec_in.wait()

    # Master sets its portion just like workers.
    my_idxs = slice(*g.sync.scat.assign_idx[g.master_rank:g.master_rank + 2])
    if g.sync.scat.use_idxs.value:
        my_idxs = g.sync.scat.idxs_arr[my_idxs]
    for shared_ID in shared_IDs:
        g.shareds.vars[shared_ID].set_value(g.shareds.shmems[shared_ID][my_idxs])

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
    Function._rank = master_rank

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
    g.inputs.build_sync()
    g.shareds.build_sync()
    g.sync.n_user_fcns.value = len(g.synk_functions)
    g.sync.dict["collect_modes"] = [fn._collect_modes for fn in g.synk_functions]
    g.sync.dict["reduce_ops"] = get_worker_reduce_ops(g.synk_functions)

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


###############################################################################
#                           Helpers                                           #


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


def check_active():
    if not g.distributed or g.closed:
        raise RuntimeError("Cannot call this function on inactive synkhronos.")
