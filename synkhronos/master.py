
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

from .variables import struct, Inputs, Shareds, Outputs, BaseFunction, BaseData
from .common import use_gpu, _alloc_scat_idxs, alloc_shared_IDs, get_my_scat_idxs
from .common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                    ALL_GATHER, GATHER, CPU_COMM, SCATTER, DATA, DATA_FREE,
                    DTYPES, DATA_CREATE, DATA_ALLOC, COLLECT_MODES, REDUCE_OPS,
                    REDUCE_OPS_WORKER, REDUCE_AVG, DATA_RESHAPE, NO_COLLECT)
from .util import (get_n_gpu, build_sync, get_op_from_avg,
                   build_collect, check_batch_types, check_collect_vs_op_IDs,
                   build_scat_idxs, oversize_shape,
                   build_collect_IDs, build_reduce_IDs, check_output_subset,
                   check_synk_inputs)

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
    # Theano
    inputs=Inputs(),
    shareds=Shareds(CREATE),
    outputs=Outputs(),
    synk_datas=list(),
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


class Function(BaseFunction):
    """ Class of instances returned by ``synkhronos.function()``.  """

    _create = CREATE
    _n_gpu = None

    def __init__(self, g_inputs, g_outputs, ID, theano_function,
                 input_IDs, output_IDs, collect_IDs, reduce_IDs):
        super().__init__(ID, theano_function)
        self._build_input_desc(g_inputs, input_IDs)
        self._build_output_desc(g_outputs, output_IDs, collect_IDs, reduce_IDs)

    @property
    def theano_function(self):
        """ Read-only: returns the underlying Theano function. """
        return self._f

    @property
    def name(self):
        """ Read-only: will be same name as underlying Theano function. """
        return self._f.name

    @property
    def collect_modes(self):
        """ Read-only: default collect modes for outputs. """
        return [COLLECT_MODES[i] for i in self._output.def_collect_IDs]

    @property
    def reduce_ops(self):
        """ Read-only: default reduce operations for outputs. """
        return [REDUCE_OPS[i] for i in self._output.def_reduce_IDs]

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
        batch = kwargs.pop("batch", None)
        collect_modes = kwargs.pop("collect_modes", None)
        reduce_ops = kwargs.pop("reduce_ops", None)
        self._share_inputs(g.sync.scat, batch, args, kwargs)
        self._share_outputs(output_subset, collect_modes, reduce_ops)
        g.sync.IDs.func.value = self._ID
        exct_in(FUNCTION)
        my_inputs = self._get_my_inputs(g.sync.scat, g.synk_datas)
        my_results = self._f(*my_inputs, output_subset=output_subset)
        results = self._collect_results(g.gpu_comm, my_results)
        exct_out()
        return results

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

    def build_inputs(self, *args, **kwargs):
        """ convenience method which internally calls synkhronos.data() for
        each input variable associated with this function; provide data inputs
        as if calling the Theano function.  All will be scattered.
        # TODO: move force_cast and oversize to function signature?
        """
        force_cast = kwargs.pop("force_cast", False)
        oversize = kwargs.pop("oversize", None)
        inputs = self._order_inputs(args, kwargs)
        synk_datas = list()
        for var_ID, inpt in zip(self._input.IDs, inputs):
            synk_data = data(variable=g.inputs.vars[var_ID],
                             data=inpt,
                             force_cast=force_cast,
                             oversize=oversize)
            synk_datas.append(synk_data)
        return tuple(synk_datas)

    ###########################################################################
    #                     Helpers (not for user)                              #

    def _build_input_desc(self, g_inputs, input_IDs):
        input_order = dict()
        for idx, input_ID in enumerate(input_IDs):
            var = g_inputs.vars[input_ID]
            name = g_inputs.names[input_ID]
            input_order[var] = idx
            if name is not None:
                input_order[name] = idx
        self._input = struct(
            IDs=input_IDs,
            order=input_order,
            dtypes=[g.inputs.vars[i].type.dtype for i in input_IDs],
            ndims=[g.inputs.vars[i].type.ndim for i in input_IDs],
        )

    def _build_output_desc(self, g_outputs, output_IDs, collect_IDs, reduce_IDs):
        self._sync.collect_modes[:] = collect_IDs
        self._sync.reduce_ops[:] = reduce_IDs
        self._output = struct(
            IDs=output_IDs,
            to_cpu=[g.outputs.to_cpu[i] for i in output_IDs],
            avg_funcs=[g.outputs.avg_funcs[i] for i in output_IDs],
            prev_subset=None,
            def_collect_IDs=collect_IDs,
            def_reduce_IDs=reduce_IDs,
            prev_collect_modes=None,
            prev_reduce_ops=None
        )

    def _share_inputs(self, sync_scat, batch, args, kwargs):
        inputs = self._order_inputs(args, kwargs)
        if inputs:
            check_synk_inputs(inputs, self._input.dtypes, self._input.ndims)
            assign_scat_idxs(sync_scat, self._n_gpu, inputs, batch)
            for idx, synk_data in enumerate(inputs):
                self._sync.data_IDs[idx] = synk_data._ID

    def _order_inputs(self, args, kwargs):
        """ Combine args and kwargs into one list of input args. """
        n_args = len(args) + len(kwargs)
        if n_args != self._n_input:
            raise TypeError("Incorrect number of data inputs to function.")
        if n_args == 0:
            return ()
        ordered_inputs = list(args) + [None] * len(kwargs)
        for var, arg in kwargs.items():
            idx = self._input.order.get(var, None)
            if idx is None:
                raise ValueError("Unrecognized keyword var or name: ", var)
            if ordered_inputs[idx] is not None:
                raise ValueError("Redundant input for variable: ", var)
            ordered_inputs[idx] = arg
        return tuple(ordered_inputs)

    def _share_outputs(self, output_subset, collect_modes, reduce_ops):
        if self._n_output > 0:
            self._share_output_subset(output_subset)
            self._share_collect(collect_modes, reduce_ops)

    def _share_output_subset(self, output_subset):
        if output_subset != self._output.prev_subset:
            if output_subset is None:
                self._sync.output_subset[:] = [True] * self._n_output
            else:
                check_output_subset(self._n_output, output_subset)
                self._sync.output_subset[:] = [False] * self._n_output
                for idx in output_subset:
                    self._sync.output_subset[idx] = True
            self._output.prev_subset = output_subset

    def _share_collect(self, collect_modes, reduce_ops):
        if collect_modes != self._output.prev_collect_modes or \
                reduce_ops != self._output.prev_reduce_ops:
            collect_IDs, reduce_IDs = self._build_collect(collect_modes, reduce_ops)
            self._sync.collect_modes[:] = collect_IDs
            self._sync.reduce_ops[:] = reduce_IDs
            self._output.prev_collect_modes = collect_modes
            self._output.prev_reduce_ops = reduce_ops

    def _build_collect(self, collect_modes, reduce_ops):
        if collect_modes is None:
            collect_IDs = self._output.def_collect_IDs
        else:
            collect_IDs = build_collect_IDs(self._n_output, collect_modes)
        if reduce_ops is None:
            reduce_IDs = self._output.def_reduce_IDs
        else:
            reduce_IDs = build_reduce_IDs(self._n_output, reduce_ops)
        reduce_IDs = check_collect_vs_op_IDs(collect_IDs, reduce_IDs)
        return collect_IDs, reduce_IDs

    def _collect_results(self, gpu_comm, my_results):
        if self._n_output == 0:
            return []  # (what a Theano function with no outputs returns)
        output_set = [i for i, x in enumerate(self._sync.output_subset) if x]
        results = list()
        if not isinstance(my_results, (list, tuple)):
            my_results = (my_results,)
        for idx, r in zip(output_set, my_results):
            mode = self._sync.collect_modes[idx]
            if mode == REDUCE:
                op = REDUCE_OPS_WORKER[self._sync.reduce_ops[idx]]  # (no avg)
                gpu_comm.reduce(r, op=op, dest=r)  # (in-place)
            elif mode == GATHER:
                r = gpu_comm.all_gather(r)
            elif mode != NO_COLLECT:
                raise RuntimeError("Unrecognized collect mode in master "
                    "function: ", mode)
            results.append(r)
        for idx_r, idx in enumerate(output_set):
            if self._sync.reduce_ops[idx] == REDUCE_AVG:
                results[idx_r] = self._output.avg_funcs[idx](results[idx_r])
            if self._output.to_cpu[idx]:
                results[idx_r] = np.array(results[idx_r])
        if len(results) == 1:
            results = results[0]
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
        raise RuntimeError("Cannot make new functions after distributing.")

    collect_IDs, reduce_IDs = build_collect(outputs, collect_modes, reduce_ops)
    gpu_outputs, output_IDs = g.outputs.register_set(outputs)
    theano_function = theano.function(inputs, gpu_outputs, **kwargs)
    input_IDs = g.inputs.register_func(theano_function)
    g.shareds.register_func(theano_function)
    synk_function = Function(g_inputs=g.inputs,
                             g_outputs=g.outputs,
                             ID=len(g.synk_functions),  # Fcn can ID itself
                             theano_function=theano_function,
                             input_IDs=input_IDs,
                             output_IDs=output_IDs,
                             collect_IDs=collect_IDs,
                             reduce_IDs=reduce_IDs,
                             )
    g.synk_functions.append(synk_function)
    return synk_function


###############################################################################
#                                                                             #
#                    Data Management (shared memory)                          #
#                                                                             #
###############################################################################


class Data(BaseData):
    """ User will hold some of these: required instead of numpy arrays as
    inputs to functions or to collective communications. """

    _create = True

    ###########################################################################
    #                                  User                                   #

    @property
    def dtype(self):
        """ Read-only """
        return self._dtype

    @property
    def ndim(self):
        """ Read-only """
        return self._ndim

    @property
    def length(self):
        """ Read-only: Current active length along `0-th` dimension
        (not necessarily same as length of numpy array under data attribute).
        """
        return self._length

    @property
    def alloc_size(self):
        """ Number of elements in underlying shared memory allocation. """
        return len(self._alloc_size)

    def set_value(self, input_data, force_cast=False, oversize=None):
        """ Change data values and length.
        If need be, reshape or reallocate shared memory.
        TODO: make oversize only apply to underlying shmem, but numpy array is
        always slice of same shape as input data.
        """
        data = self._condition_data(input_data, force_cast)
        if data.size > self._alloc_size:
            shape = oversize_shape(data, oversize)
            self._alloc_and_signal(g.sync.IDs, shape)
        elif not self._check_shape(data):
            self._reshape_and_signal(g.sync.IDs, data, oversize)
        data_len = len(data)
        self._data[:data_len] = data
        self._length = data_len

    def get_value(self, use_length=True):
        """ Returns the active slice of the numpy array of data. """
        if use_length:
            return self._data[:self._length]
        else:
            return self._data[:]

    def set_length(self, length):
        """ Change the length of the active slice of the numpy array of data.
        Does not change underlying memory allocation.
        """
        length = int(length)
        if self._data is None:
            raise TypeError("Cannot set length on Input without data.")
        elif length > len(self._data):
            raise ValueError("Cannot set length longer than current data.")
        else:
            self._length = length

    def check_input_type(self, input_data, force_cast=False):
        """ See resulting data or receive exception without raising. """
        try:
            data = self._check_input_type(input_data, force_cast)
        except TypeError as exc:
            return exc
        return data

    def free_memory(self):
        """ Removes references in master and workers
        (only way to shrink alloc_size) """
        self._free_shmem()
        g.sync.IDs.data.value = self._ID
        g.sync.IDs.op.value = DATA_FREE
        exct_in(DATA)
        exct_out()

    ###########################################################################
    #                           Helpers                                       #

    def _alloc_and_signal(self, sync_IDs, shape):
        self._tag += 1
        self._alloc_shmem(shape, self._tag)
        if self._ndim > 0:
            sync_IDs.shape[:self._ndim] = shape
        sync_IDs.data.value = self._ID
        sync_IDs.tag.value = self._tag
        sync_IDs.op.value = DATA_ALLOC
        exct_in(DATA)
        exct_out()

    def _reshape_and_signal(self, sync_IDs, data, oversize):
        shape = data.shape
        shape[0] = np.floor(self._alloc_size / np.prod(shape[1:])).astype('int32')
        shape[0] = min(shape[0], oversize * data.shape[0])
        self._reshape_shmem(shape)
        sync_IDs.data.value = self._ID
        sync_IDs.shape[:self._ndim] = shape
        sync_IDs.op.value = DATA_RESHAPE
        exct_in(DATA)
        exct_out()

    def _check_data(self):
        if self._data is not self.data:
            raise RuntimeError("Internal data state inconsistent; data "
                "attribute has been improperly modified in {}.".format(self))

    def _check_shape(self, data):
        shape_OK = False
        if self._data is not None:
            if self._data.shape[1:] == data.shape[1:] and \
                    len(data) <= len(self._data):
                shape_OK = True
        return shape_OK

    def _condition_data(self, input_data, force_cast):
        """ takes in any data and returns numpy array """
        if force_cast:
            if not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data, dtype=self._dtype)
            else:
                input_data = input_data.astype(self.dtype)
        else:
            if not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data)
            if input_data.dtype != self._dtype:
                common_dtype = np.find_common_type([input_data.dtype, self._dtype], [])
                if common_dtype == self._dtype:
                    input_data = input_data.astype(self._dtype)
                else:
                    raise TypeError("Non up-castable data type provided for "
                        "input..., received: {}, expected: {}.  Could use param "
                        "'force_cast=True' to force to expected dtype.".format(
                            input_data.dtype, self._dtype))
        if input_data.ndim != self._ndim:
            raise TypeError("Wrong data ndim provided for input {}, "
                "received: {}, expected: {}".format(self._var, input_data.ndim,
                    self._ndim))
        return input_data


###############################################################################
#                                   User                                      #


def data(variable=None, data=None, scatter=True, force_cast=False,
         oversize=None, dtype=None, ndim=None):
    """ Returns a Data object, which is the only type that synkhronos
    functions can receive for Theano inputs.
    """
    data_ID = len(g.synk_datas)
    if variable is not None:
        if not g.inputs.is_member(variable) and not g.shareds.is_member(variable):
            raise ValueError("Unrecognized input or shared variable: ", variable)
        dtype = variable.type.dtype
        ndim = variable.type.ndim
    if dtype is None or ndim is None:
        if data is None:
            raise TypeError("Must provide variable, or data, or dtype & ndim.")
        np_data = np.array(data)
        dtype = np_data.dtype.name if dtype is None else dtype
        ndim = np_data.ndim if ndim is None else ndim
    synk_data = Data(data_ID, dtype, ndim, bool(scatter))
    g.synk_datas.append(synk_data)
    init_data_worker(g.sync.IDs, dtype, ndim, bool(scatter))
    if data is not None:
        synk_data.set_value(data, force_cast, oversize)  # (checks dtype & ndim)
    return synk_data


###############################################################################
#                           Helpers                                           #


def init_data_worker(sync_IDs, dtype, ndim, scatter):
    sync_IDs.op.value = DATA_CREATE
    sync_IDs.dtype.value = DTYPES.index(dtype)
    sync_IDs.ndim.value = ndim
    sync_IDs.scatter.value = scatter
    exct_in(DATA)
    exct_out()


def alloc_scat_idxs(sync_scat, n_idxs):
    size = int(n_idxs * 1.1)  # (always some extra)
    sync_scat.idxs_tag.value += 1
    sync_scat.idxs_size.value = size
    tag = sync_scat.idxs_tag.value
    sync_scat.idxs_arr = _alloc_scat_idxs(size, tag, CREATE)


def assign_scat_idxs(sync_scat, n_gpu, synk_datas, batch):
    batch = check_batch_types(batch)
    lengths = [synk_data.length for synk_data in synk_datas if synk_data._scatter]
    sync_scat.assign_idxs[:] = build_scat_idxs(n_gpu, lengths, batch)
    if batch is not None and not isinstance(batch, (int, slice)):
        sync_scat.use_idxs_arr.value = True
        n_idxs = len(batch)
        if sync_scat.idxs_arr is None or n_idxs > sync_scat.idxs_arr.size:
            alloc_scat_idxs(sync_scat, n_idxs)  # (will be oversized)
        sync_scat.idxs_arr[:n_idxs] = batch
    else:
        sync_scat.use_idxs_arr.value = False


###############################################################################
#                                                                             #
#                      GPU Collectives.                                       #
#                                                                             #
###############################################################################


def gpu_comm_prep(g_shareds, comm_ID, shared_vars, op_ID=None):
    check_active()
    g.sync.IDs.comm.value = comm_ID
    shared_IDs = g_shareds.get_IDs(shared_vars)
    n_shared = len(shared_IDs)
    g.sync.IDs.vars[:n_shared] = shared_IDs
    g.sync.IDs.n_shared.value = n_shared
    if op_ID is not None:
        g.sync.IDs.op.value = op_ID
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
    shared_IDs = gpu_comm_prep(g.shareds, BROADCAST, shared_vars)
    exct_in(GPU_COMM)
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
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
    shared_IDs = gpu_comm_prep(g.shareds, GATHER, shared_vars)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying destination, can only gather one var.")
    exct_in(GPU_COMM)
    results = list()
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
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
    op, avg, op_ID = get_op_from_avg(op)
    shared_IDs = gpu_comm_prep(g.shareds, REDUCE, shared_vars, op_ID)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying desination, can only reduce one var.")
    if avg and (not in_place or dest is not None):
        raise ValueError("Can only use 'average' op with in-place reduce "
            "(requires None dest).")
    exct_in(GPU_COMM)
    results = list()
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
        dest = src if dest is None and in_place else dest
        results.append(g.gpu_comm.reduce(src, op, dest))
    if avg:
        for shared_ID in shared_IDs:
            g.shareds.avg_funcs[shared_ID]()
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
    op, avg, op_ID = get_op_from_avg(op)
    shared_IDs = gpu_comm_prep(g.shareds, ALL_REDUCE, shared_vars, op_ID)
    exct_in(GPU_COMM)
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
        g.gpu_comm.all_reduce(src, op, src)
    if avg:
        for shared_ID in shared_IDs:
            g.shareds.avg_funcs[shared_ID]()
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
    shared_IDs = gpu_comm_prep(g.shareds, ALL_GATHER, [source, dest])
    exct_in(GPU_COMM)
    src = g.shareds.get_gpuarray(shared_IDs[0])
    dest = g.shareds.get_gpuarray(shared_IDs[1])
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
    shared_IDs = g.shareds.get_IDs(variables)
    dtypes = [g.shareds.vars[var_ID].type.dtype for var_ID in shared_IDs]
    ndims = [g.shareds.vars[var_ID].type.ndim for var_ID in shared_IDs]
    check_synk_inputs(synk_datas, dtypes, ndims)
    assign_scat_idxs(g.sync.scat, g.n_gpu, synk_datas, batch)
    n_shared = len(shared_IDs)
    g.sync.IDs.comm.value = SCATTER
    g.sync.IDs.n_shared.value = n_shared
    g.sync.IDs.vars[:n_shared] = shared_IDs
    g.sync.IDs.datas[:n_shared] = [data._ID for data in synk_datas]
    exct_in(CPU_COMM)
    # Master sets its portion just like workers.
    my_idxs = get_my_scat_idxs(g.sync.scat, g.master_rank)
    for shared_ID, synk_data in zip(shared_IDs, synk_datas):
        g.shareds.vars[shared_ID].set_value(synk_data._data[my_idxs])
    exct_out()


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
    from .worker import worker_exct

    n_gpu, master_rank = get_n_gpu(n_gpu, master_rank)
    sync = build_sync(n_gpu)

    for rank in [r for r in range(n_gpu) if r != master_rank]:
        args = (rank, n_gpu, master_rank, sync)
        g.processes.append(mp.Process(target=worker_exct, args=args))
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
    g.sync.IDs.vars, g.sync.IDs.datas = \
        alloc_shared_IDs(len(g.shareds.vars), CREATE)
    g.sync.init.n_user_fcns.value = len(g.synk_functions)

    # Signal workers to receive.
    g.sync.init.distributed.value = True
    g.sync.init.barriers.distribute.wait()

    g.outputs.set_avg_facs(g.n_gpu)
    g.sync.init.barriers.distribute_out.wait()
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
        if not g.sync.init.distributed.value:
            try:
                g.sync.init.barriers.distribute.wait(1)
            except BrokenBarrierError:
                pass
        else:
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


def exct_in(exct_ID):
    g.sync.exct.ID.value = exct_ID
    g.sync.exct.barrier_in.wait()


def exct_out():
    g.sync.exct.barrier_out.wait()
    if not g.sync.exct.workers_OK.value:
        raise RuntimeError("Encountered worker error during execution loop.")


###############################################################################
#                                                                             #
#                            Data Management                                  #
#                                                                             #
###############################################################################




