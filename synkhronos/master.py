
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

from .variables import (struct, Shareds, Outputs, BaseFunction, BaseData,
                        BaseScatterer)
from .common import use_gpu
from .common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                    ALL_GATHER, GATHER, CPU_COMM, SCATTER, DATA, DATA_FREE,
                    DTYPES, DATA_CREATE, DATA_ALLOC, COLLECT_MODES, REDUCE_OPS,
                    REDUCE_OPS_WORKER, REDUCE_AVG, DATA_RESHAPE, NO_COLLECT)
from .util import (get_n_gpu, build_sync, get_op_and_avg, build_scat_idxs,
                   build_def_collect, check_collect_vs_op_IDs, check_batch_types,
                   build_collect_IDs, build_reduce_IDs, check_output_subset,
                   check_synk_inputs, build_sync_scat)
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
    shareds=Shareds(),
    outputs=Outputs(),
    # GPU
    synk_functions=list(),
    accumulators=Accumulators(),
    gpu_comm=None,
)


###############################################################################
#                                                                             #
#                           Building Functions.                               #
#                                                                             #
###############################################################################


class Function(BaseFunction):
    """ Class of instances returned by ``synkhronos.function()``.  """

    _create = CREATE
    _inv_n_gpu = None

    def __init__(self, ID, theano_function,
                 synk_outputs, collect_IDs, reduce_IDs):
        super().__init__(ID, theano_function)
        self._build_input_desc()
        self._build_output_desc(synk_outputs, collect_IDs, reduce_IDs)

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
        num_slices = kwargs.pop("num_slices", 1)
        self._share_inputs(g.scatterer, batch, args, kwargs)
        new_collect = self._share_f_info(g.sync.func, num_slices,
            output_subset, collect_modes, reduce_ops)
        if num_slices > 1 and new_collect:  # (before barrier, in master)
            self._set_accum_fs(g.accumulators)
        exct_in(FUNCTION, self._ID)
        my_inputs, scatter = g.scatterer.get_my_inputs(self._n_input)
        if num_slices > 1 and any(scatter):
            my_results = \
                self._sliced_f(my_inputs, scatter, num_slices, output_subset)
        else:
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
        force_cast = kwargs.pop("force_cast", False)
        oversize = kwargs.pop("oversize", None)
        scatter = kwargs.pop("scatter", False)
        inputs = self._order_inputs(args, kwargs)
        if not isinstance(scatter, (list, tuple)):
            scatter = [scatter] * self._n_input
        elif len(scatter) != self._n_input:
            raise ValueError("Scatter must be single boolean or list/tuple of "
                "length equal to the number of inputs.")
        synk_datas = list()
        for var, inpt, scat in zip(self._input.vars, inputs, scatter):
            synk_data = data(variable=var,
                             data=inpt,
                             scatter=scat,
                             force_cast=force_cast,
                             oversize=oversize)
            g.scatterer.append(synk_data)
        return tuple(synk_datas)

    ###########################################################################
    #                     Helpers (not for user)                              #

    def _build_input_desc(self):
        input_orderer = dict()
        inputs = [i.variable for i in self._f.maker.inputs if not i.implicit]
        for idx, var in enumerate(inputs):
            input_orderer[var] = idx
            if var.name is not None:
                input_orderer[var.name] = idx
        self._input = struct(
            vars=inputs,
            orderer=input_orderer,
        )

    def _build_output_desc(self, synk_outputs, collect_IDs, reduce_IDs):
        self._output = struct(
            to_cpu=[out.to_cpu for out in synk_outputs],
            avg_fs=[out.avg_f for out in synk_outputs],
            full_set=list(range(self._n_output)),
            prev_subset=-1,  # (invalid value so not same as first call)
            def_collect_IDs=collect_IDs,
            def_reduce_IDs=reduce_IDs,
            prev_collect_modes=None,
            prev_reduce_ops=None,
        )

    def _share_inputs(self, scatterer, batch, args, kwargs):
        synk_inputs = self._order_inputs(args, kwargs)
        if synk_inputs:
            check_synk_inputs(synk_inputs, self._input.vars)
            scatterer.assign_inputs(synk_inputs, batch)

    def _order_inputs(self, args, kwargs):
        """ Combine args and kwargs into one list of input args. """
        n_args = len(args) + len(kwargs)
        if n_args != self._n_input:
            raise TypeError("Incorrect number of data inputs to function.")
        if n_args == 0:
            return ()
        ordered_inputs = list(args) + [None] * len(kwargs)
        for var, arg in kwargs.items():
            idx = self._input.orderer.get(var, None)
            if idx is None:
                raise ValueError("Unrecognized keyword var or name: ", var)
            if ordered_inputs[idx] is not None:
                raise ValueError("Redundant input for variable: ", var)
            ordered_inputs[idx] = arg
        return tuple(ordered_inputs)

    def _share_f_info(self, sync_func, num_slices,
                      output_subset, collect_modes, reduce_ops):
        if self._n_input > 0:
            num_slices = int(num_slices)
            if num_slices < 1:
                raise ValueError("Invalid number of slices: ", num_slices)
            sync_func.n_slices.value = num_slices
        if self._n_output > 0:
            new_subset = self._share_output_subset(sync_func, output_subset)
            new_collect = self._share_collect(sync_func, collect_modes, reduce_ops)
            return new_subset or new_collect

    def _share_output_subset(self, sync_func, output_subset):
        is_new_subset = output_subset != self._output.prev_subset
        if is_new_subset:
            if output_subset is None:
                self._output_set = self._output.full_set
            else:
                check_output_subset(self._n_output, output_subset)
                self._output_set = list()
                for i in output_subset:
                    self._output_set.append(i)
            self._output.prev_subset = output_subset
        if output_subset is None:
            for i in range(self._n_output):
                sync_func.output_subset[i] = True
        else:
            for i in range(self._n_output):
                sync_func.output_subset[i] = False
            for i in output_subset:
                sync_func.output_subset[i] = True
        return is_new_subset

    def _share_collect(self, sync_func, collect_modes, reduce_ops):
        is_new_collect = collect_modes != self._output.prev_collect_modes or \
            reduce_ops != self._output.prev_reduce_ops
        if is_new_collect:
            self._output.prev_collect_modes = collect_modes
            self._output.prev_reduce_ops = reduce_ops
            collect_IDs, op_IDs = self._build_collect(collect_modes, reduce_ops)
            self._collects = collect_IDs[self._output_set]
            self._ops = op_IDs[self._output_set]
        for i, (mode_ID, op_ID) in enumerate(zip(self._collects, self._ops)):
            sync_func.collect_modes[i] = mode_ID  # (by position in output_set)
            sync_func.reduce_ops[i] = op_ID
        return is_new_collect

    def _build_collect(self, collect_modes, reduce_ops):
        if collect_modes is None and reduce_ops is None:
            return self._output.def_collect_IDs, self._output.def_reduce_IDs
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
        results = list()
        if not isinstance(my_results, (list, tuple)):
            my_results = (my_results,)
        for r, mode_ID, op_ID in zip(my_results, self._collects, self._ops):
            if mode_ID == REDUCE:
                op = REDUCE_OPS_WORKER[op_ID]  # (no avg)
                gpu_comm.reduce(r, op=op, dest=r)  # (in-place)
            elif mode_ID == GATHER:
                r = gpu_comm.all_gather(r)
            elif mode_ID != NO_COLLECT:
                raise RuntimeError("Unrecognized collect mode in master "
                    "function: ", mode_ID)
            results.append(r)
        for i_r, (r, i_o, op_ID) in \
                enumerate(zip(results, self._output_set, self._ops)):
            if op_ID == REDUCE_AVG:
                results[i_r] = self._output.avg_funcs[i_o](r, self._inv_n_gpu)
            if self._output.to_cpu[i_o]:
                results[i_r] = np.array(results[i_r])
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

    collect_IDs, reduce_IDs = build_def_collect(outputs, collect_modes, reduce_ops)
    synk_outputs, gpu_outputs = g.outputs.register_set(outputs, g.accumulators)
    theano_function = theano.function(inputs, gpu_outputs, **kwargs)
    g.shareds.register_func(theano_function, g.accumulators)
    synk_function = Function(ID=len(g.synk_functions),  # Fcn can ID itself
                             theano_function=theano_function,
                             outputs=synk_outputs,
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

    def __len__(self):
        return 0 if self._data is None else len(self._data)

    @property
    def dtype(self):
        """ Read-only """
        return self._dtype

    @property
    def ndim(self):
        """ Read-only """
        return self._ndim

    @property
    def alloc_size(self):
        """ Number of elements in underlying shared memory allocation. """
        return len(self._alloc_size)

    def set_value(self, input_data, force_cast=False, oversize=1):
        """ Change data values and length.
        If need be, reshape or reallocate shared memory.
        Oversize only applies to underlying shared memory.  Numpy wrapper will
        be of exact shape of 'input_data'.
        """
        data = self._condition_data(input_data, force_cast)
        if data.size > self._alloc_size:
            self._alloc_and_signal(g.sync.data, data.shape, float(oversize))
        elif self._data.shape != data.shape:
            self._shape_and_signal(g.sync.data, data.shape)
        self._data[:] = data

    def condition_data(self, input_data, force_cast=False):
        """ See resulting data would be used internally, or raise error. """
        return self._condition_data(input_data, force_cast)

    def free_memory(self):
        """ Removes references in master and workers
        (only way to shrink alloc_size) """
        self._free_shmem()
        g.sync.data.ID.value = self._ID
        exct_in(DATA, DATA_FREE)
        exct_out()

    ###########################################################################
    #                           Helpers                                       #

    def _alloc_and_signal(self, sync_data, shape, oversize):
        self._tag += 1
        if oversize < 1 or oversize > 2:
            raise ValueError("param 'oversize' must be in range [1, 2].")
        size = int(np.prod(shape) * oversize)
        self._alloc_shmem(size, self._tag)
        if self._ndim > 0:
            sync_data.shape[:self._ndim] = shape
        sync_data.tag.value = self._tag
        exct_in(DATA, DATA_ALLOC)
        self._shape_data(shape)
        exct_out()

    def _shape_and_signal(self, sync_data, shape):
        sync_data.ID.value = self._ID
        if self._ndim > 0:
            sync_data.shape[:self._ndim] = shape
        exct_in(DATA, DATA_RESHAPE)
        self._shape_data(shape)
        exct_out()

    def _check_data(self):
        if self._data is not self.data:
            raise RuntimeError("Internal data state inconsistent; data "
                "attribute has been improperly modified in {}.".format(self))

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
    if variable is not None:
        if not isinstance(variable, (theano.tensor.TensorVariable,
                                     theano.compile.sharedvalue.SharedVariable)):
            raise TypeError("Optional param 'variable' must be type "
                "TensorVariable or SharedVariable.")
        dtype = variable.dtype
        ndim = variable.ndim
    if dtype is None or ndim is None:
        if data is None:
            raise TypeError("Must provide variable, or data, or dtype & ndim.")
        np_data = np.array(data)
        dtype = np_data.dtype.name if dtype is None else dtype
        ndim = np_data.ndim if ndim is None else ndim
    synk_data = Data(dtype, ndim, bool(scatter))
    g.scatterer.append(synk_data)
    g.sync.data.dtype.value = DTYPES.index(dtype)
    g.sync.data.ndim.value = ndim
    g.sync.data.scatter.value = bool(scatter)
    exct_in(DATA, DATA_CREATE)  # init in worker, eagerly
    exct_out()
    if data is not None:
        synk_data.set_value(data, force_cast, oversize)  # (checks dtype & ndim)
    return synk_data


###############################################################################
#                            Scatterer.                                       #


class Scatterer(BaseScatterer):

    create = True

    def __init__(self, n_gpu, n_in_out, rank):
        self.sync = build_sync_scat(n_gpu, n_in_out)
        super().__init__(n_gpu, rank)

    def assign_inputs(self, synk_datas, batch):
        batch = check_batch_types(batch)
        lengths = [len(sd) for sd in synk_datas if sd._scatter]
        self.sync.assign_idxs[:] = build_scat_idxs(self.n_gpu, lengths, batch)
        if batch is not None and not isinstance(batch, (int, slice)):
            self.sync.use_idxs_arr.value = True
            n_idxs = len(batch)
            if self.sync.idxs_arr is None or n_idxs > self.sync.idxs_arr.size:
                self.alloc_idxs_arr(n_idxs)  # (will be oversized)
            self.sync.idxs_arr[:n_idxs] = batch
        else:
            self.sync.use_idxs_arr.value = False
        for i, synk_data in enumerate(synk_datas):
            self.sync.data_IDs[i] = synk_data._ID

    def alloc_idxs_arr(self, n_idxs):
        size = int(n_idxs * 1.1)  # (always some extra)
        self.sync.idxs_tag.value += 1
        self.sync.idxs_size.value = size
        tag = self.sync.idxs_tag.value
        self._alloc_idxs_arr(size, tag)


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
    exct_in(GPU_COMM, BROADCAST)
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
    shared_IDs = gpu_comm_prep(g.sync.comm, g.shareds, shared_vars)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying destination, can only gather one var.")
    exct_in(GPU_COMM, GATHER)
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
    op, avg, op_ID = get_op_and_avg(op)
    shared_IDs = gpu_comm_prep(g.sync.comm, g.shareds, shared_vars, op_ID)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying desination, can only reduce one var.")
    if avg and (not in_place or dest is not None):
        raise ValueError("Can only use 'average' op with in-place reduce "
            "(requires None dest).")
    exct_in(GPU_COMM, REDUCE)
    results = list()
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
        dest = src if dest is None and in_place else dest
        results.append(g.gpu_comm.reduce(src, op, dest))
    if avg:
        g.shareds.call_avg_funcs(shared_IDs)
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
    exct_in(GPU_COMM, ALL_REDUCE)
    for shared_ID in shared_IDs:
        src = g.shareds.get_gpuarray(shared_ID)
        g.gpu_comm.all_reduce(src, op, src)
    if avg:
        g.shareds.call_avg_funcs(shared_IDs)
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
    exct_in(GPU_COMM, ALL_GATHER)
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


def fork(n_gpu=None, master_rank=0, n_in_out=20, n_shared=100):
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
    g.shareds.set_n_gpu(n_gpu)
    g.sync = build_sync(n_gpu, n_in_out, n_shared)
    g.scatterer = Scatterer(n_gpu, n_in_out, master_rank)

    for rank in [r for r in range(n_gpu) if r != master_rank]:
        args = (rank, n_gpu, master_rank, g.sync, g.scatterer.sync)
        g.processes.append(mp.Process(target=worker_exct, args=args))
    for p in g.processes:
        p.start()

    atexit.register(_close)

    gpu_comm = use_gpu(master_rank, n_gpu, g.sync)
    if not gpu_comm:
        raise RuntimeError("At least one synkhronos worker failed to "
            "initialize GPU during fork.")
    else:
        print("Synkhronos: " + str(n_gpu) + " GPUs succesfully initialized, "
            "master rank: " + str(master_rank))

    g.forked = True
    g.gpu_comm = gpu_comm
    Function._inv_n_gpu = 1 / n_gpu
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
    theano_functions = [synk_func._f for synk_func in g.synk_functions]
    with open(PKL_FILE, "wb") as f:
        pickle.dump(theano_functions, f, pickle.HIGHEST_PROTOCOL)
    g.sync.init.distributed.value = True
    g.sync.init.barriers.distribute.wait()
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
#                           Execution Helpers                                 #


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


def exct_in(exct_ID, sub_ID):
    g.sync.exct.ID.value = exct_ID
    g.sync.exct.sub_ID.value = sub_ID
    g.sync.exct.barrier_in.wait()


def exct_out():
    g.sync.exct.barrier_out.wait()
    if not g.sync.exct.workers_OK.value:
        raise RuntimeError("Encountered worker error during execution loop.")

