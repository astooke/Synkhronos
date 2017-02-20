"""
Classes and functions used by master but which don't MODIFY globals.
(Might still read from globals passed explicitly as parameter.)
"""

import numpy as np
import multiprocessing as mp
import ctypes

from .common import REDUCE, COLLECT_MODES, REDUCE_OPS, REDUCE_NO_OP
from .variables import struct, BaseData


###############################################################################
#                              (fork)                                         #


def n_gpu_getter(mp_n_gpu):
    """
    Call in a subprocess because it prevents future subprocesses from using GPU.
    """
    from pygpu import gpuarray
    mp_n_gpu.value = gpuarray.count_devices("cuda", 0)


def get_n_gpu(n_gpu, master_rank):
    master_rank = int(master_rank)
    if n_gpu is not None:
        n_gpu = int(n_gpu)
    else:
        #  Detect the number of devices present and use all.
        mp_n_gpu = mp.RawValue('i', 0)
        p = mp.Process(target=n_gpu_getter, args=(mp_n_gpu,))
        p.start()
        p.join()
        n_gpu = mp_n_gpu.value
        if n_gpu == 0:
            # TODO: CPU-only mode.
            raise RuntimeError("No cuda GPU detected by pygpu.")
        elif n_gpu == 1:
            # TODO: Allow to run on one GPU.
            raise RuntimeError("Only one GPU detected; just use Theano.)")
        else:
            print("Detected and attempting to use {} GPUs...".format(n_gpu))

    if master_rank not in list(range(n_gpu)):
        raise ValueError("Invalid value for master rank: ", master_rank)

    return n_gpu, master_rank


def build_sync(n_gpu):

    mgr = mp.Manager()
    dictionary = mgr.dict()
    barriers = struct(
        gpu_inits=[mp.Barrier(n_gpu) for _ in range(3)],
        distribute=mp.Barrier(n_gpu),
        distribute_out=mp.Barrier(n_gpu),
        delete_pkl=mp.Barrier(n_gpu - 1),
    )
    init = struct(
        dict=dictionary,
        n_user_fcns=mp.RawValue('i', 0),
        distributed=mp.RawValue(ctypes.c_bool, False),
        barriers=barriers,
    )
    execute = struct(
        quit=mp.RawValue(ctypes.c_bool, False),
        ID=mp.RawValue('i', 0),
        barrier_in=mp.Barrier(n_gpu),
        barrier_out=mp.Barrier(n_gpu),
        workers_OK=mp.Value(ctypes.c_bool, True)
    )
    IDs = struct(
        func=mp.RawValue('i', 0),
        comm=mp.RawValue('i', 0),
        op=mp.RawValue('i', 0),
        data=mp.RawValue('i', 0),
        n_shared=mp.RawValue('i', 0),
        vars=None,  # (allocated later)
        datas=None,
        dtype=mp.RawValue('i', 0),
        ndim=mp.RawValue('i', 0),
        shape=np.ctypeslib.as_array(mp.RawArray('i', 10)),  # (ndim <= 10)
        tag=mp.RawValue('i', 0),
    )
    scat = struct(
        assign_idxs=np.ctypeslib.as_array(mp.RawArray('i', n_gpu + 1)),
        use_idxs_arr=mp.RawValue(ctypes.c_bool, False),
        idxs_tag=mp.RawValue('i', 0),
        idxs_size=mp.RawValue('i', 0),
        idxs_arr=None,  # (allocated later)
    )
    sync = struct(
        init=init,
        exct=execute,
        IDs=IDs,
        scat=scat,
    )
    return sync


###############################################################################
#                           (function)                                        #


def _check_collect_reduce(args, check_list, name):
    if not isinstance(args, (list, tuple)):
        args = (args,)
    for arg in args:
        if arg not in check_list:
            raise ValueError("Unrecognized ", name, ": ", arg, ". Must be "
                "in: ", check_list)


def _build_collect_reduce(n_outputs, args, check_list, name):
    _check_collect_reduce(args, check_list, name)
    if isinstance(args, (list, tuple)):
        if len(args) != n_outputs:
            raise ValueError("Number of ", name, " args does not match number "
                "of outputs (or enter a single string to be used for all "
                "outputs).  None is a valid entry.")
        arg_IDs = tuple([check_list.index(arg) for arg in args])
    else:
        arg_ID = check_list.index(arg)
        arg_IDs = (arg_ID,) * n_outputs
    return arg_IDs


def build_collect_IDs(n_outputs, collect_modes):
    return _build_collect_reduce(n_outputs, collect_modes, COLLECT_MODES, "collect mode")


def build_reduce_IDs(n_outputs, reduce_ops):
    return _build_collect_reduce(n_outputs, reduce_ops, REDUCE_OPS, "reduce operation")


def check_collect_vs_op_IDs(collect_IDs, reduce_IDs):
    for idx, (collect_ID, reduce_ID) in enumerate(zip(collect_IDs, reduce_IDs)):
        if collect_ID == REDUCE:
            if reduce_ID == REDUCE_NO_OP:
                raise ValueError("Had reduce operation 'None' for reduce "
                    "collection mode.")
        else:
            reduce_IDs[idx] = REDUCE_NO_OP  # ignores any input value
    return reduce_IDs


def build_collect(outputs, collect_modes, reduce_ops):
    if outputs is None:
        return (), ()
    n_outputs = len(outputs) if isinstance(outputs, (list, tuple)) else 1
    collect_IDs = build_collect_IDs(n_outputs, collect_modes)
    reduce_IDs = build_reduce_IDs(n_outputs, reduce_ops)
    reduce_IDs = check_collect_vs_op_IDs(collect_IDs, reduce_IDs)
    return collect_IDs, reduce_IDs


def check_ouput_subset(n_outputs, output_subset):
    if not isinstance(output_subset, list):
        raise TypeError("Optional param 'output_subset' must be a "
            "list of ints.")
    for idx in output_subset:
        if not isinstance(idx, int):
            raise TypeError("Optional param 'output_subset' must a "
                "list of ints.")
        if idx < 0 or idx > n_outputs - 1:
            raise ValueError("Output subset entry out of range: ", idx)


def check_synk_inputs(synk_datas, dtypes, ndims):
    for idx, (data, dtype, ndim) in enumerate(zip(synk_datas, dtypes, ndims)):
        if not isinstance(data, BaseData):
            raise ValueError("All function inputs must be of type SynkData.")
        if data.dtype != dtype:
            raise TypeError("Incorrect input dtype for position {}; "
                "expected: {}, received: {}.".format(idx, dtype, data.dtype))
        if data.ndim != ndim:
            raise TypeError("Incorrect input dimensions for position {}; "
                "expected: {}, received: {}.".format(idx, ndim, data.ndim))
        data._check_data()


###############################################################################
#                       Shared Memory Management                              #


def oversize_shape(data, oversize):
    """ Master only """
    shape = data.shape
    if oversize is not None:
        if oversize < 1 or oversize > 2:
            raise ValueError("Param 'oversize' must be in range 1 to 2"
                " (direct multiplicative factor on 0-th index size).")
        if shape:
            shape = list(shape)
            shape[0] = int(np.ceil(shape[0] * oversize))
    return shape


def build_scat_idxs(n_gpu, lengths, batch):
    if batch is not None:
        if isinstance(batch, int):  # (size from 0 is used)
            max_idx = batch
            space_start = 0
            space_end = batch
        elif isinstance(batch, slice):  # (slice is used)
            max_idx = batch.stop
            space_start = batch.start
            space_end = batch.stop
        else:  # (explicit indices are used)
            max_idx = max(batch)
            space_start = 0
            space_end = len(batch)
        if max_idx > min(lengths):
            raise ValueError("Requested index out of range of input lengths.")
    else:  # (i.e. no batch directive provided, use full array lengths)
        space_start = 0
        space_end = lengths[0]
        if lengths.count(space_end) != len(lengths):  # (fast)
            raise ValueError("If not providing param 'batch', all "
                "inputs must be the same length.  Had lengths: ", lengths)
    return np.linspace(space_start, space_end, n_gpu + 1, dtype=np.int32)


def check_batch_types(batch):
    if batch is not None:
        if isinstance(batch, (list, tuple)):
            batch = np.array(batch, dtype='int32')
        if isinstance(batch, np.ndarray):
            if batch.ndim > 1:
                raise ValueError("Array for param 'batch' must be "
                    "1-dimensional, got: ", batch.ndim)
            if "int" not in batch.dtype.name:
                raise ValueError("Array for param 'batch' must be integer "
                    "dtype, got: ", batch.dtype.name)
        elif not isinstance(batch, (int, slice)):
            raise TypeError("Param 'batch' must be either an integer, a slice, "
                "or a list, tuple, or numpy array of integers.")
    return batch


###############################################################################
#                           GPU Collectives                                   #


def get_op_from_avg(op):
    _check_collect_reduce(op, REDUCE_OPS[:-1], "reduce operation")  # (disallow 'None')
    op_ID = REDUCE_OPS.index(op)  # (worker can handle avg fine)
    if op == "avg":
        op = "sum"
        avg = True
    else:
        avg = False
    return op, avg, op_ID


###############################################################################
#                           User                                              #

def make_slices(data_collection):
    """Make a set of slice objects according to lengths of data subsets.

    Example:
        >>> slice_1, slice_2 = make_slices([0, 1, 2, 3], [10, 11, 12])
        >>> slice_1
        slice(0, 3, None)
        >>> slice_2
        slice(3, 6, None)

    Args:
        data_collection (list): collection of data arrays whose lengths to use

    Returns:
        slice: slice objects

    Raises:
        TypeError: if input is not list or tuple
    """
    if not isinstance(data_collection, (list, tuple)):
        raise TypeError("Expected list or tuple for input.")
    endings = [0]
    for data_arr in data_collection:
        endings.append(endings[-1] + len(data_arr))
    slices = list()
    for i in range(len(data_collection)):
        slices.append(slice(endings[i], endings[i + 1]))
    return tuple(slices)

