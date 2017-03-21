"""
Classes and functions used by master but which don't MODIFY globals.
(Might still read from globals passed explicitly as parameter.)
"""

import numpy as np
import multiprocessing as mp
from ctypes import c_bool, c_long, c_int
import zmq

from .common import GPU_REDUCE, CPU_REDUCE, COLLECT_MODES, REDUCE_OPS, REDUCE_NO_OP
from .variables import BaseData
from .cpu_comm import CpuCommMaster


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


###############################################################################
#                              fork                                           #


def n_gpu_getter(mp_n_gpu):
    """
    Call in a subprocess because it prevents future subprocesses from using GPU.
    """
    try:
        from pygpu import gpuarray
    except ImportError as exc:
        raise exc("Must have pygpu installed to use GPUs.")
    mp_n_gpu.value = gpuarray.count_devices("cuda", 0)


def get_n_gpu(n_gpu, master_rank):
    detected_n_gpu = mp.RawValue('i', 0)
    p = mp.Process(target=n_gpu_getter, args=(detected_n_gpu,))
    p.start()
    p.join()
    detected_n_gpu = detected_n_gpu.value
    if detected_n_gpu == 0:
        raise NotImplementedError
    n_gpu = detected_n_gpu if n_gpu is None else int(n_gpu)
    if n_gpu > detected_n_gpu:
        raise ValueError("Requested to use {} GPUs but only found {}.".format(
            n_gpu, detected_n_gpu))
    if n_gpu == 1:
        raise NotImplementedError("Only one GPU requested/detected; just use Theano for now.)")
    else:
        if int(master_rank) not in list(range(n_gpu)):
            raise ValueError("Invalid value for master rank: {}".format(
                master_rank))
        print("Synkhronos attempting to use {} of {} detected GPUs...".format(
            n_gpu, detected_n_gpu))
    return n_gpu


def build_sync(n_parallel, n_in_out, n_shared, max_dim):

    mgr = mp.Manager()
    dictionary = mgr.dict()
    barriers = struct(
        gpu_inits=[mp.Barrier(n_parallel) for _ in range(3)],
        distribute=mp.Barrier(n_parallel),
        distribute_out=mp.Barrier(n_parallel),
        delete_pkl=mp.Barrier(n_parallel - 1),
    )
    init = struct(
        dict=dictionary,
        distributed=mp.RawValue(c_bool, False),
        semaphore=mp.Semaphore(0),
        barriers=barriers,
    )
    exct = struct(
        quit=mp.RawValue(c_bool, False),
        ID=mp.RawValue('i', 0),
        sub_ID=mp.RawValue('i', 0),
        barrier_in=mp.Barrier(n_parallel),
        barrier_out=mp.Barrier(n_parallel),
        workers_OK=mp.Value(c_bool, True)  # (not Raw)
    )
    comm = struct(
        op=mp.RawValue('i', 0),
        n_shared=mp.RawValue('i', 0),
        vars=mp.RawArray('I', n_shared),
        datas=mp.RawArray('I', n_in_out),
    )
    data = struct(
        ID=mp.RawValue('i', 0),
        dtype=mp.RawValue('i', 0),  # (by ID)
        ndim=mp.RawValue('i', 0),
        shape=np.ctypeslib.as_array(mp.RawArray('l', max_dim)),
        tag=mp.RawValue('i', 0),
        scatter=mp.RawValue(c_bool, True),
        alloc_size=mp.RawValue('l', 0),
    )
    func = struct(
        output_subset=mp.RawArray(c_bool, [True] * n_in_out),
        collect_modes=mp.RawArray('B', n_in_out),
        reduce_ops=mp.RawArray('B', n_in_out),
        n_slices=mp.RawValue('i', 0),
        new_collect=mp.RawValue(c_bool, True),
    )
    sync = struct(
        init=init,
        exct=exct,
        comm=comm,
        data=data,
        func=func,
    )
    return sync


def build_sync_scat(n_parallel, n_in_out):
    sync = struct(
        assign_idxs=np.ctypeslib.as_array(mp.RawArray('l', n_parallel + 1)),
        use_idxs_arr=mp.RawValue(c_bool, False),
        tag=mp.RawValue('i', 0),
        size=mp.RawValue('l', 0),
        idxs_arr=None,  # (allocated later; only need if shuffling)
        data_IDs=mp.RawArray('i', n_in_out),
    )
    return sync


def init_cpu_comm(n_parallel, master_rank, sync_init):
    cpu_comm = CpuCommMaster(n_parallel)
    ports = list(cpu_comm.ports)
    ports.append(ports[master_rank])
    sync_init.dict["ports"] = ports
    for _ in range(n_parallel - 1):
        sync_init.semaphore.release()
    return cpu_comm



###############################################################################
#                           Collect & Reduce                                  #


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
        arg_IDs = [check_list.index(arg) for arg in args]
    else:
        arg_ID = check_list.index(args)
        arg_IDs = [arg_ID] * n_outputs
    return arg_IDs


def build_collect_IDs(n_outputs, collect_modes):
    return _build_collect_reduce(n_outputs, collect_modes, COLLECT_MODES, "collect mode")


def build_reduce_IDs(n_outputs, reduce_ops):
    return _build_collect_reduce(n_outputs, reduce_ops, REDUCE_OPS, "reduce operation")


def check_collect_vs_op_IDs(collect_IDs, reduce_IDs):
    for idx, (collect_ID, reduce_ID) in enumerate(zip(collect_IDs, reduce_IDs)):
        if collect_ID in [GPU_REDUCE, CPU_REDUCE]:
            if reduce_ID == REDUCE_NO_OP:
                raise ValueError("Had reduce operation 'None' for reduce "
                    "collection mode.")
        else:
            reduce_IDs[idx] = REDUCE_NO_OP  # ignores any input value
    return reduce_IDs


def build_def_collect(outputs, collect_modes, reduce_ops):
    """ For comms only: Function has its own method. """
    if outputs is None:
        return (), ()
    n_outputs = len(outputs) if isinstance(outputs, (list, tuple)) else 1
    collect_IDs = build_collect_IDs(n_outputs, collect_modes)
    reduce_IDs = build_reduce_IDs(n_outputs, reduce_ops)
    reduce_IDs = check_collect_vs_op_IDs(collect_IDs, reduce_IDs)
    return collect_IDs, reduce_IDs


###############################################################################
#                           Inputs and Outputs                                #


def prep_outputs(outputs):
    from theano.gpuarray.type import GpuArrayVariable
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
    to_cpu = [not isinstance(var, GpuArrayVariable) for var in outputs]
    gpu_vars = [var.transfer(None) for var in outputs]
    return gpu_vars, to_cpu


def check_output_subset(n_outputs, output_subset):
    if not isinstance(output_subset, list):
        raise TypeError("Optional param 'output_subset' must be a "
            "list of ints.")
    for idx in output_subset:
        if not isinstance(idx, int):
            raise TypeError("Optional param 'output_subset' must a "
                "list of ints.")
        if idx < 0 or idx > n_outputs - 1:
            raise ValueError("Output subset entry out of range: ", idx)


def check_synk_inputs(synk_datas, vars):
    for idx, (data, var) in enumerate(zip(synk_datas, vars)):
        if not isinstance(data, BaseData):
            raise ValueError("All function inputs must be of type SynkData.")
        if data.dtype != var.dtype:
            raise TypeError("Incorrect input dtype for position {}; expected: "
                "{}, received: {}.".format(idx, var.dtype, data.dtype))
        if data.ndim != var.ndim:
            raise TypeError("Incorrect input dimensions for position {}; "
                "expected: {}, received: {}.".format(idx, var.ndim, data.ndim))
        data._check_data()


###############################################################################
#                               Scatterer                                     #


def build_scat_idxs(n_parallel, lengths, batch):
    if batch is not None:
        if isinstance(batch, int):  # (size from 0 is used)
            max_idx = batch
            start = 0
            end = batch
        elif isinstance(batch, slice):  # (slice is used)
            max_idx = batch.stop
            start = batch.start
            end = batch.stop
        else:  # (explicit indices are used)
            max_idx = max(batch)
            start = 0
            end = len(batch)
        if max_idx > min(lengths):
            raise ValueError("Requested index out of range of input lengths.")
    else:  # (i.e. no batch directive provided, use full array lengths)
        start = 0
        end = lengths[0]
        if lengths.count(end) != len(lengths):  # (fast)
            raise ValueError("If not providing param 'batch', all "
                "inputs must be the same length.  Had lengths: ", lengths)
    return np.linspace(start, end, n_parallel + 1, dtype=np.int64)


def check_batch_types(batch):
    if batch is not None:
        if isinstance(batch, (list, tuple)):
            batch = np.array(batch, dtype='int64')
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


def get_op_and_avg(op):
    _check_collect_reduce(op, REDUCE_OPS[:-1], "reduce operation")  # (disallow 'None')
    op_ID = REDUCE_OPS.index(op)
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

