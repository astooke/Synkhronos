"""
Classes and functions used by master but which don't MODIFY globals.
(Might still read from globals passed explicitly as parameter.)
"""

import numpy as np
import multiprocessing as mp
from ctypes import c_bool

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
    init = struct(
        dict=dictionary,
        semaphore=mp.Semaphore(0),
        barriers=[mp.Barrier(n_parallel) for _ in range(3)],
    )
    dist = struct(
        barrier=mp.Barrier(n_parallel - 1),
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
        dist=dist,
        exct=exct,
        comm=comm,
        data=data,
        func=func,
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

