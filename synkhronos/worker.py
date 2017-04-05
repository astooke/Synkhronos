
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything unique to the workers.
"""

import os
import pickle
from threading import BrokenBarrierError
import atexit

from .function import WorkerFunction
from .data import BaseData
from .scatterer import WorkerScatterer
from .shareds_registry import SharedsRegistry

from .variables import Shareds, BaseFunction, BaseData, BaseScatterer
from .cpu_comm import CpuCommWorker
from .common import *
from .accumulators import Accumulators


CREATE = False

###############################################################################
#                                                                             #
#                             Tasks                                           #
#                                                                             #
###############################################################################


def unpack_distro(distribution, accumulators):
    """
    Worker will recover variables in the same order as the master committed
    them, so they will have the same ID (index).
    """
    synk_funcs = list()
    g_shareds = Shareds()
    for i, f_info in enumerate(distribution):
        if f_info["sliced_function"] is None:  # (avoided duplicate pickling)
            f_info["sliced_function"] = f_info["theano_function"]
        synk_funcs.append(WorkerFunction(ID=i,
                                         accumulators=accumulators,
                                         **f_info,
                                         )
        )
        g_shareds.register_func(f_info["theano_function"], accumulators)
    return synk_funcs, g_shareds


def receive_distribution(sync_dist, accumulators, n_parallel):
    with open(PKL_FILE, "rb") as f:
        distribution = pickle.load(f)  # should be all in one list
    if sync_dist.barrier.wait() == 0:  # only one worker does it
        os.remove(PKL_FILE)  # leave no trace
    synk_functions, g_shareds = unpack_distro(distribution, accumulators)
    g_shareds.set_n_parallel(n_parallel)
    return synk_functions, g_shareds


def do_gpu_comms(comm_ID, sync_comm, g_shareds, gpu_comm, master_rank):
    shared_IDs = sync_comm.vars[:sync_comm.n_shared.value]
    if comm_ID == GPU_ALL_GATHER:
        src = g_shareds.get_array(shared_IDs[0])
        dest = g_shareds.get_array(shared_IDs[1])
        gpu_comm.all_gather(src, dest)
    else:
        if comm_ID == GPU_REDUCE:
            op = REDUCE_OPS_WORKER[sync_comm.op.value]
        elif comm_ID == GPU_ALL_REDUCE:
            op = REDUCE_OPS[sync_comm.op.value]
            avg = op == "avg"
            op = "sum" if avg else op
        for shared_ID in shared_IDs:
            src = g_shareds.get_array(shared_ID)
            if comm_ID == GPU_BROADCAST:
                gpu_comm.broadcast(src, root=master_rank)
            elif comm_ID == GPU_REDUCE:
                gpu_comm.reduce(src, op=op, root=master_rank)
            elif comm_ID == GPU_ALL_REDUCE:
                gpu_comm.all_reduce(src, op=op, dest=src)
            elif comm_ID == GPU_GATHER:
                gpu_comm.all_gather(src)
            else:
                raise RuntimeError("Unrecognized GPU communication type in "
                    "worker.")
        if comm_ID == GPU_ALL_REDUCE and avg:
            g_shareds.call_avg_fs(shared_IDs)


def do_cpu_comms(comm_ID, sync_comm, scatterer, g_shareds):
    n_shared = sync_comm.n_shared.value
    shared_vars = g_shareds.get_vars_from_IDs(sync_comm.vars[:n_shared])
    if comm_ID == SCATTER:
        my_inputs, _ = scatterer.get_my_inputs(n_shared)
        for var, my_input in zip(shared_vars, my_inputs):
            var.set_value(my_input)
    else:
        raise RuntimeError("Unrecognized CPU comm type in worker.")


def manage_data(data_op, sync_data, scatterer):
    if data_op == DATA_CREATE:
        dtype = DTYPES[sync_data.dtype.value]
        ndim = sync_data.ndim.value
        scatter = sync_data.scatter.value
        scatterer.append(BaseData(len(scatterer), dtype, ndim, scatter))
        return
    synk_data = scatterer.get_data(sync_data.ID.value)
    ndim = synk_data.ndim
    shape = sync_data.shape[:ndim]
    if data_op == DATA_ALLOC:
        synk_data._alloc_shmem(sync_data.alloc_size.value, sync_data.tag.value)
        synk_data._shape_data(shape)
    elif data_op == DATA_RESHAPE:
        synk_data._shape_data(shape)
    elif data_op == DATA_FREE:
        synk_data._free_shmem()
    else:
        raise RuntimeError("Unrecognized data management ID in worker.")


def error_close(sync_exct):
    sync_exct.workers_OK.value = False
    try:
        sync_exct.barrier_out.wait(1)
    except BrokenBarrierError:
        pass


###############################################################################
#                                                                             #
#                          Main Executable                                    #
#                                                                             #
###############################################################################


def profiling_worker(*args, **kwargs):
    import cProfile
    cProfile.runctx('worker_exct(*args, **kwargs)', locals(), globals(),
        "worker_nored_trust_inputs.prof")


def worker_exct(rank, n_parallel, master_rank, sync, sync_scat):

    sync.init.semaphore.acquire()  # blocks worker but not master
    cpu_comm = CpuCommWorker(sync.init.dict["ports"][rank])

    gpu_comm = init_gpu(rank, n_parallel, sync, CREATE)
    if not gpu_comm:
        return  # (exit quietly)

    scatterer = WorkerScatterer(sync_scat, n_parallel, rank)
    accumulators = Accumulators()
    Function.master_rank = master_rank

    atexit.register(error_close, sync.exct)

    while True:
        sync.exct.barrier_in.wait()
        if sync.exct.quit.value:
            atexit.unregister(error_close)
            return  # (exit successfully)
        exct_ID = sync.exct.ID.value
        sub_ID = sync.exct.sub_ID.value
        if exct_ID == DISTRIBUTE:
            synk_fs, g_shareds = receive_distribution(sync.dist, accumulators, n_parallel)
        elif exct_ID == FUNCTION:
            synk_fs[sub_ID](sync.func, scatterer, gpu_comm, cpu_comm, accumulators)
        elif exct_ID == GPU_COMM:
            do_gpu_comms(sub_ID, sync.comm, g_shareds, gpu_comm, master_rank)
        elif exct_ID == CPU_COMM:
            do_cpu_comms(sub_ID, sync.comm, scatterer, g_shareds)
        elif exct_ID == DATA:
            manage_data(sub_ID, sync.data, scatterer)
        else:
            raise RuntimeError("Unrecognized exctution type in worker.")
        sync.exct.barrier_out.wait()  # Prevent premature shmem overwriting.
