
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything unique to the workers.
"""

import os
import pickle
from threading import BrokenBarrierError
import atexit

from .function import WorkerFunction
from .data import WorkerData
from .scatterer import scatterer
from .shareds_registry import SharedsRegistry
from . import exct
from .synchronize import give_syncs

from .cpu_comm import CpuCommWorker
from .common import *


CREATE = False


###############################################################################
#                                                                             #
#                             Tasks                                           #
#                                                                             #
###############################################################################


def receive_distribution(sync_dist, n_parallel):
    with open(PKL_FILE, "rb") as f:
        distribution = pickle.load(f)
    if sync_dist.barrier.wait() == 0:  # (only one worker does it)
        os.remove(PKL_FILE)  # (leave no trace)
    synk_funcs = list()
    shareds_registry = SharedsRegistry()
    for i, f_info in enumerate(distribution):
        if f_info["sliced_function"] is None:  # (avoided duplicate pickling)
            f_info["sliced_function"] = f_info["theano_function"]
        assert f_info["ID"] == i
        synk_funcs.append(WorkerFunction(**f_info))
        shareds_registry.register_func(f_info["theano_function"])
    shareds_registry.set_n_parallel(n_parallel)
    return synk_functions, shareds_registry


def do_gpu_comms(comm_ID, sync_comm, shareds_registry, gpu_comm, master_rank):
    shared_IDs = sync_comm.vars[:sync_comm.n_shared.value]
    if comm_ID == GPU_ALL_GATHER:
        src = shareds_registry.get_array(shared_IDs[0])
        dest = shareds_registry.get_array(shared_IDs[1])
        gpu_comm.all_gather(src, dest)
    else:
        op = sync_comm.op.value.decode('utf-8')
        avg = op == "avg"
        op = "sum" if avg else op
        for shared_ID in shared_IDs:
            src = shareds_registry.get_array(shared_ID)
            if comm_ID == GPU_BROADCAST:
                gpu_comm.broadcast(src, root=master_rank)
            elif comm_ID == GPU_REDUCE:
                # NOTE: kwarg 'dest' only needed for NCCL bug.
                gpu_comm.reduce(src, op=op, root=master_rank, dest=src)
            elif comm_ID == GPU_ALL_REDUCE:
                gpu_comm.all_reduce(src, op=op, dest=src)
            elif comm_ID == GPU_GATHER:
                gpu_comm.all_gather(src)
            else:
                raise RuntimeError("Unrecognized GPU communication type in "
                    "worker.")
        if comm_ID == GPU_ALL_REDUCE and avg:
            shareds_registry.call_avg_fs(shared_IDs)


def do_cpu_comms(comm_ID, sync_comm, shareds_registry):
    n_shared = sync_comm.n_shared.value
    shared_vars = shareds_registry.get_vars_from_IDs(sync_comm.vars[:n_shared])
    if comm_ID == exct.SCATTER:
        my_inputs, _ = scatterer.get_my_inputs(n_shared)
        for var, my_input in zip(shared_vars, my_inputs):
            var.set_value(my_input)
    else:
        raise RuntimeError("Unrecognized CPU comm type in worker.")


def manage_data(data_op, data_ID):
    if data_op == exct.DATA_CREATE:
        scatterer.append(WorkerData(len(scatterer)))
        return
    synk_data = scatterer.get_data(data_ID)
    if data_op == exct.DATA_ALLOC:
        synk_data.alloc_shmem()
        synk_data.shape_data()
    elif data_op == exct.DATA_RESHAPE:
        synk_data.shape_data()
    elif data_op == exct.DATA_FREE:
        synk_data.free_memory()
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


def worker_exct(rank, n_parallel, master_rank, syncs):

    give_syncs(syncs)
    scatterer.init_parallel(n_parallel, rank, CREATE)
    WorkerFunction.master_rank = master_rank


    sync.init.semaphore.acquire()  # blocks worker but not master
    cpu_comm = CpuCommWorker(sync.init.dict["ports"][rank])

    gpu_comm = init_gpu(rank, n_parallel, sync, CREATE)
    if not gpu_comm:
        return  # (exit quietly)


    atexit.register(error_close, sync.exct)

    while True:
        sync.exct.barrier_in.wait()
        if sync.exct.quit.value:
            atexit.unregister(error_close)
            return  # (exit successfully)
        exct_ID = sync.exct.ID.value
        sub_ID = sync.exct.sub_ID.value
        if exct_ID == exct.DISTRIBUTE:
            synk_fs, shareds_registry = receive_distribution(sync.dist, n_parallel)
        elif exct_ID == exct.FUNCTION:
            synk_fs[sub_ID](gpu_comm, cpu_comm)
        elif exct_ID == exct.GPU_COMM:
            do_gpu_comms(sub_ID, sync.comm, shareds_registry, gpu_comm, master_rank)
        elif exct_ID == exct.CPU_COMM:
            do_cpu_comms(sub_ID, sync.comm, shareds_registry)
        elif exct_ID == exct.DATA:
            manage_data(sub_ID, sync.data.ID.value)
        else:
            raise RuntimeError("Unrecognized exctution type in worker.")
        sync.exct.barrier_out.wait()  # Prevent premature shmem overwriting.
