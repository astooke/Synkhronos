
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything unique to the workers.
"""

import os
import pickle
from threading import BrokenBarrierError
import atexit

from .util import init_gpu
from .function import WorkerFunction
from .data import WorkerData
from .scatterer import scatterer
from .shareds_registry import SharedsRegistry
from . import exct
from .synchronize import give_syncs
from .comm import cpu_comm_worker, gpu_comm

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


def do_gpu_comms(comm_ID, sync_comm, shareds_registry):
    shared_IDs = sync_comm.vars[:sync_comm.n_shared.value]
    if comm_ID == exct.GPU_ALL_GATHER:
        src = shareds_registry.get_array(shared_IDs[0])
        dest = shareds_registry.get_array(shared_IDs[1])
        gpu_comm.all_gather(src, dest)
    else:
        op = sync_comm.op.value.decode('utf-8')
        avg = op == "avg"
        op = "sum" if avg else op
        for shared_ID in shared_IDs:
            src = shareds_registry.get_array(shared_ID)
            if comm_ID == exct.GPU_BROADCAST:
                gpu_comm.broadcast(src)
            elif comm_ID == exct.GPU_REDUCE:
                # NOTE: kwarg 'dest' only needed for NCCL bug.
                gpu_comm.reduce(src=src, op=op, dest=src)
            elif comm_ID == exct.GPU_ALL_REDUCE:
                gpu_comm.all_reduce(src=src, op=op, dest=src)
            elif comm_ID == exct.GPU_GATHER:
                gpu_comm.all_gather(src)
            else:
                raise RuntimeError("Unrecognized GPU communication type in "
                    "worker.")
        if comm_ID == exct.GPU_ALL_REDUCE and avg:
            shareds_registry.call_avg_fs(shared_IDs)


def do_cpu_comms(comm_ID, sync_coll, shareds_registry):
    n_shared = sync_coll.n_shared.value
    shared_vars = shareds_registry.get_vars_from_IDs(sync_coll.vars[:n_shared])
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


###############################################################################
#                                                                             #
#                          Main Executable                                    #
#                                                                             #
###############################################################################


def profiling_worker(*args, **kwargs):
    import cProfile
    cProfile.runctx('worker_main(*args, **kwargs)', locals(), globals(),
        "worker_nored_trust_inputs.prof")


def worker_main(rank, n_parallel, master_rank, use_gpu, syncs):

    give_syncs(syncs)
    scatterer.assign_rank(n_parallel, rank, CREATE)
    WorkerFunction.master_rank = master_rank
    cpu_comm.connect(rank)
    if use_gpu:
        init_gpu(rank)
        if gpu_comm is not None:
            gpu_comm.connect(n_parallel, rank, master_rank)
    atexit.register(exct.worker_error_close)

    while True:
        exct.sync.barrier_in.wait()
        if exct.sync.quit.value:
            atexit.unregister(exct.worker_error_close)
            return  # (exit successfully)
        exct_ID = exct.sync.ID.value
        sub_ID = exct.sync.sub_ID.value
        if exct_ID == exct.DISTRIBUTE:
            synk_fs, shareds_registry = receive_distribution(syncs.dist, n_parallel)
        elif exct_ID == exct.FUNCTION:
            synk_fs[sub_ID](gpu_comm, cpu_comm)
        elif exct_ID == exct.GPU_COMM:
            do_gpu_comms(sub_ID, syncs.coll, shareds_registry)
        elif exct_ID == exct.CPU_COMM:
            do_cpu_comms(sub_ID, syncs.coll, shareds_registry)
        elif exct_ID == exct.DATA:
            manage_data(sub_ID, syncs.data.ID.value)
        else:
            raise RuntimeError("Unrecognized exctution type in worker.")
        sync.exct.barrier_out.wait()  # Prevent premature shmem overwriting.
