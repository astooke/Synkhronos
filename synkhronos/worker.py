
import atexit

from .synchronize import give_syncs
from .scatterer import scatterer
from .comm import connect_as_worker
from .function_builder import receive_distribution
from .data_module import WorkerData
from .collectives import worker_gpu_coll, worker_cpu_coll, worker_synk_coll
from . import exct


###############################################################################
#                                                                             #
#                             Tasks                                           #
#                                                                             #
###############################################################################


def manage_data(data_op, sync_data):
    if data_op == exct.CREATE:
        scatterer.append(WorkerData(len(scatterer)))
        return
    synk_data = scatterer.get_data(sync_data.ID.value)
    if data_op == exct.ALLOC:
        synk_data.alloc_shmem()
        synk_data.shape_data()
    elif data_op == exct.RESHAPE:
        synk_data.shape_data()
    elif data_op == exct.FREE:
        synk_data.free_memory()
    else:
        raise RuntimeError("Unrecognized data management ID in worker.")


###############################################################################
#                                                                             #
#                          Main Executive                                     #
#                                                                             #
###############################################################################


def profiling_worker(*args, **kwargs):
    import cProfile
    cProfile.runctx('worker_main(*args, **kwargs)', locals(), globals(),
        "worker_nored_trust_inputs.prof")


def worker_main(rank, n_parallel, master_rank, use_gpu, syncs):
    give_syncs(syncs)
    if use_gpu:
        exct.init_gpus(rank)
    scatterer.assign_rank(n_parallel, rank, create=False)
    connect_as_worker(n_parallel, rank, master_rank, use_gpu)
    atexit.register(exct.worker_error_close)

    while True:
        exct.sync.barrier_in.wait()
        if exct.sync.quit.value:
            atexit.unregister(exct.worker_error_close)
            return  # (exit successfully)
        exct_ID = exct.sync.ID.value
        sub_ID = exct.sync.sub_ID.value
        if exct_ID == exct.DISTRIBUTE:
            synk_fs = receive_distribution()
        elif exct_ID == exct.FUNCTION:
            synk_fs[sub_ID]()
        elif exct_ID == exct.GPU_COLL:
            worker_gpu_coll(sub_ID)
        elif exct_ID == exct.CPU_COLL:
            worker_cpu_coll(sub_ID)
        elif exct_ID == exct.SYNK_COLL:
            worker_synk_coll(sub_ID)
        elif exct_ID == exct.DATA:
            manage_data(sub_ID, syncs.data)
        else:
            raise RuntimeError("Invalid worker exec ID: {}".format(exct_ID))
        exct.sync.barrier_out.wait()  # Prevent premature shmem overwriting.
