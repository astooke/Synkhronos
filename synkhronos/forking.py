
from .gpu_utils import get_n_gpu
from .scatterer import scatterer
from . import exct
from .synchronize import build_syncs, give_syncs
from .comm import connect_as_master
from .worker import worker_main, profiling_worker


###############################################################################
#                                                                             #
#                      API for Initializing and Exiting.                      #
#                                                                             #
###############################################################################


def fork(n_parallel=None, use_gpu=True, master_rank=0,
         profile_workers=False, max_n_var=1000, max_dim=16):
    if exct.state.forked:
        raise RuntimeError("Only fork once.")
    target = profiling_worker if profile_workers else worker_main

    if use_gpu:
        n_gpu = get_n_gpu()
        n_parallel = check_n_parallel(n_parallel, master_rank, n_gpu)
    else:
        raise NotImplementedError

    syncs = build_syncs(n_parallel, max_n_var, max_dim)
    exct.fork(target, n_parallel, master_rank, use_gpu, syncs)
    give_syncs(syncs)
    if use_gpu:
        exct.init_gpus(master_rank, n_parallel)
    scatterer.assign_rank(n_parallel, master_rank)
    connect_as_master(n_parallel, master_rank, master_rank, use_gpu)

    return n_parallel


def close():
    """Close workers and join their processes.  Called automatically on exit.
    """
    if not exct.state.forked:
        print("WARNING: Calling synkhronos.close() before forking has no effect.")
    elif exct.state.closed:
        print("WARNING: Called synkhronos.close() after already closed.")
    else:
        exct.close()


###############################################################################
#                                                                             #
#                             Helpers.                                        #
#                                                                             #
###############################################################################


def check_n_parallel(n_parallel, master_rank, n_gpu):
    if n_gpu < 2:
        raise NotImplementedError("Less than 2 GPUs found on computer.")
    if master_rank > n_gpu - 1 or master_rank < 0:
        raise ValueError("Invalid value for master rank.")
    n_parallel = n_gpu if n_parallel is None else n_parallel
    if n_parallel > n_gpu:
        raise ValueError("Requested to use {} GPUs but found only {}".format(
            n_parallel, n_gpu))
    print("Synkhronos attempting to use {} of {} detected GPUs...".format(
        n_parallel, n_gpu))
    return n_parallel

