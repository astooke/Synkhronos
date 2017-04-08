
import multiprocessing as mp

from .util import init_gpu
from .scatterer import scatterer
from .collectives import shareds_registry
from . import exct
from .synchronize import build_syncs, give_syncs
from .comm import cpu_comm_master, gpu_comm
from .worker import worker_main, profiling_worker


__all__ = ["fork", "close", "get_n_gpu"]


###############################################################################
#                                                                             #
#                      API for Initializing and Exiting.                      #
#                                                                             #
###############################################################################


def fork(n_parallel=None, use_gpu=True, master_rank=0,
         profile_workers=False, max_n_var=100, max_dim=16):
    """Fork a python sub-process for each additional GPU and initialize.

    Call this function before building any Theano variables.  (Theano must be
    configured to ipmort to CPU only.)  Initializes one GPU on each process,
    including the master, and initializes GPU collective communications via
    pygpu & NVIDIA NCCL.

    Args:
        n_parallel (None, optional): Number of GPUs to use (default is all)
        master_rank (int, optional): default is 0

    Raises:
        RuntimeError: If already forked or fails to initialize.

    Returns:
        int: number of GPUs using.
    """
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
    scatterer.assign_rank(n_parallel, master_rank, create=True)
    cpu_comm_master.connect(n_parallel, master_rank, master_rank)
    if use_gpu:
        init_gpu(master_rank)
        if gpu_comm is None:
            print("WARNING: Using GPUs, but unable to import GPU collectives "
                "from pygpu (you might need to install NCCL); "
                "reverting to CPU-based collectives.")
        else:
            gpu_comm.connect(n_parallel, master_rank)
        if not exct.sync.workers_OK.value:
            close()
            raise RuntimeError("Workers did not initialize GPUs.")
        print("Synkhronos: " + str(n_parallel) + " GPUs succesfully initialized, "
            "master rank: " + str(master_rank))
    shareds_registry.set_n_parallel(n_parallel)
    Function._inv_n = 1 / n_parallel
    return n_parallel


def close():
    """Close workers and join their processes.  Called automatically on exit.
    """
    if not exct.state.forked:
        print("WARNING: Calling synkhronos.close before forking has no effect.")
    elif exct.state.closed:
        print("WARNING: Called synkhronos.close after already closed.")
    else:
        exct.close()


def get_n_gpu():
    detected_n_gpu = mp.RawValue('i', 0)
    p = mp.Process(target=n_gpu_subprocess, args=(detected_n_gpu,))
    p.start()
    p.join()
    n_gpu = int(detected_n_gpu.value)
    if n_gpu == -1:
        raise ImportError("Must be able to import pygpu to use GPUs.")
    return n_gpu


###############################################################################
#                                                                             #
#                             Helpers.                                        #
#                                                                             #
###############################################################################


def n_gpu_subprocess(mp_n_gpu):
    """
    Call in a subprocess because it prevents future subprocesses from using GPU.
    """
    try:
        from pygpu import gpuarray
        mp_n_gpu.value = gpuarray.count_devices("cuda", 0)
    except ImportError as exc:
        mp_n_gpu.value = -1


def check_n_parallel(n_parallel, master_rank, n_gpu):
    if n_gpu < 2:
        raise NotImplementedError
    if master_rank > n_gpu - 1 or master_rank < 0:
        raise ValueError("Invalid value for master rank.")
    n_parallel = n_gpu if n_parallel is None else n_parallel
    if n_parallel > n_gpu:
        raise ValueError("Requested to use {} GPUs but found only {}".format(
            n_parallel, n_gpu))
    print("Synkhronos attempting to use {} of {} detected GPUs...".format(
        n_parallel, n_gpu))



