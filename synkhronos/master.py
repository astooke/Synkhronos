
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has (almost) everything exposed to the user.
"""
# import ipdb


from .scatterer import scatterer
from .shareds_registry import SharedsRegistry
from .collectives import shared_registry
from . import exct
from .accumulators import Accumulators
from .synchronize import build_syncs, give_syncs


from .common import *
from .util import *

__all__ = ["fork", "distribute", "close"]

CREATE = True

# Globals  (only functions exposed to user will use via global access)
g = struct(
    # Multiprocessing
    sync=None,
    scatterer=None,
    # Theano
    shareds=SharedsRegistry(),
    # GPU
    synk_functions=list(),
    accumulators=Accumulators(),
    gpu_comm=None,
    # ZMQ
    cpu_comm=None,
)


###############################################################################
#                                                                             #
#                       Initializing and Exiting.                             #
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
    if profile_workers:
        from .worker import profiling_worker as target
    else:
        from .worker import worker_exct as target

    if use_gpu:
        n_parallel = get_n_gpu(n_parallel, master_rank)
    else:
        raise NotImplementedError

    syncs = build_syncs(n_parallel, max_n_var, max_dim)
    exct.fork(target, n_parallel, master_rank, syncs)
    give_syncs(syncs)
    scatterer.init_parallel(n_parallel, master_rank, create=True)


    g.cpu_comm = init_cpu_comm(n_parallel, master_rank, g.sync.init)

    if use_gpu:
        g.gpu_comm = init_gpu(master_rank, n_parallel, g.sync)
        if not g.gpu_comm:
            raise RuntimeError("At least one synkhronos worker failed to "
                "initialize GPU during fork.")
        else:
            print("Synkhronos: " + str(n_parallel) + " GPUs succesfully initialized, "
                "master rank: " + str(master_rank))

    shared_registry.set_n_parallel(n_parallel)
    exct.state.forked = True
    Function._inv_n = 1 / n_parallel
    return n_parallel


def close():
    """Close workers and join their processes.  Called automatically on exit.
    """
    if not exct.state.forked:
        print("WARNING: Calling close() before forking has no effect.")
    elif exct.state.closed:
        print("WARNING: Called close() after synkhronos already closed.")
    else:
        exct.close()

