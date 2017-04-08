
import multiprocessing as mp
from threading import BrokenBarrierError
import atexit

from .util import struct

# Exec IDs
DISTRIBUTE = 0
FUNCTION = 1
DATA = 2
GPU_COMM = 3
CPU_COMM = 4

# Data Sub IDs
DATA_CREATE = 0
DATA_ALLOC = 1
DATA_RESHAPE = 2
DATA_FREE = 3

# Collectives Sub IDs
GPU_REDUCE = 0  # NOTE: matches position in COLLECT_MODES (maybe not anymore?)
CPU_REDUCE = 1
GPU_GATHER = 2
CPU_GATHER = 3
NO_COLLECT = 4

GPU_BROADCAST = 5
CPU_BROADCAST = 6
GPU_ALL_REDUCE = 7
CPU_ALL_REDUCE = 8
GPU_ALL_GATHER = 9
CPU_ALL_GATHER = 10

# CPU_COMM IDs
SCATTER = 11


state = struct(forked=False, distributed=False, closed=False)
sync = None  # Will be assigned back by master.
processes = list()


def fork(target, n_parallel, master_rank, *args, **kwargs):
    for rank in [r for r in range(n_parallel) if r != master_rank]:
        w_args = (rank, n_parallel, master_rank, *args)
        processes.append(mp.Process(target=target, args=w_args, kwargs=kwargs))
    for p in processes: p.start()
    state.forked = True
    atexit.register(close)


def check_active():
    if not state.distributed or state.closed:
        raise RuntimeError("Cannot call this function on inactive synkhronos.")


def launch(exct_ID, sub_ID=0):
    sync.ID.value = exct_ID
    sync.sub_ID.value = sub_ID
    sync.barrier_in.wait()


def join():
    sync.barrier_out.wait()
    if not sync.workers_OK.value:
        raise RuntimeError("Encountered worker error during execution loop.")


def close():
    if state.forked and not state.closed:
        sync.quit.value = True
        try:
            sync.barrier_in.wait(1)
        except BrokenBarrierError:
            pass
        state.closed = True
        for p in processes: p.join()


def worker_error_close():
    sync.workers_OK.value = False
    try:
        sync.barrier_out.wait(1)
    except BrokenBarrierError:
        pass
