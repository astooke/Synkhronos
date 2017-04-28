
import multiprocessing as mp
from threading import BrokenBarrierError
import atexit

from .util import struct

# Exec IDs
DISTRIBUTE = 0
FUNCTION = 1
DATA = 2
GPU_COLL = 3
CPU_COLL = 4
SYNK_COLL = 5

# Data Sub IDs
CREATE = 0
ALLOC = 1
RESHAPE = 2
FREE = 3

# Collectives Sub IDs
BROADCAST = 0
SCATTER = 1
GATHER = 2
ALL_GATHER = 3
REDUCE = 4
ALL_REDUCE = 5
SET_VALUE = 6
GET_VALUE = 7
GET_LENGTHS = 8
GET_SHAPES = 9

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


def init_gpus(rank, n_parallel=None):
    import theano
    import theano.gpuarray
    try:
        theano.gpuarray.use("cuda" + str(rank))
    except Exception as exc:
        if n_parallel is not None:
            raise exc("Master unable to use GPU.")
        else:
            sync.workers_OK.value = False
            raise exc("Worker rank {} unable to use GPU.".format(rank))
    finally:
        sync.barrier_out.wait()
    if n_parallel is not None:
        if sync.workers_OK.value:
            print("Synkhronos: {} GPUs initialized, master rank: {}".format(
                n_parallel, rank))
        else:
            raise RuntimeError("Workers did not initialize GPUs.")
