
"""
Constants and utilities used in across master and workers.
"""

import os
from .shmemarray import NpShmemArray


PID = str(os.getpid())

PRE = "/synk_" + PID
SCAT_IDXS_TAG = PRE + "_scat_idxs_"
SHRD_IDS_TAG = PRE + "_shared_ids_"
DATA_IDS_TAG = PRE + "_data_ids_"

# Exec types
FUNCTION = 0
GPU_COMM = 1
CPU_COMM = 2
DATA = 3

# DATA OP IDs
DATA_CREATE = 0
DATA_ALLOC = 1
DATA_RESHAPE = 2
DATA_FREE = 3

# GPU_COMM IDs
REDUCE = 0  # NOTE: matches position in COLLECT_MODES
GATHER = 1  # NOTE: matches position in COLLECT_MODES
BROADCAST = 2
ALL_REDUCE = 3
ALL_GATHER = 4

# CPU_COMM IDs
SCATTER = 0

# Where to put functions on their way to workers
# (possibly need to make this secure somehow?)
PKL_FILE = "synk_f_dump_" + PID + ".pkl"

# Function Outputs
COLLECT_MODES = ["reduce", "gather", None]  # NOTE: matches GPU_COMM IDs
NO_COLLECT = 2
REDUCE_OPS = ["avg", "sum", "prod", "min", "max", None]  # (disallow others)
REDUCE_AVG = 0
REDUCE_NO_OP = 7
REDUCE_OPS_WORKER = ["sum", "sum", "prod", "min", "max", None]  # (no average)

DTYPES = ['float64', 'float32', 'float16',
          'int8', 'int16', 'int32', 'int64',
          'uint8', 'uint16', 'uint32', 'uint64',
          ]


def use_gpu(rank, n_gpu, sync, is_master=True):
    """
    Happens after atexit.register(_close) in master and when g.forked=False,
    but before atexit.register(error_close) in workers, so should be careful.

    TODO: probably can simplify or otherwise improve the error catching.
    """
    dev_str = "cuda" + str(rank)
    try:
        import theano.gpuarray
        theano.gpuarray.use(dev_str)
        from pygpu import collectives as gpu_coll
        gpu_ctx = theano.gpuarray.get_context(None)
        clique_id = gpu_coll.GpuCommCliqueId(gpu_ctx)
    except ImportError as e:
        if is_master:
            raise e  # (only master raises ImportError, will join subprocesses)
        else:
            return  # (workers exit quietly)
    except Exception as e:
        sync.exct.workers_OK.value = False  # (let others know it failed)
        raise e
    finally:
        sync.init.barriers.gpu_inits[0].wait()
    if not sync.exct.workers_OK.value:
        return False  # (someone else failed)

    if is_master:
        sync.init.dict["comm_id"] = clique_id.comm_id
        sync.init.barriers.gpu_inits[1].wait()
    else:
        sync.init.barriers.gpu_inits[1].wait()
        clique_id.comm_id = sync.init.dict["comm_id"]

    try:
        gpu_comm = gpu_coll.GpuComm(clique_id, n_gpu, rank)
    except Exception as e:
        sync.exct.workers_OK.value = False
        raise e
    finally:
        sync.init.barriers.gpu_inits[2].wait()

    if not sync.exct.workers_OK.value:
        return False  # (someone else failed)
    else:
        return gpu_comm  # (success)


def _alloc_scat_idxs(size, tag, create):
    tag = SCAT_IDXS_TAG + str(tag)
    return NpShmemArray('int32', size, tag, create)


def alloc_shared_IDs(size, create):
    shared_IDs = NpShmemArray('uint32', size, SHRD_IDS_TAG, create)
    data_IDs = NpShmemArray('uint32', size, DATA_IDS_TAG, create)
    return shared_IDs, data_IDs


def get_my_scat_idxs(sync_scat, rank):
    my_idxs = slice(*sync_scat.assign_idxs[rank:rank + 2])
    if sync_scat.use_idxs_arr.value:
        my_idxs = sync_scat.idxs_arr[my_idxs]
    return my_idxs
