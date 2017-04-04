
"""
Constants and utilities used in across master and workers.
"""

import os

import inspect
import synkhronos
PKL_PATH = inspect.getfile(synkhronos).rsplit("__init__.py")[0] + "pkl/"


PID = str(os.getpid())

PRE = "/synk_" + PID


# Exec types
FUNCTION = 0
GPU_COMM = 1
CPU_COMM = 2
DATA = 3
DISTRIBUTE = 4

# DATA OP IDs
DATA_CREATE = 0
DATA_ALLOC = 1
DATA_RESHAPE = 2
DATA_FREE = 3

# COMM IDs
GPU_REDUCE = 0  # NOTE: matches position in COLLECT_MODES
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

# Where to put functions on their way to workers
PKL_FILE = PKL_PATH + "synk_f_dump_" + PID + ".pkl"

# Function Outputs
COLLECT_MODES = ["gpu_reduce", "reduce", "gpu_gather", "gather", None]
REDUCE_OPS = ["avg", "sum", "prod", "min", "max", None]  # (disallow others)
REDUCE_AVG = 0
REDUCE_NO_OP = 7
REDUCE_OPS_WORKER = ["sum", "sum", "prod", "min", "max", None]  # (no average)

DTYPES = ['float64', 'float32', 'float16',
          'int8', 'int16', 'int32', 'int64',
          'uint8', 'uint16', 'uint32', 'uint64',
          ]


def init_gpu(rank, n_gpu, sync, is_master=True):
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
        sync.init.barriers[0].wait()
    if not sync.exct.workers_OK.value:
        return False  # (someone else failed)

    if is_master:
        sync.init.dict["comm_id"] = clique_id.comm_id
        sync.init.barriers[1].wait()
    else:
        sync.init.barriers[1].wait()
        clique_id.comm_id = sync.init.dict["comm_id"]

    try:
        gpu_comm = gpu_coll.GpuComm(clique_id, n_gpu, rank)
    except Exception as e:
        sync.exct.workers_OK.value = False
        raise e
    finally:
        sync.init.barriers[2].wait()

    if not sync.exct.workers_OK.value:
        return False  # (someone else failed)
    else:
        return gpu_comm  # (success)



