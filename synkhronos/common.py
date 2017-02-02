
"""
Constants and utilities used in across master and workers.
"""

import os


PID = str(os.getpid())

# Exec types
FUNCTION = 0
GPU_COMM = 1
CPU_COMM = 2

# GPU_COMM IDs
BROADCAST = 0
REDUCE = 1
ALL_REDUCE = 2
ALL_GATHER = 3
GATHER = 4

# CPU_COMM IDs
SCATTER = 0

# Where to put functions on their way to workers
# (possibly need to make this secure somehow?)
PKL_FILE = "synk_function_dump_" + PID + ".pkl"

REDUCE_OPS = {"+": 0,
              "sum": 0,
              "add": 0,
              "*": 1,
              "prod": 1,
              "product": 1,
              "max": 2,
              "maximum": 2,
              "min": 3,
              "minimum": 3,
              "avg": 4,
              "average": 4,
              "mean": 4,
              }

WORKER_OPS = {0: "sum",  # Make sure this is inverse of REDUCE_OPS
              1: "prod",
              2: "max",
              3: "min",
              4: "avg",
              }

AVG_ALIASES = ["avg", "average", "mean"]


def use_gpu(rank, n_gpu, sync, is_master=True):
    """
    Happens after atexit.register(_close) in master and when g.forked=False,
    but before atexit.register(error_close) in workers, so should be careful.
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
        sync.workers_OK.value = False  # (let others know it failed)
        raise e
    finally:
        sync.barriers.gpu_inits[0].wait()
    if not sync.workers_OK.value:
        return False  # (someone else failed)

    if is_master:
        sync.dict["comm_id"] = clique_id.comm_id
        sync.barriers.gpu_inits[1].wait()
    else:
        sync.barriers.gpu_inits[1].wait()
        clique_id.comm_id = sync.dict["comm_id"]

    try:
        gpu_comm = gpu_coll.GpuComm(clique_id, n_gpu, rank)
    except Exception as e:
        sync.workers_OK.value = False
        raise e
    finally:
        sync.barriers.gpu_inits[2].wait()

    if not sync.workers_OK.value:
        return False  # (someone else failed)
    else:
        return gpu_comm  # (success)
