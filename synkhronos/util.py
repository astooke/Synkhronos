"""
Classes and functions used by master but which don't MODIFY globals.
(Might still read from globals passed explicitly as parameter.)
"""

import numpy as np
import multiprocessing as mp
import ctypes

from .common import REDUCE_OPS, AVG_ALIASES
from .variables import struct, SynkFunction


COLLECT_MODES = ["reduce", "gather", None]


###############################################################################
#                              (fork)                                         #


def n_gpu_getter(mp_n_gpu):
    """
    Call in a subprocess because it prevents future subprocesses from using GPU.
    """
    from pygpu import gpuarray
    mp_n_gpu.value = gpuarray.count_devices("cuda", 0)


def get_n_gpu(n_gpu, master_rank):
    master_rank = int(master_rank)
    if n_gpu is not None:
        n_gpu = int(n_gpu)
    else:
        #  Detect the number of devices present and use all.
        mp_n_gpu = mp.RawValue('i', 0)
        p = mp.Process(target=n_gpu_getter, args=(mp_n_gpu,))
        p.start()
        p.join()
        n_gpu = mp_n_gpu.value
        if n_gpu == 0:
            raise RuntimeError("No cuda GPU detected by pygpu.")
        elif n_gpu == 1:
            raise RuntimeError("Only one GPU detected; just use Theano.)")
        else:
            print("Detected and attempting to use {} GPUs.".format(n_gpu))

    if master_rank not in list(range(n_gpu)):
        raise ValueError("Invalid value for master rank: ", master_rank)

    return n_gpu, master_rank


def build_sync(n_gpu):

    mgr = mp.Manager()
    dictionary = mgr.dict()
    barriers = struct(
        gpu_inits=[mp.Barrier(n_gpu) for _ in range(3)],
        distribute=mp.Barrier(n_gpu),
        distribute_out=mp.Barrier(n_gpu),
        delete_pkl=mp.Barrier(n_gpu - 1),
        exec_in=mp.Barrier(n_gpu),
        exec_out=mp.Barrier(n_gpu),
    )
    scat = struct(
        assign_idx=np.ctypeslib.as_array(mp.RawArray('i', n_gpu + 1)),
        use_idxs=mp.RawValue(ctypes.c_bool, False),
        idxs_tag=mp.RawValue('i', 0),
        idxs_size=mp.RawValue('i', 0),
        idxs_arr=None,
    )
    sync = struct(
        dict=dictionary,  # use for setup e.g. Clique comm_id; serializes.
        quit=mp.RawValue(ctypes.c_bool, False),
        workers_OK=mp.Value(ctypes.c_bool, True),  # (not RawValue)
        n_user_fcns=mp.RawValue('i', 0),
        distributed=mp.RawValue(ctypes.c_bool, False),
        exec_ID=mp.RawValue('i', 0),
        func_ID=mp.RawValue('i', 0),
        comm_ID=mp.RawValue('i', 0),
        op_ID=mp.RawValue('i', 0),
        n_shared=mp.RawValue('i', 0),
        scat=scat,
        barriers=barriers,
    )
    return sync


###############################################################################
#                           (distribute)                                      #


def get_worker_reduce_ops(synk_functions):
    reduce_ops_all = [fn.reduce_ops.copy() for fn in synk_functions]
    for ops in reduce_ops_all:
        for idx, op in enumerate(ops):
            if op in AVG_ALIASES:
                ops[idx] = "sum"  # (no averaging of outputs in workers)
    return reduce_ops_all


###############################################################################
#                           (function)                                        #


def check_collect(outputs, collect_modes, reduce_ops):
    if outputs is None:
        return [], []
    n_outputs = len(outputs) if isinstance(outputs, (list, tuple)) else 1
    if not isinstance(collect_modes, (list, tuple)):
        collect_modes = [collect_modes] * n_outputs
    if len(collect_modes) != n_outputs:
        raise ValueError("Number of collect modes does not match number of "
            "outputs (or enter a single string to be used for all outputs).")
    for mode in collect_modes:
        if mode not in COLLECT_MODES:
            raise ValueError("Unrecognized collect_mode: ", mode,
                " .  Must be in: ", COLLECT_MODES)
    if not isinstance(reduce_ops, (list, tuple)):
        tmp_ops = list()
        for mode in collect_modes:
            if mode == "reduce":
                tmp_ops.append(reduce_ops)
            else:
                tmp_ops.append(None)
        reduce_ops = tmp_ops
    if len(reduce_ops) != n_outputs:
        raise ValueError("Number of reduce ops does not match number of "
            "outputs (use None for non-reduce outputs, or a single string for "
            "all reduced outputs).")
    else:
        for idx, op in enumerate(reduce_ops):
            if collect_modes[idx] == "reduce":
                if op not in REDUCE_OPS:
                    raise ValueError("Unrecognized reduce op: ", op)
    return collect_modes, reduce_ops


###############################################################################
#                       Shared Memory Management                              #

def _assign_scat_idx(n_gpu, shmem_sizes, scat_arg):
    if scat_arg is not None:
        if isinstance(scat_arg, int):  # (scat_size is used)
            max_idx = scat_arg
            space_start = 0
            space_end = scat_arg
        elif isinstance(scat_arg, slice):  # (scat_slice is used)
            max_idx = scat_arg.stop
            space_start = scat_arg.start
            space_end = scat_arg.stop
        else:  # (scat_idxs is used)
            max_idx = max(scat_arg)
            space_start = 0
            space_end = len(scat_arg)
        if max_idx > min(shmem_sizes):
            raise ValueError("Requested index out of range of shared memory.")
    else:  # (i.e. no scat input directive provided, use full arrays)
        space_start = 0
        space_end = shmem_sizes[0]
        if shmem_sizes.count(space_end) != len(shmem_sizes):  # (fast)
            raise ValueError("If not providing scatter directive, all "
                "shared memory arrays must be the same size in 0-th "
                "dimension.  Had 0-th dim sizes: ", shmem_sizes)
    return np.linspace(space_start, space_end, n_gpu + 1, dtype=np.int32)


def check_scat_types(scat_idxs, scat_slice, scat_size):
    if sum([scat_idxs is None, scat_slice is None, scat_size is None]) < 2:
        raise ValueError("Specify only one of: scat_idxs, scat_slice, scat_size.")
    if scat_idxs is not None:
        if not isinstance(scat_idxs, (list, tuple)):
            if isinstance(scat_idxs, np.ndarray):
                if scat_idxs.ndim > 1:
                    raise ValueError("Numpy array for scat_idxs must "
                        "be 1 dimensional, got: ", scat_idxs.ndim)
            else:
                raise TypeError("Param scat_idxs must be a list, tuple "
                    "or 1-dimensional numpy array of ints.")
        return scat_idxs
    elif scat_slice is not None:
        if not isinstance(scat_slice, slice):
            if not isinstance(scat_slice, slice):
                raise TypeError("Param scat_slice must be a slice object.")
            if scat_slice.step is not None and scat_slice.step > 1:
                raise NotImplementedError  # (could be done)
        return scat_slice
    elif scat_size is not None:
        if not isinstance(scat_size, int):
            raise TypeError("Param scat_size must be an integer.")
        return scat_size
    else:
        return None


###############################################################################
#                           GPU Collectives                                   #


def get_shared_IDs(g_shareds, shared_vars=None, synk_functions=None):
    shared_IDs = list()
    if synk_functions is not None:
        if not isinstance(synk_functions, (list, tuple)):
            synk_functions = (synk_functions,)
        for synk_fcn in synk_functions:
            if not isinstance(synk_fcn, SynkFunction):
                raise TypeError("Expected Synkhronos function(s).")
            shared_IDs += synk_fcn._shared_IDs
    if shared_vars is not None:
        if not isinstance(shared_vars, (list, tuple)):
            shared_vars = (shared_vars,)
        for var in shared_vars:
            if var is None:
                raise ValueError("Received None for one or mored shared "
                    "variables.")
            if var in g_shareds.names:
                shared_IDs.append(g_shareds.names.index(var))
            elif var in g_shareds.vars:
                shared_IDs.append(g_shareds.vars.index(var))
            else:
                raise ValueError("Unrecognized shared variable or name: ", var)
    return tuple(sorted(set(shared_IDs)))


def check_op(op):
    if op not in REDUCE_OPS:
        raise ValueError("Unrecognized reduction operator: ", op,
            ", must be one of: ", [k for k in REDUCE_OPS.keys()])
    return REDUCE_OPS[op]
