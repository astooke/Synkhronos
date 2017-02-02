"""
Classes and functions used by master but which don't MODIFY globals.
(Might still read from globals passed explicitly as parameter.)
"""

import numpy as np
import multiprocessing as mp
import ctypes

from .common import REDUCE_OPS, AVG_ALIASES
from .variables import struct, SynkFunction


COLLECT_MODES = ["reduce", "gather"]


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
        delete_pkl=mp.Barrier(n_gpu - 1),
        exec_in=mp.Barrier(n_gpu),
        exec_out=mp.Barrier(n_gpu),
    )
    sync = struct(
        dict=dictionary,  # use for setup e.g. Clique comm_id; serializes.
        quit=mp.RawValue(ctypes.c_bool, False),
        workers_OK=mp.Value(ctypes.c_bool, True),  # (not RawValue)
        n_user_fcns=mp.RawValue('i', 0),
        distributed=mp.RawValue(ctypes.c_bool, False),
        exec_type=mp.RawValue('i', 0),
        func_ID=mp.RawValue('i', 0),
        comm_ID=mp.RawValue('i', 0),
        n_shared=mp.RawValue('i', 0),
        barriers=barriers,
    )
    return sync


###############################################################################
#                           (distribute)                                      #


def get_worker_reduce_ops(synk_functions):
    reduce_ops_all = [fn.reduce_ops for fn in synk_functions]
    for ops in reduce_ops_all:
        for idx, op in enumerate(ops):
            if op in AVG_ALIASES:
                ops[idx] = "sum"  # (no averaging of outputs in workers)
    return reduce_ops_all


###############################################################################
#                           (function)                                        #


def check_func_scatter(inputs, broadcast_inputs, scatter_inputs):
    if broadcast_inputs is not None and scatter_inputs is not None:
        raise ValueError("May specify either broadcast_inputs or "
            "scatter_inputs but not both.")
    if broadcast_inputs is None and scatter_inputs is None:
        inputs_scatter = [True] * len(inputs)  # (default is to scatter all)
    elif broadcast_inputs is not None:
        if not isinstance(broadcast_inputs, (tuple, list)):
            raise TypeError("Optional param broadcast_inputs must be list or "
                "tuple.")
        inputs_scatter = [True] * len(inputs)
        for bc_inpt in broadcast_inputs:
            if bc_inpt not in inputs:
                raise ValueError("Elements of param broadcast_inputs must "
                    "be in the list of inputs provided.")
            inputs_scatter[inputs.index(bc_inpt)] = False
    else:  # (scatter_inputs is not None)
        if not isinstance(scatter_inputs, (list, tuple)):
            raise TypeError("Optional param scatter_inputs must be list or "
                "tuple.")
        inputs_scatter = [False] * len(inputs)
        for sc_inpt in scatter_inputs:
            if sc_inpt not in inputs:
                raise ValueError("Elements of param scatter_inputs must "
                    "be in the list of inputs provided.")
            inputs_scatter[inputs.index(sc_inpt)] = True
    return inputs_scatter


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
#                           GPU Collectives                                   #


def get_shared_IDs(g_shareds, synk_functions=None, shared_vars=None):
    """ this one is clean from global accesses """
    if synk_functions is None and shared_vars is None:
        return tuple(range(g_shareds.num))  # default is all shareds
    else:
        # Type and existence checking.
        if synk_functions is None:
            synk_functions = []
        else:
            if not isinstance(synk_functions, (list, tuple)):
                synk_functions = (synk_functions,)
            for synk_fcn in synk_functions:
                if not isinstance(synk_fcn, SynkFunction):
                    raise TypeError("Expected Synkhronos function(s).")
        if shared_vars is None:
            shared_vars = []
        else:
            if not isinstance(shared_vars, (list, tuple)):
                shared_vars = (shared_vars,)
            for var in shared_vars:
                if var is None:
                    raise ValueError("Received None for one or mored shared "
                        "variables.")
                if var not in g_shareds.names and var not in g_shareds.vars:
                    raise ValueError("Unrecognized shared variable or name: ", var)

        shared_IDs = list()
        for synk_fcn in synk_functions:
            shared_IDs += synk_fcn._shared_IDs
        for var in shared_vars:
            if var in g_shareds.names:
                shared_IDs.append(g_shareds.names.index(var))
            else:
                shared_IDs.append(g_shareds.vars.index(var))
    return tuple(sorted(set(shared_IDs)))


def check_op(op):
    if op not in REDUCE_OPS:
        raise ValueError("Unrecognized reduction operator: ", op,
            ", must be one of: ", [k for k in REDUCE_OPS.keys()])
    return REDUCE_OPS[op]


###############################################################################
#                       CPU Comm                                              #


def check_shared_var(g_shareds, shared_var):
    if shared_var not in g_shareds.vars and shared_var not in g_shareds.names:
        raise ValueError("Unrecognized theano shared variable or name: ",
            shared_var)
    if shared_var in g_shareds.vars:
        shared_ID = g_shareds.vars.index(shared_var)
    else:
        shared_ID = g_shareds.names.index(shared_var)
        shared_var = g_shareds.vars[shared_ID]
    return shared_var, shared_ID


def check_scatter_sources(g_shareds, n_gpu, sources, shared_ID):
    if not isinstance(sources, (tuple, list)):
        raise TypeError("Param sources must be a list or tuple of arguments, "
            "each for shared.set_value().")
    if len(sources) != n_gpu:
        raise ValueError("Source list must have as many elements as there are "
            "GPUs.")
    shape = g_shareds.gpuarrays[shared_ID].shape
    dtype = g_shareds.vars[shared_ID].type.dtype
    for idx, src in enumerate(sources):
        if not isinstance(src, np.ndarray):
            src = np.asarray(src, dtype=dtype)
            sources[idx] = src
        elif src.dtype != dtype:
            raise TypeError("Must provide the same data type as the shared "
                "var: ", dtype)
        if src.shape != shape:
            raise ValueError("Source is not same shape as shared variable: ",
                shape)
    return sources
