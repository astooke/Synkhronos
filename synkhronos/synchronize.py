

import multiprocessing as mp
import numpy as np
import ctypes
# import ipdb

from .util import struct


def np_mp_arr(t_or_tc, size_or_init):
    return np.ctypeslib.as_array(mp.RawArray(t_or_tc, size_or_init))


def build_syncs(n_parallel, max_n_var=100, max_dim=16):

    mgr = mp.Manager()
    dictionary = mgr.dict()
    comm = struct(
        dict=dictionary,
        semaphore=mp.Semaphore(0),
        barrier=mp.Barrier(n_parallel),
    )
    fbld = struct(
        barrier=mp.Barrier(n_parallel - 1),
    )
    exct = struct(
        quit=mp.RawValue(ctypes.c_bool, False),
        ID=mp.RawValue(ctypes.c_uint, 0),
        sub_ID=mp.RawValue(ctypes.c_uint, 0),
        barrier_in=mp.Barrier(n_parallel),
        barrier_out=mp.Barrier(n_parallel),
        workers_OK=mp.Value(ctypes.c_bool, True)  # (not Raw)
    )
    coll = struct(
        nccl=mp.RawValue(ctypes.c_bool, True),
        op=mp.RawArray(ctypes.c_char, 4),  # 'c'
        n_shared=mp.RawValue(ctypes.c_uint, 0),
        vars=mp.RawArray(ctypes.c_uint, max_n_var),
        datas=mp.RawArray(ctypes.c_uint, max_n_var),
        shapes=np_mp_arr(ctypes.c_uint, max_n_var * max_dim).reshape(max_n_var, max_dim),
        nd_up=mp.RawValue(ctypes.c_uint, 0),
        rank=mp.RawValue(ctypes.c_uint, 0),
    )
    data = struct(
        ID=mp.RawValue('i', 0),
        dtype=mp.RawArray(ctypes.c_char, 20),  # (by name)
        ndim=mp.RawValue('i', 0),
        minibatch=mp.RawValue(ctypes.c_bool, False),
        shape=np_mp_arr('i', max_dim),
        tag=mp.RawValue('i', 0),
        alloc_size=mp.RawValue(ctypes.c_ulong, 0),
    )
    func = struct(
        output_subset=mp.RawArray(ctypes.c_bool, [True] * max_n_var),
        n_slices=mp.RawValue('i', 0),
        is_new_subset=mp.RawValue(ctypes.c_bool, False),
    )
    scat = struct(
        assign_idxs=np_mp_arr(ctypes.c_ulong, n_parallel + 1),
        use_idxs_arr=mp.RawValue(ctypes.c_bool, False),
        tag=mp.RawValue(ctypes.c_uint, 0),
        size=mp.RawValue(ctypes.c_ulong, 0),
        idxs_arr=None,  # (allocated later; only need if shuffling)
        data_IDs=mp.RawArray(ctypes.c_uint, max_n_var),
        use_batch_s=mp.RawValue(ctypes.c_bool, False),
        assign_idxs_s=np_mp_arr(ctypes.c_ulong, n_parallel * 2).reshape(n_parallel, 2),
        use_idxs_arr_s=mp.RawValue(ctypes.c_bool, False),
        tag_s=mp.RawValue(ctypes.c_uint, 0),
        size_s=mp.RawValue(ctypes.c_ulong, 0),
        idxs_arr_s=None,
    )

    syncs = struct(
        comm=comm,
        fbld=fbld,
        exct=exct,
        coll=coll,
        data=data,
        func=func,
        scat=scat,
    )
    return syncs


def give_syncs(syncs):
    from . import comm
    from . import exct
    from . import collectives as coll
    from . import data_module as data
    from . import function_module as func
    from . import function_builder as fbld
    from . import scatterer as scat

    comm.sync = syncs.comm
    exct.sync = syncs.exct
    coll.sync = syncs.coll
    data.sync = syncs.data
    func.sync = syncs.func
    fbld.sync = syncs.fbld
    scat.sync = syncs.scat
