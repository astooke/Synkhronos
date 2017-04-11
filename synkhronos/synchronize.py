

import multiprocessing as mp
import numpy as np
from ctypes import c_bool


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
        quit=mp.RawValue(c_bool, False),
        ID=mp.RawValue('i', 0),
        sub_ID=mp.RawValue('i', 0),
        barrier_in=mp.Barrier(n_parallel),
        barrier_out=mp.Barrier(n_parallel),
        workers_OK=mp.Value(c_bool, True)  # (not Raw)
    )
    coll = struct(
        nccl=mp.RawValue(c_bool, True),
        op=mp.RawArray('c', 4),
        n_shared=mp.RawValue('i', 0),
        vars=mp.RawArray('i', max_n_var),
        datas=mp.RawArray('i', max_n_var),
        shapes=np_mp_arr('i', max_n_var * max_dim).reshape(max_n_var, max_dim),
        nd_up=mp.RawValue('i', 0),
    )
    data = struct(
        ID=mp.RawValue('I', 0),
        dtype=mp.RawArray('c', 20),  # (by name)
        ndim=mp.RawValue('I', 0),
        shape=np_mp_arr('Q', max_dim),
        tag=mp.RawValue('I', 0),
        alloc_size=mp.RawValue('Q', 0),
    )
    func = struct(
        output_subset=mp.RawArray(c_bool, [True] * max_n_var),
        collect_modes=mp.RawArray('B', max_n_var),
        reduce_ops=mp.RawArray('B', max_n_var),
        n_slices=mp.RawValue('i', 0),
        new_collect=mp.RawValue(c_bool, True),
    )
    scat = struct(
        assign_idxs=np_mp_arr('Q', n_parallel + 1),
        use_idxs_arr=mp.RawValue(c_bool, False),
        tag=mp.RawValue('I', 0),
        size=mp.RawValue('Q', 0),
        idxs_arr=None,  # (allocated later; only need if shuffling)
        data_IDs=mp.RawArray('I', max_n_var),
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
    from . import comms as comm
    from . import exct
    from . import collectives as coll
    from . import data
    from . import function as func
    from . import function_builder as fbld
    from . import scatterer as scat

    comm.sync = syncs.comm
    exct.sync = syncs.exct
    coll.sync = syncs.coll
    data.sync = syncs.data
    func.sync = syncs.func
    fbld.sync = syncs.fbld
    scat.sync = syncs.scat
