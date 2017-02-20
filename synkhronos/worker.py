
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything unique to the workers.
"""

import os
import pickle
from threading import BrokenBarrierError
import atexit

from .variables import Inputs, Shareds, BaseFunction, BaseData
from .common import use_gpu, _alloc_scat_idxs, alloc_shared_IDs, get_my_scat_idxs
from .common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                  ALL_GATHER, GATHER, WORKER_OPS, AVG_ALIASES, CPU_COMM, SCATTER,
                  DATA_CREATE, DATA_ALLOC, DTYPES, COLLECT_MODES,
                  REDUCE_OPS_WORKER, DATA, DATA_RESHAPE, DATA_FREE)


CREATE = False


class Function(BaseFunction):

    _create = CREATE
    rank = None
    master_rank = None

    def __call__(self, sync_scat, g_in_datas, gpu_comm):
        """
        1. Gather the right inputs from mp shared values.
        2. Execute local theano function on those inputs.
        3. Send results back to master.
        """
        self._check_idxs_alloc(sync_scat)
        my_inputs = self._get_my_inputs(sync_scat, g_in_datas)
        output_subset, output_set = self._receive_output_subset()
        my_results = self._f(*my_inputs, output_subset=output_subset)
        self._collect_results(my_results, gpu_comm, output_set)

    def check_idxs_alloc(self, sync_scat):
        if self._n_input == 0:
            return
        if sync_scat.use_idxs_arr.value:
            if sync_scat.my_idxs_tag != sync_scat.idxs_tag.value:
                alloc_scat_idxs(sync_scat)

    def _receive_output_subset(self):
        if self._n_output == 0:
            return None, None
        output_set = [i for i, x in enumerate(self._sync.output_subset) if x]
        output_subset = None if all(output_set) else output_set
        return output_subset, output_set

    def _collect_results(self, my_results, gpu_comm, output_set):
        if self._n_output == 0:
            return
        if not isinstance(my_results, (list, tuple)):
            my_results = (my_results,)
        for idx, r in zip(output_set, my_results):
            mode = self._sync.collect_modes[idx]
            if mode == REDUCE:
                op = REDUCE_OPS_WORKER[self._sync.reduce_ops[idx]]
                gpu_comm.reduce(r, op=op, root=self.master_rank)
            elif mode == GATHER:
                gpu_comm.all_gather(r)
            elif mode >= len(COLLECT_MODES):
                raise RuntimeError("Unrecognized collect mode in worker function.")


def unpack_functions(theano_functions, n_fcn):
    """
    Worker will recover variables in the same order as the master committed
    them, so they will have the same ID (index).
    """
    synk_functions = list()
    g_inputs = Inputs()
    g_shareds = Shareds(CREATE)
    for idx, fcn in enumerate(theano_functions[:n_fcn]):
        g_inputs.register_func(fcn)
        g_shareds.register_func(fcn)
        synk_functions.append(Function(theano_function=fcn))
    g_shareds.avg_functions = theano_functions[n_fcn:]
    # g_shareds.unpack_avg_facs()  # (only needed for changing avg_fac later)
    return synk_functions, g_inputs, g_shareds


def receive_distribution(rank, n_gpu, sync_init, sync_IDs):
    sync_init.barriers.distribute.wait()
    if not sync_init.distributed.value:
        return False
    with open(PKL_FILE, "rb") as f:
        theano_functions = pickle.load(f)  # should be all in one list
    if sync_init.barriers.delete_pkl.wait() == 0:
        os.remove(PKL_FILE)  # leave no trace
    synk_functions, g_inputs, g_shareds = \
        unpack_functions(theano_functions, sync_init.n_user_fcns.value)
    sync_IDs.vars, sync_IDs.datas = alloc_shared_IDs(len(g_shareds.vars), CREATE)
    sync_init.barriers.distribute_out.wait()
    return synk_functions, g_inputs, g_shareds


def do_gpu_comms(sync_IDs, g_shareds, gpu_comm, master_rank):
    shared_IDs = sync_IDs.vars[:sync_IDs.n_shared.value]
    comm_ID = sync_IDs.comm.value
    if comm_ID == ALL_GATHER:
        src = g_shareds.get_gpuarray(shared_IDs[0])
        dest = g_shareds.get_gpuarray(shared_IDs[1])
        gpu_comm.all_gather(src, dest)
    else:
        if comm_ID in [REDUCE, ALL_REDUCE]:
            op = WORKER_OPS[sync_IDs.op.value]
            avg = op in AVG_ALIASES
            op = "sum" if avg else op
        for shared_ID in shared_IDs:
            src = g_shareds.get_gpuarray(shared_ID)
            if comm_ID == BROADCAST:
                gpu_comm.broadcast(src, root=master_rank)
            elif comm_ID == REDUCE:
                gpu_comm.reduce(src, op=op, root=master_rank)
            elif comm_ID == ALL_REDUCE:
                gpu_comm.all_reduce(src, op=op, dest=src)
            elif comm_ID == GATHER:
                gpu_comm.all_gather(src)
            else:
                raise RuntimeError("Unrecognized GPU communication type in "
                    "worker.")
        if comm_ID == ALL_REDUCE and avg:
            for shared_ID in shared_IDs:
                g_shareds.avg_functions[shared_ID]()


def do_cpu_comms(sync_IDs, sync_scat, g_shareds, g_datas, rank):
    comm_ID = sync_IDs.comm.value
    n_shared = sync_IDs.n_shared.value
    shared_IDs = sync_IDs.vars[:n_shared]
    data_IDs = sync_IDs.datas[:n_shared]
    if comm_ID == SCATTER:
        my_idxs = get_my_scat_idxs(sync_scat, rank)
        for shared_ID, data_ID in zip(shared_IDs, data_IDs):
            g_shareds.vars[shared_ID].set_value(g_datas[data_ID].data[my_idxs])
    else:
        raise RuntimeError("Unrecognized CPU comm type in worker.")


def alloc_scat_idxs(sync_scat):
    size = sync_scat.idxs_size.value
    sync_scat.my_idxs_tag = sync_scat.idex_tag.value
    sync_scat.idxs_arr = _alloc_scat_idxs(size, sync_scat.my_idxs_tag, CREATE)


def manage_data(sync_IDs, g_datas):
    data_op = sync_IDs.op.value
    if data_op == DATA_CREATE:
        dtype = DTYPES[sync_IDs.dtype.value]
        ndim = sync_IDs.ndim.value
        data_ID = len(g_datas)
        g_datas.append(BaseData(data_ID, dtype, ndim))
    elif data_op == DATA_ALLOC:
        synk_data = g_datas[sync_IDs.data.value]
        ndim = synk_data.ndim
        shape = 1 if ndim == 0 else sync_IDs.shape[:ndim]
        tag = sync_IDs.tag.value
        synk_data._alloc_shmem(shape, tag)
    elif data_op == DATA_RESHAPE:
        synk_data = g_datas[sync_IDs.data.value]
        shape = sync_IDs.shape[:synk_data._ndim]
        synk_data._reshape_shmem(shape)
    elif data_op == DATA_FREE:
        synk_data = g_datas[sync_IDs.data.value]
        synk_data._free_shmem()
    else:
        raise RuntimeError("Unrecognized data management ID in worker.")


def error_close(sync_exct):
    sync_exct.workers_OK.value = False
    try:
        sync_exct.barrier_out.wait(1)
    except BrokenBarrierError:
        pass


def worker_exct(rank, n_gpu, master_rank, sync):
    gpu_comm = use_gpu(rank, n_gpu, sync, False)
    if not gpu_comm:
        return  # (exit quietly)

    distribution = receive_distribution(rank, n_gpu, sync.init, sync.IDs)
    if not distribution:
        return  # (exit quietly)
    else:
        synk_functions, g_inputs, g_shareds = distribution

    Function.rank = rank  # endow all functions
    Function.master_rank = master_rank
    g_datas = list()
    sync.scat.my_idxs_tag = -1

    atexit.register(error_close, sync.exct)

    while True:
        sync.exct.barrier_in.wait()
        if sync.exct.quit.value:
            atexit.unregister(error_close)
            return  # (exit successfully)
        if sync.exct.ID.value == FUNCTION:
            synk_functions[sync.sub_ID.func.value](sync.scat, g_inputs, gpu_comm)
        elif sync.exct.ID.value == GPU_COMM:
            do_gpu_comms(sync.IDs, g_shareds, gpu_comm, master_rank)
        elif sync.exct.ID.value == CPU_COMM:
            do_cpu_comms(sync.IDs, sync.scat, g_shareds, g_datas, rank)
        elif sync.exct.ID.value == DATA:
            manage_data(sync.IDs, g_datas)

        else:
            raise RuntimeError("Unrecognized exctution type in worker.")
        sync.exct.barrier_out.wait()  # Prevent premature shmem overwriting.
