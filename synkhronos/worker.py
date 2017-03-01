
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything unique to the workers.
"""

import os
import pickle
from threading import BrokenBarrierError
import atexit

from .variables import Shareds, BaseFunction, BaseData, BaseScatterer
from .common import use_gpu
from .common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                  ALL_GATHER, GATHER, CPU_COMM, SCATTER,
                  DATA_CREATE, DATA_ALLOC, DTYPES, NO_COLLECT,
                  REDUCE_OPS, REDUCE_OPS_WORKER, DATA, DATA_RESHAPE, DATA_FREE)
from .accumulators import Accumulators


CREATE = False


###############################################################################
#                                                                             #
#                               Classes                                       #
#                                                                             #
###############################################################################


class Function(BaseFunction):

    _create = CREATE
    master_rank = None

    def __call__(self, sync_func, scatterer, gpu_comm, accumulators):
        """
        1. Gather the right inputs from mp shared values.
        2. Execute local theano function on those inputs.
        3. Send results back to master.
        """
        if self._n_input > 0:
            scatterer.check_idxs_alloc()
        my_inputs, scatter = scatterer.get_my_inputs(self._n_input)
        output_subset, new_collect, num_slices = self.receive_f_info(sync_func)
        if num_slices == 1 or not any(scatter):
            my_results = self._f(*my_inputs, output_subset=output_subset)
        else:
            if new_collect:
                self._set_accum_fs(accumulators)
            my_results = \
                self._sliced_f(my_inputs, scatter, num_slices, output_subset)
        self.collect_results(my_results, gpu_comm)

    def receive_f_info(self, sync_func):
        if self._n_output == 0:
            return None, False
        output_set = list()
        for i in range(self._n_output):
            if sync_func.output_subset[i]:
                output_set.append(i)
        new_collect = any(output_set != self._output_set) or \
            any(sync_func.collect_modes[output_set] != self._collects) or \
            any(sync_func.reduce_ops[output_set] != self._ops)
        if new_collect:
            self._output_set = output_set
            self._collects = sync_func.collect_modes[output_set]
            self._ops = sync_func.reduce_ops[output_set]
        output_subset = None if len(output_set) == self._n_output else output_set
        return output_subset, new_collect, sync_func.n_slices.value

    def collect_results(self, my_results, gpu_comm):
        if self._n_output == 0:
            return
        if not isinstance(my_results, (list, tuple)):
            my_results = (my_results,)
        for r, mode_ID, op_ID in zip(my_results, self._collects, self._ops):
            if mode_ID == REDUCE:
                op = REDUCE_OPS_WORKER[op_ID]
                gpu_comm.reduce(r, op=op, root=self.master_rank)
            elif mode_ID == GATHER:
                gpu_comm.all_gather(r)
            elif mode_ID != NO_COLLECT:
                raise RuntimeError("Unrecognized collect mode in worker function.")


class Scatterer(BaseScatterer):

    create = False

    def __init__(self, sync, *args):
        super().__init__(*args)
        self.sync = sync
        self.tag = -1

    def get_my_idxs(self):
        self.check_idxs_alloc()
        return super().get_my_idxs()

    def check_idxs_alloc(self):
        """ (lazy update) """
        if self.sync.use_idxs_arr.value:
            if self.tag != self.sync.tag.value:
                size = self.sync.size.value
                self.tag = self.sync.tag.value
                self._alloc_idxs_arr(size, self.tag)

    def get_data(self, data_ID):
        return self.synk_datas[data_ID]


###############################################################################
#                                                                             #
#                             Tasks                                           #
#                                                                             #
###############################################################################


def unpack_functions(theano_functions, n_gpu):
    """
    Worker will recover variables in the same order as the master committed
    them, so they will have the same ID (index).
    """
    synk_functions = list()
    g_shareds = Shareds(n_gpu)
    for idx, fcn in enumerate(theano_functions):
        g_shareds.register_func(fcn)
        synk_functions.append(Function(ID=idx, theano_function=fcn))
    return synk_functions, g_shareds


def receive_distribution(rank, n_gpu, sync_init):
    sync_init.barriers.distribute.wait()
    if not sync_init.distributed.value:
        return False
    with open(PKL_FILE, "rb") as f:
        theano_functions = pickle.load(f)  # should be all in one list
    if sync_init.barriers.delete_pkl.wait() == 0:
        os.remove(PKL_FILE)  # leave no trace
    synk_functions, g_shareds = unpack_functions(theano_functions)
    sync_init.barriers.distribute_out.wait()
    return synk_functions, g_shareds


def do_gpu_comms(comm_ID, sync_comm, g_shareds, gpu_comm, master_rank):
    shared_IDs = sync_comm.vars[:sync_comm.n_shared.value]
    if comm_ID == ALL_GATHER:
        src = g_shareds.get_gpuarray(shared_IDs[0])
        dest = g_shareds.get_gpuarray(shared_IDs[1])
        gpu_comm.all_gather(src, dest)
    else:
        if comm_ID == REDUCE:
            op = REDUCE_OPS_WORKER[sync_comm.op.value]
        elif comm_ID == ALL_REDUCE:
            op = REDUCE_OPS[sync_comm.op.value]
            avg = op == "avg"
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
            g_shareds.call_avg_funcs(shared_IDs)


def do_cpu_comms(comm_ID, sync_comm, scatterer, g_shareds):
    n_shared = sync_comm.n_shared.value
    shared_vars = g_shareds.get_vars_from_IDs(sync_comm.vars[:n_shared])
    if comm_ID == SCATTER:
        my_inputs, _ = scatterer.get_my_inputs(n_shared)
        for var, my_input in zip(shared_vars, my_inputs):
            var.set_value(my_input)
    else:
        raise RuntimeError("Unrecognized CPU comm type in worker.")


def manage_data(data_op, sync_data, scatterer):
    if data_op == DATA_CREATE:
        dtype = DTYPES[sync_data.dtype.value]
        ndim = sync_data.ndim.value
        scatter = sync_data.scatter.value
        scatterer.append(BaseData(dtype, ndim, scatter))
        return
    synk_data = scatterer.get_data(sync_data.ID.value)
    if data_op == DATA_ALLOC:
        ndim = synk_data._ndim
        shape = 1 if ndim == 0 else sync_data.shape[:ndim]
        synk_data._alloc_shmem(sync_data.alloc_size.value, sync_data.tag.value)
        synk_data._shape_data(shape)
    elif data_op == DATA_RESHAPE:
        shape = 1 if synk_data._ndim == 0 else sync_data.shape[:synk_data._ndim]
        synk_data._shape_data(shape)
    elif data_op == DATA_FREE:
        synk_data._free_shmem()
    else:
        raise RuntimeError("Unrecognized data management ID in worker.")


def error_close(sync_exct):
    sync_exct.workers_OK.value = False
    try:
        sync_exct.barrier_out.wait(1)
    except BrokenBarrierError:
        pass


###############################################################################
#                                                                             #
#                          Main Executable                                    #
#                                                                             #
###############################################################################


def worker_exct(rank, n_gpu, master_rank, sync, sync_scat):
    gpu_comm = use_gpu(rank, n_gpu, sync, False)
    if not gpu_comm:
        return  # (exit quietly)

    distribution = receive_distribution(rank, n_gpu, sync.init)
    if not distribution:
        return  # (exit quietly)
    else:
        synk_fs, g_shareds = distribution

    scatterer = Scatterer(sync_scat, n_gpu, rank)
    accumulators = Accumulators()
    Function.master_rank = master_rank

    atexit.register(error_close, sync.exct)

    while True:
        sync.exct.barrier_in.wait()
        if sync.exct.quit.value:
            atexit.unregister(error_close)
            return  # (exit successfully)
        exct_ID = sync.exct.ID.value
        sub_ID = sync.exct.sub_ID.value
        if exct_ID == FUNCTION:
            synk_fs[sub_ID](sync.func, scatterer, gpu_comm, accumulators)
        elif exct_ID == GPU_COMM:
            do_gpu_comms(sub_ID, sync.comm, g_shareds, gpu_comm, master_rank)
        elif exct_ID == CPU_COMM:
            do_cpu_comms(sub_ID, sync.comm, scatterer, g_shareds)
        elif exct_ID == DATA:
            manage_data(sub_ID, sync.data, scatterer)
        else:
            raise RuntimeError("Unrecognized exctution type in worker.")
        sync.exct.barrier_out.wait()  # Prevent premature shmem overwriting.
