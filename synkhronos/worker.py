
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything unique to the workers.
"""

import os
import pickle
from threading import BrokenBarrierError
import atexit


from .variables import Shareds, BaseFunction, BaseData, BaseScatterer
from .cpu_comm import CpuCommWorker
from .common import *
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

    def __call__(self, sync_func, scatterer, gpu_comm, cpu_comm, accumulators):
        """
        1. Gather the right inputs from mp shared values.
        2. Execute local theano function on those inputs.
        3. Send results back to master.
        """
        if self._n_input > 0:
            scatterer.check_idxs_alloc()
        my_inputs, scatter = scatterer.get_my_inputs(self._n_input)
        output_subset, num_slices = self.receive_f_info(sync_func, accumulators)
        my_results = \
            self._sliced_f(my_inputs, scatter, num_slices, output_subset) \
            if num_slices > 1 and any(scatter) else \
            self._f(*my_inputs, output_subset=output_subset)
        self.collect_results(my_results, gpu_comm, cpu_comm)

    def receive_f_info(self, sync_func, accumulators):
        num_slices = sync_func.n_slices.value
        if self._n_output == 0:
            return None, num_slices
        if sync_func.new_collect.value:
            o_set = [i for i in range(self._n_output)
                if sync_func.output_subset[i]]
            self._output_set = o_set
            self._collects = list()
            self._ops = list()
            for o in o_set:
                self._collects.append(sync_func.collect_modes[o])
                self._ops.append(sync_func.reduce_ops[o])
            if num_slices > 1:
                self._set_accum_fs(accumulators)
        output_subset = \
            None if len(self._output_set) == self._n_output else self._output_set
        return output_subset, num_slices

    def collect_results(self, my_results, gpu_comm, cpu_comm):
        if self._n_output == 0:
            return
        if not isinstance(my_results, (list, tuple)):
            my_results = (my_results,)
        for r, mode_ID, op_ID in zip(my_results, self._collects, self._ops):
            if mode_ID == GPU_REDUCE:
                op = REDUCE_OPS_WORKER[op_ID]
                gpu_comm.reduce(r, op=op, root=self.master_rank)
            elif mode_ID == GPU_GATHER:
                gpu_comm.all_gather(r)
            elif mode_ID in [CPU_REDUCE, CPU_GATHER]:
                cpu_comm.send(r)
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


def unpack_functions(theano_functions, accumulators):
    """
    Worker will recover variables in the same order as the master committed
    them, so they will have the same ID (index).
    """
    synk_functions = list()
    g_shareds = Shareds()
    for idx, fcn in enumerate(theano_functions):
        g_shareds.register_func(fcn, accumulators)
        synk_functions.append(Function(ID=idx, theano_function=fcn))
    return synk_functions, g_shareds


def receive_distribution(rank, sync_init, accumulators):
    sync_init.barriers.distribute.wait()
    if not sync_init.distributed.value:
        return False, None
    with open(PKL_FILE, "rb") as f:
        theano_functions = pickle.load(f)  # should be all in one list
    if sync_init.barriers.delete_pkl.wait() == 0:
        os.remove(PKL_FILE)  # leave no trace
    synk_functions, g_shareds = unpack_functions(theano_functions, accumulators)
    sync_init.barriers.distribute_out.wait()
    return synk_functions, g_shareds


def do_gpu_comms(comm_ID, sync_comm, g_shareds, gpu_comm, master_rank):
    shared_IDs = sync_comm.vars[:sync_comm.n_shared.value]
    if comm_ID == GPU_ALL_GATHER:
        src = g_shareds.get_array(shared_IDs[0])
        dest = g_shareds.get_array(shared_IDs[1])
        gpu_comm.all_gather(src, dest)
    else:
        if comm_ID == GPU_REDUCE:
            op = REDUCE_OPS_WORKER[sync_comm.op.value]
        elif comm_ID == GPU_ALL_REDUCE:
            op = REDUCE_OPS[sync_comm.op.value]
            avg = op == "avg"
            op = "sum" if avg else op
        for shared_ID in shared_IDs:
            src = g_shareds.get_array(shared_ID)
            if comm_ID == GPU_BROADCAST:
                gpu_comm.broadcast(src, root=master_rank)
            elif comm_ID == GPU_REDUCE:
                gpu_comm.reduce(src, op=op, root=master_rank)
            elif comm_ID == GPU_ALL_REDUCE:
                gpu_comm.all_reduce(src, op=op, dest=src)
            elif comm_ID == GPU_GATHER:
                gpu_comm.all_gather(src)
            else:
                raise RuntimeError("Unrecognized GPU communication type in "
                    "worker.")
        if comm_ID == GPU_ALL_REDUCE and avg:
            g_shareds.call_avg_fs(shared_IDs)


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
    ndim = synk_data._ndim
    shape = 1 if ndim == 0 else sync_data.shape[:ndim]
    if data_op == DATA_ALLOC:
        synk_data._alloc_shmem(sync_data.alloc_size.value, sync_data.tag.value)
        synk_data._shape_data(shape)
    elif data_op == DATA_RESHAPE:
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


def profiling_worker(*args, **kwargs):
    import cProfile
    cProfile.runctx('worker_exct(*args, **kwargs)', locals(), globals(),
        "train_only_no_red_5itr.prof")


def worker_exct(rank, n_parallel, master_rank, sync, sync_scat):

    sync.init.semaphore.acquire()  # blocks worker but not master
    cpu_comm = CpuCommWorker(sync.init.dict["ports"][rank])

    gpu_comm = init_gpu(rank, n_parallel, sync, False)
    if not gpu_comm:
        return  # (exit quietly)


    accumulators = Accumulators()
    synk_fs, g_shareds = receive_distribution(rank, sync.init, accumulators)
    if not synk_fs:
        return  # (exit quietly; distro failed somehow)
    g_shareds.set_n_parallel(n_parallel)
    scatterer = Scatterer(sync_scat, n_parallel, rank)
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
            synk_fs[sub_ID](sync.func, scatterer, gpu_comm, cpu_comm, accumulators)
        elif exct_ID == GPU_COMM:
            do_gpu_comms(sub_ID, sync.comm, g_shareds, gpu_comm, master_rank)
        elif exct_ID == CPU_COMM:
            do_cpu_comms(sub_ID, sync.comm, scatterer, g_shareds)
        elif exct_ID == DATA:
            manage_data(sub_ID, sync.data, scatterer)
        else:
            raise RuntimeError("Unrecognized exctution type in worker.")
        sync.exct.barrier_out.wait()  # Prevent premature shmem overwriting.
