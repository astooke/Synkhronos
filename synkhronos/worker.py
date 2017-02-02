
"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything unique to the workers.
"""

import os
import pickle
from threading import BrokenBarrierError

from .variables import Inputs, Shareds, SynkFunction
from .common import use_gpu
from .common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                  ALL_GATHER, GATHER, WORKER_OPS, AVG_ALIASES, CPU_COMM, SCATTER)


class Function(SynkFunction):

    rank = None
    master_rank = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_output_subset_shmem(False)

    def __call__(self, sync, g_inputs, gpu_comm):
        """
        1. Gather the right inputs from mp shared values.
        2. Execute local theano function on those inputs.
        3. Send results back to master.
        """
        my_inputs = self.receive_inputs(g_inputs)
        output_subset, output_set = self._receive_output_subset()
        my_results = self._call_theano_function(my_inputs, output_subset)  # (always returns a tuple)
        self._collect_results(my_results, gpu_comm, output_set)

    def receive_inputs(self, g_inputs):
        assign_idx = g_inputs.sync.assign_idx[self._ID]
        my_idx = (assign_idx[self.rank], assign_idx[self.rank + 1])
        my_inputs = list()
        for input_ID, scatter in zip(self._input_IDs, self._inputs_scatter):
            if g_inputs.sync.tags[input_ID] != g_inputs.tags[input_ID]:
                # then a new shmem has been allocated, need to get it.
                shape = g_inputs.sync.shapes[input_ID][:]
                tag_ID = g_inputs.sync.tags[input_ID]
                g_inputs.alloc_shmem(input_ID, shape, tag_ID, False)
            shmem = g_inputs.shmems[input_ID]
            if scatter:
                my_inputs.append(shmem[my_idx[0]:my_idx[1]])
            else:
                my_inputs.append(shmem[:g_inputs.sync.max_idx[input_ID]])
        return tuple(my_inputs)

    def _receive_output_subset(self):
        output_set = [i for i, x in enumerate(self._output_subset_shmem) if x]
        output_subset = None if all(output_set) else output_set
        return output_subset, output_set

    def _collect_results(self, my_results, gpu_comm, output_set):
        for idx, r in zip(output_set, my_results):
            mode = self._collect_modes[idx]
            op = self._reduce_ops[idx]
            if mode == "reduce":
                gpu_comm.reduce(r, op=op, root=self.master_rank)
            elif mode == "gather":
                gpu_comm.all_gather(r)
            elif mode is not None:
                raise RuntimeError("Unrecognized collect mode in worker function.")


def unpack_functions(theano_functions, sync_dict, n_fcn):
    """
    Worker will recover variables in the same order as the master committed
    them, so they will have the same ID (index).
    """
    collect_modes_all = sync_dict["collect_modes"]
    reduce_ops_all = sync_dict["reduce_ops"]
    inputs_scatter_all = sync_dict["inputs_scatter"]
    synk_functions = list()
    g_inputs = Inputs()
    g_shareds = Shareds()
    for idx, fcn in enumerate(theano_functions[:n_fcn]):
        input_IDs = g_inputs.register_func(fcn)
        g_shareds.register_func(fcn, build_avg_func=False)
        synk_functions.append(Function(ID=idx,
                                       theano_function=fcn,
                                       input_IDs=input_IDs,
                                       inputs_scatter=inputs_scatter_all[idx],
                                       collect_modes=collect_modes_all[idx],
                                       reduce_ops=reduce_ops_all[idx],
                                       )
                              )
    g_shareds.avg_functions = theano_functions[n_fcn:]
    # g_shareds.unpack_avg_facs()  # (only needed for changing avg_fac later)
    return synk_functions, g_inputs, g_shareds


def receive_distribution(rank, n_gpu, sync):
    sync.barriers.distribute.wait()
    if not sync.distributed.value:
        return False
    sync_dict = sync.dict.copy()  # (retrieve it as a normal dict)
    with open(PKL_FILE, "rb") as f:
        theano_functions = pickle.load(f)  # should be all in one list
    if sync.barriers.delete_pkl.wait() == 0:
        os.remove(PKL_FILE)  # leave no trace
    synk_functions, g_inputs, g_shareds = \
        unpack_functions(theano_functions, sync_dict, sync.n_user_fcns.value)
    g_inputs.build_sync(len(synk_functions), n_gpu, False)
    g_shareds.build_sync(False)
    return synk_functions, g_inputs, g_shareds


def do_gpu_comms(sync, g_shareds, gpu_comm, master_rank):
    shared_IDs = g_shareds.sync.shared_IDs[:sync.n_shared.value]
    comm_ID = sync.comm_ID.value
    if comm_ID in [REDUCE, ALL_REDUCE]:
        op = WORKER_OPS.get(sync.comm_op.value, None)
        assert op is not None
        avg = op in AVG_ALIASES
        op = "sum" if avg else op
    if comm_ID == ALL_GATHER:
        src = g_shareds.gpuarrays[shared_IDs[0]]
        dest = g_shareds.gpuarrays[shared_IDs[1]]
        gpu_comm.all_gather(src, dest)
    else:
        for shared_ID in shared_IDs:
            src = g_shareds.gpuarrays[shared_ID]
            if comm_ID == BROADCAST:
                gpu_comm.broadcast(src, root=master_rank)
            elif comm_ID == REDUCE:
                gpu_comm.reduce(src, op=op, root=master_rank)
            elif comm_ID == ALL_REDUCE:
                gpu_comm.all_reduce(src, op=op, dest=src)
            elif comm_ID == GATHER:
                gpu_comm.all_gather(src)
            else:
                raise RuntimeError("Unrecognized GPU communication \
                    type in worker.")
        if comm_ID == ALL_REDUCE and avg:
            for shared_ID in shared_IDs:
                g_shareds.avg_functions[shared_ID]()


def do_cpu_comms(sync, g_shareds, rank):
    shared_ID = sync.shared_IDs[0]
    comm_ID = sync.comm_ID.value
    if comm_ID == SCATTER:
        if g_shareds.shmems[shared_ID] is None:
            g_shareds.alloc_shmem(shared_ID, rank, False)
        g_shareds.vars[shared_ID].set_value(g_shareds.shmems[shared_ID])
    else:
        raise RuntimeError("Unrecognized CPU comm type in worker.")


def error_close(sync):
    sync.workers_OK.value = False
    try:
        sync.barriers.exec_out.wait(1)
    except BrokenBarrierError:
        pass


def worker_exec(rank, n_gpu, master_rank, sync):
    gpu_comm = use_gpu(rank, n_gpu, sync, False)
    if not gpu_comm:
        return  # (exit quietly)

    distribution = receive_distribution(rank, n_gpu, sync)
    if not distribution:
        return  # (exit quietly)
    else:
        synk_functions, g_inputs, g_shareds = distribution

    Function.rank = rank  # endow all functions
    Function.master_rank = master_rank

    import atexit
    atexit.register(error_close, sync)

    while True:
        sync.barriers.exec_in.wait()
        if sync.quit.value:
            atexit.unregister(error_close)
            return  # (exit successfully)
        if sync.exec_type.value == FUNCTION:
            synk_functions[sync.func_ID.value](sync, g_inputs, gpu_comm)
        elif sync.exec_type.value == GPU_COMM:
            do_gpu_comms(sync, g_shareds, gpu_comm, master_rank)
        elif sync.exec_type.value == CPU_COMM:
            do_cpu_comms(sync, g_shareds, rank)
        else:
            raise RuntimeError("Unrecognized execution type in worker.")
        sync.barriers.exec_out.wait()  # Prevent premature shmem overwriting.
