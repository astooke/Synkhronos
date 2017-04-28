
from pygpu.gpuarray import GpuArray

from .data_module import Data
from .scatterer import scatterer
from .gpu_utils import alloc_gpu_arr
from . import comm
from . import exct

__all__ = ["broadcast", "scatter", "gather", "all_gather", "reduce",
           "all_reduce", "set_value", "get_value", "get_lengths", "get_shapes"]

sync = None  # (assigned later by master)


###############################################################################
#                                                                             #
#        API for Collective Communications over Theano Shared Variables       #
#                                                                             #
###############################################################################


def broadcast(shared_vars, values=None, nccl=True):
    exct.check_active()
    gpu_vars, cpu_vars, synk_vars, synk_datas = \
        process_bcast_vars_vals(shared_vars, values, nccl)
    if gpu_vars:
        broadcast_comm(gpu_vars, gpu=True)
    if cpu_vars:
        broadcast_comm(cpu_vars, gpu=False)
    if synk_vars:
        broadcast_synk(synk_vars, synk_datas)


def scatter(shared_vars, values, batch=None):
    exct.check_active()
    synk_vars, cpu_vars, synk_datas, cpu_datas = \
        process_scat_vars_vals(shared_vars, values)
    if cpu_vars:
        scatter_cpu(cpu_vars, cpu_datas, batch)
    if synk_vars:
        scatter_synk(synk_vars, synk_datas, batch)


def gather(shared_vars, nd_up=1, nccl=True):
    exct.check_active()
    gpu_vars, cpu_vars, gpu_idxs, cpu_idxs = process_coll_vars(shared_vars, nccl)
    results = [None] * (len(gpu_vars) + len(cpu_vars))
    gpu_results = gather_comm(gpu_vars, nd_up, gpu=True) if gpu_vars else list()
    cpu_results = gather_comm(cpu_vars, nd_up, gpu=False) if cpu_vars else list()
    for i, r in zip(gpu_idxs, gpu_results):
        results[i] = r
    for i, r in zip(cpu_idxs, cpu_results):
        results[i] = r
    return results[0] if len(results) == 1 else tuple(results)


def all_gather(shared_vars, nccl=True):
    exct.check_active()
    gpu_vars, cpu_vars, _, _ = process_coll_vars(shared_vars, nccl)
    if gpu_vars:
        all_gather_comm(gpu_vars, gpu=True)
    if cpu_vars:
        all_gather_comm(cpu_vars, gpu=False)


def reduce(shared_vars, op="avg", in_place=True, nccl=True):
    exct.check_active()
    gpu_vars, cpu_vars, gpu_idxs, cpu_idxs = process_coll_vars(shared_vars, nccl)
    gpu_results = reduce_comm(gpu_vars, op, in_place, gpu=True)
    cpu_results = reduce_comm(cpu_vars, op, in_place, gpu=False)
    if not in_place:
        results = [None] * (len(gpu_idxs) + len(cpu_idxs))
        for i, r in zip(gpu_idxs, gpu_results):
            results[i] = r
        for i, r in zip(cpu_idxs, cpu_results):
            results[i] = r
        return results[0] if len(results) == 1 else tuple(results)


def all_reduce(shared_vars, op="avg", nccl=True):
    exct.check_active()
    gpu_vars, cpu_vars, _, _ = process_coll_vars(shared_vars, nccl)
    if gpu_vars:
        all_reduce_comm(gpu_vars, op, gpu=True)
    if cpu_vars:
        all_reduce_comm(cpu_vars, op, gpu=False)


def set_value(rank, shared_vars, values, batch=None):
    exct.check_active()
    synk_vars, cpu_vars, synk_datas, cpu_datas = \
        process_scat_vars_vals(shared_vars, values)
    if cpu_vars:
        set_value_cpu(rank, cpu_vars, cpu_datas, batch)
    if synk_vars:
        set_value_synk(rank, synk_vars, synk_datas, batch)


def get_value(rank, shared_vars):
    exct.check_active()
    collectives_prep(shared_vars, rank=rank)
    exct.launch(exct.CPU_COLL, exct.GET_VALUE)
    if not isinstance(shared_vars, (list, tuple)):
        results = comm.cpu.recv(rank)
    else:
        results = list()
        for _ in range(len(shared_vars)):
            results.append(comm.cpu.recv(rank))
        results = tuple(results)
    exct.join()
    return results


def get_lengths(shared_vars):
    exct.check_active()
    collectives_prep(shared_vars)
    exct.launch(exct.CPU_COLL, exct.GET_LENGTHS)
    if not isinstance(shared_vars, (list, tuple)):
        my_len = shared_vars.container.data.shape[0]
        results = comm.cpu.recv_lengths(master_len=my_len)
    else:
        results = list()
        for var in shared_vars:
            my_len = var.container.data.shape[0]
            results.append(comm.cpu.recv_lengths(master_len=my_len))
        results = tuple(results)
    exct.join()
    return results


def get_shapes(shared_vars):
    exct.check_active()
    collectives_prep(shared_vars)
    exct.launch(exct.CPU_COLL, exct.GET_SHAPES)
    if not isinstance(shared_vars, (list, tuple)):
        my_shape = shared_vars.container.data.shape
        results = comm.cpu.recv_shapes(master_shape=my_shape)
    else:
        results = list()
        for var in shared_vars:
            my_shape = var.container.data.shape
            results.append(comm.cpu.recv_shapes(master_shape=my_shape))
        results = tuple(results)
    exct.join()
    return results

###############################################################################
#                                                                             #
#                           Collectives Executives                            #
#                                                                             #
###############################################################################


def broadcast_comm(shared_vars, gpu=True):
    arrays = collectives_prep(shared_vars, gpu_bcast=gpu)
    exct_ID, comm = exct_comm_type(gpu)
    exct.launch(exct_ID, exct.BROADCAST)
    for arr in arrays:
        comm.broadcast(arr)
    exct.join()


def broadcast_synk(shared_vars, synk_datas):
    collectives_prep(shared_vars, synk_datas=synk_datas)
    exct.launch(exct.SYNK_COLL, exct.BROADCAST)
    for var, sd in shared_vars, synk_datas:
        var.set_value(sd._data)
    exct.join()


def scatter_cpu(shared_vars, values, batch=None):
    collectives_prep(shared_vars)
    exct.launch(exct.CPU_COLL, exct.SCATTER)
    results = list()
    for val in values:
        results.append(comm.cpu.scatter(val[batch]))
    for var, r in zip(shared_vars, results):
        var.set_value(r)
    exct.join()


def scatter_synk(shared_vars, synk_datas, batch=None):
    collectives_prep(shared_vars, synk_datas=synk_datas)
    scatterer.assign_inputs(synk_datas, batch)
    exct.launch(exct.SYNK_COLL, exct.SCATTER)
    results = scatterer.get_my_inputs(len(shared_vars))
    for var, r in zip(shared_vars, results):
        var.set_value(r)
    exct.join()


def gather_comm(shared_vars, nd_up, gpu=True):
    arrays = collectives_prep(shared_vars, nd_up=nd_up)
    exct_ID, comm = exct_comm_type(gpu)
    exct.launch(exct_ID, exct.GATHER)
    results = list()
    for arr in arrays:
        results.append(comm.gather(arr, nd_up))
    exct.join()
    return results


def all_gather_comm(shared_vars, gpu=True):
    arrays = collectives_prep(shared_vars)
    exct_ID, comm = exct_comm_type(gpu)
    exct.launch(exct_ID, exct.ALL_GATHER)
    results = list()
    for arr in arrays:
        results.append(comm.all_gather(arr))
    for var, r in zip(shared_vars, results):
        var.set_value(r)
    exct.join()


def reduce_comm(shared_vars, op, in_place, gpu=True):
    arrays = collectives_prep(shared_vars, op=op)
    dests = arrays if in_place else [None] * len(arrays)
    exct_ID, comm = exct_comm_type(gpu)
    exct.launch(exct_ID, exct.REDUCE)
    results = list()
    for arr, dest in zip(arrays, dests):
        results.append(comm.reduce(arr, op, dest))
    if in_place or not gpu:
        for var, r in zip(shared_vars, results):
            var.set_value(r)
    exct.join()
    if not in_place:
        return results


def all_reduce_comm(shared_vars, op, gpu=True):
    arrays = collectives_prep(shared_vars, op=op)
    exct_ID, comm = exct_comm_type(gpu)
    exct.launch(exct_ID, exct.ALL_REDUCE)
    results = list()
    for arr in arrays:
        results.append(comm.all_reduce(arr, op, arr))
    for var, r in zip(shared_vars, results):
        var.set_value(r)
    exct.join()


def set_value_cpu(rank, shared_vars, values, batch=None):
    collectives_prep(shared_vars, rank=rank)
    exct.launch(exct.CPU_COLL, exct.SET_VALUE)
    for val in values:
        if batch is None:
            comm.cpu.send(rank, val)
        else:
            comm.cpu.send(rank, val[batch])
    exct.join()


def set_value_synk(rank, shared_vars, synk_datas, batch=None):
    collectives_prep(shared_vars, rank=rank, synk_datas=synk_datas)
    scatterer.set_batch(batch)
    exct.launch(exct.SYNK_COLL, exct.SET_VALUE)
    exct.join()


###############################################################################
#                                                                             #
#                           Collectives Helpers                               #
#                                                                             #
###############################################################################


def process_bcast_vars_vals(shared_vars, values, nccl):
    shared_vars, values = check_vars_vals(shared_vars, values)
    synk_idxs = [i for i, val in enumerate(values) if isinstance(val, Data)]
    synk_vars = [shared_vars[i] for i in synk_idxs]
    synk_datas = [values[i] for i in synk_idxs]
    if nccl and comm.gpu is not None:
        is_gpu_var = [isinstance(v.container.data, GpuArray) for v in shared_vars]
        gpu_idxs = [i for i, g in enumerate(is_gpu_var) if g and i not in synk_idxs]
        gpu_vars = [shared_vars[i] for i in gpu_idxs]
        gpu_vals = [values[i] for i in gpu_idxs]
    else:
        gpu_idxs = list()
        gpu_vars = list()
        gpu_vals = list()
    cpu_idxs = [i for i in range(len(shared_vars)) if i not in synk_idxs + gpu_idxs]
    cpu_vars = [shared_vars[i] for i in cpu_idxs]
    cpu_vals = [values[i] for i in cpu_idxs]

    for var, val in zip(gpu_vars + cpu_vars, gpu_vals + cpu_vals):
        if val is not None:
            var.set_value(val)  # (broadcast will get from variable)

    return gpu_vars, cpu_vars, synk_vars, synk_datas


def process_scat_vars_vals(shared_vars, values):
    shared_vars, values = check_vars_vals(shared_vars, values)
    synk_idxs = [i for i, val in enumerate(values) if isinstance(val, Data)]
    synk_vars = [shared_vars[i] for i in synk_idxs]
    synk_datas = [values[i] for i in synk_idxs]
    cpu_idxs = [i for i in range(len(values)) if i not in synk_idxs]
    cpu_vars = [shared_vars[i] for i in cpu_idxs]
    cpu_datas = [values[i] for i in cpu_idxs]
    return synk_vars, cpu_vars, synk_datas, cpu_datas


def process_coll_vars(shared_vars, nccl):
    if not isinstance(shared_vars, (list, tuple)):
        shared_vars = [shared_vars]
    if nccl and comm.gpu is not None:
        is_gpu_var = [isinstance(v.container.data, GpuArray) for v in shared_vars]
        gpu_idxs = [i for i, g in enumerate(is_gpu_var) if g]
        gpu_vars = [shared_vars[i] for i in gpu_idxs]
    else:
        gpu_idxs = list()
        gpu_vars = list()
    cpu_idxs = [i for i in range(len(shared_vars)) if i not in gpu_idxs]
    cpu_vars = [shared_vars[i] for i in cpu_idxs]
    return gpu_vars, cpu_vars, gpu_idxs, cpu_idxs


def collectives_prep(shared_vars, synk_datas=None, op=None, gpu_bcast=False,
                     nd_up=None, rank=None):
    shared_IDs = shareds_registry.get_IDs(shared_vars)
    n_shared = len(shared_IDs)
    sync.n_shared.value = n_shared
    sync.vars[:n_shared] = shared_IDs
    if op is not None:
        sync.op.value = bytes(op, encoding='utf-8')
    if nd_up is not None:
        sync.nd_up.value = nd_up
    if gpu_bcast:  # (else worker array won't change shape)
        for i, v in enumerate(shared_vars):
            sync.shapes[i][:v.ndim] = v.container.data.shape
    if rank is not None:
        sync.rank.value = rank
    if synk_datas is not None:
        if not isinstance(synk_datas, (list, tuple)):
            synk_datas = [synk_datas]
        sync.datas[:n_shared] = [sd._ID for sd in synk_datas]
    arrays = shareds_registry.get_arrays(shared_IDs)
    return arrays


def check_vars_vals(shared_vars, values):
    if values is None:
        if not isinstance(shared_vars, (list, tuple)):
            shared_vars = [shared_vars]
        return shared_vars, [None] * len(shared_vars)
    if isinstance(shared_vars, (list, tuple)):
        if not isinstance(values, (list, tuple)) or len(values) != len(shared_vars):
            raise TypeError("Arg 'shared_vars' and optional arg 'sources' "
                "must both be individal entries or lists/tuples of same "
                "length.")
    else:
        shared_vars = [shared_vars]
        if isinstance(values, (list, tuple)):
            raise TypeError("Arg 'shared_vars' and optional arg 'sources' "
                "must both be individal entries or lists/tuples of same "
                "length.")
        values = [values]
    for i, (var, val) in enumerate(zip(shared_vars, values)):
        if val is not None and (var.ndim != val.ndim or var.dtype != val.dtype):
            raise TypeError("Had bad value at position {}."
                "expected ndim and dtype: {}, {}, "
                "received ndim and dtype: {}, {}".format(
                    i, var.ndim, var.dtype, val.ndim, val.dtype))
    return shared_vars, values


def exct_comm_type(gpu):
    if gpu:
        return (exct.GPU_COLL, comm.gpu)
    else:
        return (exct.CPU_COLL, comm.cpu)

###############################################################################
#                                                                             #
#                           Worker Tasks                                      #
#                                                                             #
###############################################################################


def worker_gpu_coll(comm_ID):
    shared_IDs = sync.vars[:sync.n_shared.value]
    arrays = shareds_registry.get_arrays(shared_IDs)
    if comm_ID == exct.BROADCAST:
        shared_vars = shareds_registry.get_vars(shared_IDs)
        for i, (arr, var) in enumerate(zip(arrays, shared_vars)):
            shape = tuple(sync.shapes[i][:var.ndim])
            if shape != arr.shape:
                arr = alloc_gpu_arr(arr.dtype, shape)
                comm.gpu.broadcast(arr)
                var.set_value(arr)
            else:
                comm.gpu.broadcast(arr)
    elif comm_ID == exct.GATHER:
        nd_up = sync.nd_up.value
        for arr in arrays:
            comm.gpu.gather(arr, nd_up)
    elif comm_ID == exct.ALL_GATHER:
        results = list()
        for arr in arrays:
            results.append(comm.gpu.all_gather(arr))
        for var, r in zip(shareds_registry.get_vars(shared_IDs), results):
            var.set_value(r)
    elif comm_ID == exct.REDUCE:
        op = sync.op.value.decode(encoding='utf-8')
        for arr in arrays:
            comm.gpu.reduce(arr, op)
    elif comm_ID == exct.ALL_REDUCE:
        op = sync.op.value.decode(encoding='utf-8')
        results = list()
        for arr in arrays:
            results.append(comm.gpu.all_reduce(arr, op))
        for var, r in zip(shareds_registry.get_vars(shared_IDs), results):
            var.set_value(r)
    else:
        raise RuntimeError("Invalid worker GPU Comm ID: {}".format(comm_ID))


def worker_cpu_coll(comm_ID, rank):
    shared_IDs = sync.vars[:sync.n_shared.value]
    shared_vars = shareds_registry.get_vars(shared_IDs)
    arrays = shareds_registry.get_arrays(shared_IDs)
    if comm_ID == exct.BROADCAST:
        for var in shared_vars:
            var.set_value(comm.cpu.recv_pub())
    elif comm_ID == exct.SCATTER:
        for var in shared_vars:
            var.set_value(comm.cpu.recv_pair())
    elif comm_ID in [exct.GATHER, exct.REDUCE]:
        for arr in arrays:
            comm.cpu.send(arr)
    elif comm_ID in [exct.ALL_GATHER, exct.ALL_REDUCE]:
        results = list()
        for arr in arrays:
            results.append(comm.cpu.send_recv(arr))
        for var, r in zip(shared_vars, results):
            var.set_value(r)
    elif comm_ID == exct.SET_VALUE:
        if rank == sync.rank.value:
            for var in shared_vars:
                var.set_value(comm.cpu.recv_pair())
    elif comm_ID == exct.GET_VALUE:
        if rank == sync.rank.value:
            for var in shared_vars:
                comm.cpu.send(var.get_value())
    elif comm_ID == exct.GET_LENGTHS:
        for arr in arrays:
            comm.cpu.send_length(arr)
    elif comm_ID == exct.GET_SHAPES:
        for arr in arrays:
            comm.cpu.send_shape(arr)
    else:
        raise RuntimeError("Invalid worker CPU Comm ID: {}".format(comm_ID))


def worker_synk_coll(comm_ID):
    shared_IDs = sync.vars[:sync.n_shared.value]
    shared_vars = shareds_registry.get_vars(shared_IDs)
    if comm_ID == exct.BROADCAST:
        data_IDs = sync.datas[:sync.n_shared.value]
        synk_datas = [scatterer[i] for i in data_IDs]
        for var, sd in zip(shared_vars, synk_datas):
            var.set_value(sd._data)
    elif comm_ID == exct.SCATTER:
        my_inputs = scatterer.get_my_inputs(len(shared_IDs))
        for var, my_data in zip(shared_vars, my_inputs):
            var.set_value(my_data)
    else:
        raise RuntimeError("Invalid worker Synk Comm ID: {}".format(comm_ID))


###############################################################################
#                                                                             #
#                   Theano Shared Variables Registry                          #
#                                                                             #
###############################################################################


class SharedsRegistry(object):

    def __init__(self):
        self.vars = list()
        self.names = list()

    def reset(self):
        self.__init__()

    def register_func(self, f):
        for var in f.get_shared():
            self.register(var)

    def register(self, var):
        if var not in self.vars:
            self.vars.append(var)
            self.names.append(var.name)  # (could be None)

    def get_ID(self, var_or_name):
        if var_or_name is None:
            raise TypeError("Cannot find using NoneType.")
        try:
            return self.vars.index(var_or_name)
        except ValueError:
            pass
        try:
            return self.names.index(var_or_name)
        except ValueError as exc:
            raise exc("Unrecognized shared variable or name: ", var_or_name)

    def get_IDs(self, vars_or_names):
        if not isinstance(vars_or_names, (list, tuple)):
            vars_or_names = (vars_or_names,)
        var_IDs = list()
        for var in vars_or_names:
            var_IDs.append(self.get_ID(var))
        if len(set(var_IDs)) != len(var_IDs):
            raise ValueError("Redundant variables provided.")
        return tuple(var_IDs)

    def get_arrays(self, idxs):
        return [self.vars[i].container.data for i in idxs]

    def get_vars(self, idxs):
        return [self.vars[i] for i in idxs]


shareds_registry = SharedsRegistry()
