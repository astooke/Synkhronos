
from pygpu.gpuarray import GpuArray
from numpy import ndarray

from .data import check_synk_inputs, Data
from .scatterer import scatterer
# TODO: from .comm import cpu_comm_master
from .comm_manager import comm
from . import exct


__all__ = ["broadcast", "gather", "reduce", "all_reduce", "all_gather", "scatter"]

sync = None  # (assigned later by master)


###############################################################################
#                                                                             #
#        Collective Communications over Theano Shared Variables               #
#                                                                             #
###############################################################################


# def coll_prep(shared_vars, sources=None, op=None, nccl=True):
#     exct.check_active()
#     if sources is not None:
#         if isinstance(shared_vars, (list, tuple)):
#             if not isinstance(sources, (list, tuple)) or len(sources) != len(shared_vars):
#                 raise TypeError("Arg 'shared_vars' and optional arg 'sources' "
#                     "must both be individal entries or lists/tuples of same "
#                     "length.")
#         elif isinstance(sources, (list, tuple)):
#             raise TypeError("Arg 'shared_vars' and optional arg 'sources' "
#                 "must both be individal entries or lists/tuples of same "
#                 "length.")
#     shared_IDs = g_shareds.get_IDs(shared_vars)
#     if sources is None:
#         sources = [shareds_registry.get_array(i) for i in shared_IDs]
#     elif not isinstance(sources, (list, tuple)):
#         sources = [sources]
#     src_types = [type(d) for d in sources]
#     src_type = src_types[0]
#     if src_types.count(src_type) != len(src_types):
#         raise TypeError("All broadcast data sources must be same type.")
#     if src_type not in (GpuArray, ndarray, Data):
#         raise TypeError("Broadcast data source must be type "
#             "pygpu.GpuArray, numpy.ndarray, or synkhronos.Data")
#     n_shared = len(shared_IDs)
#     sync.nccl.value = nccl
#     sync.src_type.value = SOURCE_TYPES[src_type]
#     sync.n_shared.value = n_shared
#     sync.vars[:n_shared] = shared_IDs
#     if src_type is Data:
#         sync.sources[:n_shared] = [synk_data.ID for synk_data in sources]
#     if op is not None:
#         sync.op.value = bytes(op, encoding='utf-8')
#     return shared_IDs, sources, src_type


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
        if val is not None and
                (var.ndim != val.ndim or var.dtype != val.dtype):
            raise TypeError("Had bad value at position {}."
                "expected ndim and dtype: {}, {}, "
                "received ndim and dtype: {}, {}".format(
                    i, var.ndim, var.dtype, val.ndim, val.dtype))
    return shared_vars, values


def broadcast_comm(shared_vars, gpu=True):
    shared_IDs = shareds_registry.get_IDs(shared_vars)
    arrays = shareds_registry.get_arrays(shared_IDs)
    n_shared = len(shared_IDs)
    sync.n_shared.value = n_shared
    sync.vars[:n_shared] = shared_IDs
    if gpu:
        exct_ID = exct.GPU_COMM
        comm = gpu_comm
    else:
        exct_ID = exct.CPU_COMM
        comm = cpu_comm
    exct.launch(exct_ID, exct.BROADCAST)
    for arr in arrays:
        comm.broadcast(arr)
    exct.join()


def broadcast_synk(shared_vars, synk_datas):
    shared_IDs = shareds_registry.get_IDs(shared_vars)
    data_IDs = [sd.ID for sd in synk_datas]
    n_shared = len(shared_IDs)
    sync.n_shared.value = n_shared
    sync.vars[:n_shared] = shared_IDs
    sync.datas[:n_shared] = data_IDs
    exct.launch(exct.SYNK_COMM, exct.BROADCAST)
    for var, sd in shared_vars, synk_datas:
        var.set_value(sd._data)
    exct.join()


def process_bcast_vars_vals(shared_vars, values, nccl):
    shared_vars, values = check_vars_vals(shared_vars, values)
    synk_idxs = [i for i, val in enumerate(values) if isinstance(val, Data)]
    synk_vars = [shared_vars[i] for i in synk_idxs]
    synk_datas = [values[i] for i in synk_idxs]
    if nccl:
        is_gpu_var = [type(v.contianer.data) is GpuArray for v in shared_vars]
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
    cpu_vars = [v, for i, v in enumerate(shared_vars) if i not in used_idxs]

    for var, val in zip(gpu_vars + cpu_vars, gpu_vals + cpu_vals):
        if val is not None:
            var.set_value(val)  # (broadcast will get from variable)

    return gpu_vars, cpu_vars, synk_var, synk_datas


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

##########################################################################
##########################################################################


def gather_comm(shared_vars, nd_up, gpu=True):
    shared_IDs = shareds_registry.get_IDs(shared_vars)
    arrays = shareds_registry.get_arrays(shared_IDs)
    n_shared = len(shared_IDs)
    sync.n_shared.value = n_shared
    sync.vars[:n_shared] = shared_IDs
    if gpu:
        exct_ID = exct.GPU_COMM
        comm = gpu_comm
    else:
        exct_ID = exct.CPU_COMM
        comm = cpu_comm
    exct.launch(exct_ID, exct.GATHER)
    results = list()
    for arr in arrays:
        results.append(comm.gather(arr, nd_up))
    return results


def process_gather_vars(shared_vars):
    raise NotImplementedError

    
def gather(shared_vars, nd_up=1):
    exct.check_active()
    gpu_vars, cpu_vars = process_gather_vars(shared_vars)
    results = list()
    if gpu_vars:
        results += gather_comm(gpu_vars, nd_up, gpu=True)
    if cpu_vars:
        results += gather_comm(cpu_vars, nd_up, gpu=False)


def gather_old(shared_vars, nd_up=1):
    shared_IDs = coll_prep(shareds_registry, shared_vars)
    exct.launch(exct.GPU_COMM, exct.GPU_GATHER)
    results = list()
    for shared_ID in shared_IDs:
        src = shareds_registry.get_array(shared_ID)
        r = gpu_comm.all_gather(src=src, dest=dest, nd_up=nd_up)
        results.append(r)
    exct.join()
    if dest is None:
        return results


def reduce(shared_vars, op="avg", in_place=True, dest=None):
    """GPU-comm: workers reduce values to master only.

    Can only use destination when reducing a single shared variable.  In-place
    will overwrite the values of all shared variables involed (in the master
    only), otherwise will return new GPU-arrays.

    Args:
        shared_vars (None, optional): names or vars to be reduced
        functions (None, optional): functions to have all shared vars reduced
        op (str, optional): e.g. "sum, prod, min, max, avg"
        in_place (bool, optional): overwrite result into shared var source
        dest (None, optional): GPU-array to write result (only if one var)

    Raises:
        ValueError: If infeasible inputs.

    Returns:
        List of GPUArrays, if no destination provided and not in-place.
    """
    op, avg, op_ID = get_op_and_avg(op)
    shared_IDs = coll_prep(shareds_registry, shared_vars, op_ID)
    if len(shared_IDs) > 1 and dest is not None:
        raise ValueError("When specifying desination, can only reduce one var.")
    if avg and (not in_place or dest is not None):
        raise ValueError("Can only use 'average' op with in-place reduce "
            "(requires None dest).")
    exct.launch(exct.GPU_COMM, exct.GPU_REDUCE)
    results = list()
    for shared_ID in shared_IDs:
        src = shareds_registry.get_array(shared_ID)
        dest = src if dest is None and in_place else dest
        results.append(gpu_comm.reduce(src=src, op=op, dest=dest))
    if avg:
        shareds_registry.call_avg_fs(shared_IDs)
    exct.join()
    if not in_place and dest is None:  # (otherwise results will be Nones)
        return results


def all_reduce(shared_vars, op="avg"):
    """GPU-comm: master and workers all reduce values, in-place only.

    Args:
        shared_vars (None, optional): names or vars to be reduced
        functions (None, optional): functions to have all shared vars reduced
        op (str, optional): e.g. "sum, prod, min, max, avg"
    """
    op, avg, op_ID = get_op_and_avg(op)
    shared_IDs = coll_prep(shareds_registry, shared_vars, op_ID)
    exct.launch(exct.GPU_COMM, exct.GPU_ALL_REDUCE)
    for shared_ID in shared_IDs:
        src = shareds_registry.get_array(shared_ID)
        gpu_comm.all_reduce(src=src, op=op, dest=src)
    if avg:
        shareds_registry.call_avg_fs(shared_IDs)
    exct.join()


def all_gather(source, dest):
    """GPU-comm: master and workers all gather values into their local vars.

    Only one Theano shared variable can be used for the source, and another
    Theano shared variable of the right shape must already exist for use as the
    destination (since no new shared variables can be created in workers).

    Args:
        source (name or var): shared variable to be gathered
        dest (name or var): shared variable to receive values in
    """
    shared_IDs = coll_prep(shareds_registry, [source, dest])
    exct.launch(exct.GPU_COMM, exct.GPU_ALL_GATHER)
    src = shareds_registry.get_array(shared_IDs[0])
    dest = shareds_registry.get_array(shared_IDs[1])
    gpu_comm.all_gather(src=src, dest=dest)
    exct.join()


###############################################################################
#                                                                             #
#                         CPU-based Communications                            #
#                                                                             #
###############################################################################


def scatter(vars_and_data, batch=None):
    """Scatter data and push to master and worker GPU Theano shared variables.

    Input `shared_vars_data` can be either a dictionary, a list, or a single
    variable/name.  If a dictionary, the input is used as in `set_shmems`; the
    data is used to update the shared memory before workers use the values.
    Otherwise, the input is used to determine which Theano shared variables to
    scatter over existing data in shared memory.

    Input parameter `batch` behaves as for function calls; it can limit the
    scatter effect over some subset of the allocated shared memory.


    Args:
        shared_vars_data (TYPE): Shared variables to scatter, optionally with data
        batch (None, optional): int, slice, or list of requested indices

    Raises:
        ValueError: If no input data and shared memory does not exist yet.
    """
    exct.check_active()
    variables = tuple(vars_and_data.keys())  # (establish an order)
    synk_datas = [vars_and_data[var] for var in variables]
    shared_vars = shareds_registry.get_vars(variables)
    shared_IDs = shareds_registry.get_IDs(shared_vars)
    check_synk_inputs(synk_datas, shared_vars)
    scatterer.assign_inputs(synk_datas, batch)
    n_shared = len(shared_IDs)
    sync.n_shared.value = n_shared
    sync.vars[:n_shared] = shared_IDs
    exct.launch(exct.CPU_COMM, exct.SCATTER)
    my_inputs, scatter = scatterer.get_my_inputs(n_shared)  # (as in workers)
    if not all(scatter):
        raise TypeError("Must use synk data set to scatter.")
    for var, my_input in zip(shared_vars, my_inputs):
        var.set_value(my_input)
    exct.join()


###############################################################################
#                                                                             #
#                   Theano Shared Variables Registry                          #
#                                                                             #
###############################################################################


class SharedsRegistry(object):

    def __init__(self):
        self.vars = list()
        self.names = list()
        self.avg_fs = list()
        self.inv_n_parallel = 1

    def register_func(self, f):
        for var in f.get_shared():
            self.register(var)

    def register(self, var):
        import theano
        import theano.tensor as T
        if var not in self.vars:
            self.vars.append(var)
            self.names.append(var.name)  # (could be None)
            if "int" in var.dtype:
                self.avg_fs.append(None)  # (labmda x: x, ?)
            else:
                y = T.scalar('avg_fac', dtype=var.dtype)
                avg_f = theano.function([y], updates=[(var, var * y)])
                self.avg_fs.append(avg_f)

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
            raise exc("Unrecognized shared var or name: ", var_or_name)

    def get_IDs(self, vars_or_names):
        if not isinstance(vars_or_names, (list, tuple, dict)):
            vars_or_names = (vars_or_names,)
        var_IDs = list()
        for var in vars_or_names:
            var_IDs.append(self.get_ID(var))
        if len(set(var_IDs)) != len(var_IDs):
            raise ValueError("Redundant variables provided.")
        return tuple(var_IDs)

    def get_var(self, var_or_name):
        if var_or_name is None:
            raise TypeError("Cannot find using NoneType.")
        if var_or_name in self.vars:
            return var_or_name
        else:
            try:
                return self.vars[self.names.index(var_or_name)]
            except ValueError as exc:
                raise exc("Unrecognized shared var or name: ", var_or_name)

    def get_vars(self, vars_or_names):
        varbles = list()
        for var in vars_or_names:
            varbles.append(self.get_var(var))
        if len(set(varbles)) != len(varbles):
            raise ValueError("Redundant variables provided.")
        return tuple(varbles)

    def get_vars_from_IDs(self, IDs):
        return [self.vars[i] for i in IDs]

    def get_array(self, idx):
        """ Re-reference the variable in case GPU allocation has changed. """
        return self.vars[idx].container.data

    def set_n_parallel(self, n_parallel):
        self.inv_n_parallel = 1 / n_parallel

    def call_avg_fs(self, var_IDs, avg_fac=None):
        avg_fac = self.inv_n_parallel if avg_fac is None else avg_fac
        for var_ID in var_IDs:
            self.avg_fs[var_ID](avg_fac)


shareds_registry = SharedsRegistry()
