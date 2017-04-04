
"""
Classes for managing variables.
Inputs and Shareds used in both master and workers.
Outputs used only in master.
SynkFunction is base class for master and worker Function classes.
"""
# import ipdb
import numpy as np

from .common import (PRE, GPU_REDUCE, CPU_REDUCE, GPU_GATHER, CPU_GATHER,
                     NO_COLLECT, REDUCE_AVG, REDUCE_OPS_WORKER)
from .shmemarray import NpShmemArray, ShmemRawArray, NP_TO_C_TYPE


###############################################################################
#                                                                             #
#                   Theano Shared Variables Registry                          #
#                                                                             #
###############################################################################


class Shareds(object):

    def __init__(self):
        self.vars = list()
        self.names = list()
        self.avg_fs = list()
        self.inv_n_parallel = 1

    def register_func(self, f, accumulators):
        for var in f.get_shared():
            self.register(var, accumulators)

    def register(self, var, accumulators):
        if var not in self.vars:
            self.vars.append(var)
            self.names.append(var.name)  # (could be None)
            if "int" in var.dtype:
                self.avg_fs.append(None)  # (labmda x: x, ?)
            else:
                avg_func = \
                    accumulators.get_function("avg_shared", var)
                s_var = avg_func.get_shared()[0]
                try:
                    self.avg_fs.append(avg_func.copy(swap={s_var: var}))
                except AssertionError as exc:
                    # FIXME: handle broadcastable pattern
                    print("WARNING: Unable to make averaging function for var: "
                        "{}\nAssertionError: {}\n(Hint: if given var is not "
                        "GpuArray Type, initialize shared var with numpy "
                        "array.)".format(var, exc))
                    self.avg_fs.append(None)

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
        varbs = list()
        for var in vars_or_names:
            varbs.append(self.get_var(var))
        if len(set(varbs)) != len(varbs):
            raise ValueError("Redundant variables provided.")
        return tuple(varbs)

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


###############################################################################
#                                                                             #
#               Base Data Container for Inputs & Shareds                      #
#                                                                             #
###############################################################################


class BaseData(object):

    _create = False

    def __init__(self, ID, dtype, ndim, scatter=True, minibatch=False, name=None):
        self._ID = ID
        self._ctype = NP_TO_C_TYPE.get(dtype, None)
        if self._ctype is None:
            raise TypeError("Unsupported numpy dtype: {}".format(dtype))
        self._data = np.empty([0] * ndim, dtype=dtype)
        self._tag = 0
        self._shmem = None
        self._np_shmem = None
        self._alloc_size = 0
        self._scatter = scatter  # Currently, fixed at instantiation.
        self._minibatch = minibatch
        self._name = name

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __len__(self):
        return len(self._data)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def data(self):
        return self._data

    @property
    def alloc_size(self):
        return self._alloc_size

    @property
    def name(self):
        return self._name

    @property
    def scatter(self):
        return self._scatter

    def _alloc_shmem(self, size, tag):
        tag = PRE + "_data_" + str(self._ID) + "_" + str(tag)
        self._shmem = ShmemRawArray(self._ctype, size, tag, self._create)
        self._np_shmem = np.ctypeslib.as_array(self._shmem)
        self._alloc_size = size

    def _shape_data(self, shape):
        self._data = self._np_shmem if len(shape) == 0 else \
            self._np_shmem[:int(np.prod(shape))].reshape(shape)

    def _free_shmem(self):
        self._data = np.empty([0] * self.ndim, dtype=self.dtype)
        self._np_shmem = None
        self._shmem = None
        self._alloc_size = 0


###############################################################################
#                                                                             #
#                          Base Scatterer                                     #
#                                                                             #
###############################################################################


class BaseScatterer(object):

    create = False

    def __init__(self, n_parallel, rank):
        self.n_parallel = n_parallel
        self.rank = rank
        self.synk_datas = list()

    def __len__(self):
        return len(self.synk_datas)

    def append(self, synk_data):
        self.synk_datas.append(synk_data)

    def get_my_inputs(self, n_inputs):
        if n_inputs == 0:
            return (), ()
        my_idxs = slice(*self.sync.assign_idxs[self.rank:self.rank + 2])
        minibatch_0 = min(self.sync.assign_idxs)  # minib is exactly right size
        minibatch_idxs = \
            slice(my_idxs.start + minibatch_0, my_idxs.stop + minibatch_0)
        if self.sync.use_idxs_arr.value:
            my_idxs = self.sync.idxs_arr[my_idxs]
        my_inputs = list()
        scatter = list()
        for data_ID in self.sync.data_IDs[:n_inputs]:
            synk_data = self.synk_datas[data_ID]
            if synk_data._scatter:
                if synk_data._minibatch:  # (assumes already shuffled if needbe)
                    my_inputs.append(synk_data._data[minibatch_idxs])
                else:
                    my_inputs.append(synk_data._data[my_idxs])
                scatter.append(True)
            else:
                my_inputs.append(synk_data._data)
                scatter.append(False)
        return tuple(my_inputs), tuple(scatter)

    def _alloc_idxs_arr(self, size, tag):
        tag = PRE + "_scat_idxs_" + str(tag)
        self.sync.idxs_arr = NpShmemArray('int64', size, tag, self.create)


###############################################################################
#                                                                             #
#                       Base Synk Function                                    #
#                                                                             #
###############################################################################


class BaseFunction(object):

    _create = False
    _n_parallel = None
    _rank = None

    def __init__(self, ID, theano_function):
        self._ID = ID
        self._f = theano_function
        self._n_input = len([i for i in self._f.maker.inputs if not i.implicit])
        self._n_output = len(self._f.outputs)
        self._output_set = []
        self._collects = []
        self._ops = []

    def get_shared(self):
        return self._f.get_shared()

    def _set_accum_fs(self, accumulators, do_accum=True):
        """ before barrier in master, after barrier in workers """
        self._avg_fs = list()
        self._accum_fs = list()
        for idx, mode_ID, op_ID in zip(self._output_set, self._collects, self._ops):
            var = self._f.outputs[idx].variable
            if do_accum:
                if mode_ID in [GPU_REDUCE, CPU_REDUCE]:
                    op = REDUCE_OPS_WORKER[op_ID]
                    self._accum_fs.append(accumulators.get_function("reduce", var, op))
                elif mode_ID in [GPU_GATHER, CPU_GATHER]:
                    self._accum_fs.append(accumulators.get_function("gather", var))
                elif mode_ID == NO_COLLECT:
                    self._accum_fs.append(lambda x, y: y)
                else:
                    raise RuntimeError("Unrecognized collect mode:", mode_ID)
            if mode_ID == GPU_REDUCE and op_ID == REDUCE_AVG:
                self._avg_fs.append(accumulators.get_function("avg_output", var))
            else:
                self._avg_fs.append(None)

    def _sliced_f(self, my_inputs, scatter, num_slices, output_subset):
        # assume num_slices > 1 and any(scatter) == True
        accum_rs = None
        for sliced_inputs in slice_inputs(my_inputs, scatter, num_slices):
            sliced_rs = self._f(*sliced_inputs, output_subset=output_subset)
            accum_rs = self._accum_my_results(accum_rs, sliced_rs)
        self._avg_my_results(accum_rs, num_slices)
        return accum_rs  # (always a list, even if length 1)

    def _accum_my_results(self, accum_rs, sliced_rs):
        if accum_rs is None:
            return sliced_rs
        if not isinstance(accum_rs, (list, tuple)):
            accum_rs = [accum_rs]
        if not isinstance(sliced_rs, (list, tuple)):
            sliced_rs = (sliced_rs,)
        for idx, (accum_r, sliced_r, accum_f) in \
                enumerate(zip(accum_rs, sliced_rs, self._accum_fs)):
            accum_rs[idx] = accum_f(accum_r, sliced_r)
        return accum_rs

    def _avg_my_results(self, accum_rs, num_slices):
        inv_num = 1 / num_slices
        for i, (r, avg_f) in enumerate(zip(accum_rs, self._avg_fs)):
            if avg_f is not None:
                accum_rs[i] = avg_f(r, inv_num)
        return accum_rs


def slice_inputs(inputs, scatter, num_slices):
    length = len(inputs[scatter.index(True)])  # (all scattered are same length)
    edges = np.linspace(0, length, num_slices + 1, dtype='int64')
    for slc in [slice(*edges[i:i + 2]) for i in range(num_slices)]:
        sliced_inputs = list()
        for inpt, scat in zip(inputs, scatter):
            sliced_inputs.append(inpt[slc] if scat else inpt)
        yield tuple(sliced_inputs)
