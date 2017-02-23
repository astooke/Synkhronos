
"""
Classes for managing variables.
Inputs and Shareds used in both master and workers.
Outputs used only in master.
SynkFunction is base class for master and worker Function classes.
"""

import numpy as np
import theano
from ctypes import c_bool

from .common import PID, get_my_scat_idxs
from .shmemarray import NpShmemArray, ShmemRawArray


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


###############################################################################
#                                                                             #
#              Inputs and  Shareds Registries   (master & workers)            #
#                                                                             #
###############################################################################


PRE = "/synk_" + PID

# Functions
OTPT_SBST_TAG = PRE + "_output_subset_"
CLCT_MD_TAG = PRE + "_collect_modes_"
RDC_OP_TAG = PRE + "_reduce_ops_"
IN_DATA_ID_TAG = PRE + "_in_data_ids_"

# Datas
DATA_SHAPE_TAG = PRE + "_data_shape_"
DATA_TAG = PRE + "_data_"

AVG_FAC_NAME = "__synk_avg_fac__"


###############################################################################
#                          Inputs Registry                                    #

class Inputs(struct):

    implicit = False

    def __init__(self):
        super().__init__()
        self.vars = list()
        self.names = list()

    def register_func(self, f):
        var_IDs = list()
        n_vars = len(f.inv_finder)
        for theano_In in [f.inv_finder[f.finder[i]] for i in range(n_vars)]:
            if self.implicit == theano_In.implicit:
                var_IDs.append(self.register(theano_In.variable))
        return tuple(var_IDs)

    def register(self, variable):
        if variable in self.vars:
            var_ID = self.vars.index(variable)
        else:
            var_ID = len(self.vars)
            self.vars.append(variable)
            self.names.append(variable.name)
        return var_ID

    def is_member(self, var_or_name):
        member = False
        if var_or_name in self.vars:
            member = True
        elif var_or_name is not None and var_or_name in self.names:
            member = True
        return member

    def get_ID(self, var_or_name):
        if var_or_name in self.vars:
            return self.vars.index(var_or_name)
        elif var_or_name is not None and var_or_name in self.names:
            return self.names.index(var_or_name)
        else:
            raise ValueError("Unrecognized variable or name: ", var_or_name)

    def get_IDs(self, vars_or_names):
        if not isinstance(vars_or_names, (list, tuple)):
            vars_or_names = (vars_or_names,)
        var_IDs = list()
        for var in vars_or_names:
            var_IDs.append(self.get_ID(var))
        if len(set(var_IDs)) != len(var_IDs):
            raise ValueError("Redundant variables provided.")
        return tuple(var_IDs)


###############################################################################
#                           Shared Variables Registry                         #

class Shareds(Inputs):

    implicit = True

    def __init__(self, make_avg):
        super().__init__()
        self.make_avg = make_avg
        self.avg_facs = list()
        self.avg_funcs = list()

    def get_gpuarray(self, idx):
        """ Re-reference the variable in case GPU allocation has changed. """
        return self.vars[idx].container.data

    def register(self, var):
        old_n = len(self.vars)
        var_ID = super().register(var)
        if len(self.vars) > old_n:
            if self.make_avg:  # (shareds in master only)
                dtype = self.vars[var_ID].type.dtype
                avg_fac = theano.shared(np.array(1, dtype=dtype), name=AVG_FAC_NAME)
                avg_func = theano.function([], updates={var: var * avg_fac})
                self.avg_facs.append(avg_fac)
                self.avg_funcs.append(avg_func)

    def set_avg_facs(self, n_gpu):
        for avg_fac, var in zip(self.avg_facs, self.vars):
            if "int" in var.type.dtype:
                avg_fac.set_value(1)  # int types do not support averaging.
            else:
                avg_fac.set_value(1 / n_gpu)

    def unpack_avg_facs(self):
        """ Worker only (and only if later changing avg_fac dynamically) """
        for fcn in self.avg_functions:
            for fcn_shared in fcn.get_shared():
                if fcn_shared.name == AVG_FAC_NAME:
                    self.avg_facs.append(fcn_shared)
                    break
                else:
                    raise RuntimeError("Could not identify shared var's "
                        "averaging factor.")


###############################################################################
#                                                                             #
#           Data Container for Inputs & Shareds (master & workers)            #
#                                                                             #
###############################################################################


class BaseData(object):

    _create = False

    def __init__(self, ID, dtype, ndim, scatter=True):
        super().__init__()
        self._ID = ID
        self._dtype = dtype
        self._ndim = ndim
        self._data = None
        self.data = None
        self._length = None
        self._tag = 0
        self._shmem = None
        self._alloc_size = 0
        self._scatter = scatter

    def _alloc_shmem(self, shape, tag):
        tag = DATA_TAG + str(self._ID) + "_" + str(tag)
        self._data, self._shmem = \
            NpShmemArray(self._dtype, shape, tag, self._create, True)
        self._alloc_size = self._data.size
        self.data = self._data

    def _reshape_shmem(self, shape):
        np_arr = np.ctypeslib.as_array(self._shmem)
        self._data = np_arr[:int(np.prod(shape))].reshape(shape)
        self._length = len(self._data)
        self.data = self._data

    def _free_shmem(self):
        self._data = None
        self.data = None
        self._shmem = None
        self._length = 0
        self._alloc_size = 0


###############################################################################
#                                                                             #
#                            Outputs (master only)                            #
#                                                                             #
###############################################################################


class Outputs(struct):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vars = list()
        self.gpu_vars = list()
        self.to_cpu = list()
        self.avg_funcs = list()
        self.avg_facs = list()

    def register(self, var):
        if var in self.vars:  # (already have this var, just retrieve it)
            output_ID = self.vars.index(var)
        else:
            from theano.gpuarray.type import GpuArrayVariable
            output_ID = len(self.vars)
            self.vars.append(var)
            to_cpu = False if isinstance(var, GpuArrayVariable) else True
            self.to_cpu.append(to_cpu)
            gpu_var = var.transfer(None)
            self.gpu_vars.append(gpu_var)
            avg_fac = theano.shared(np.array(1, dtype=var.type.dtype))
            avg_otpt = (avg_fac * gpu_var).transfer(None)
            avg_func = theano.function([gpu_var], avg_otpt)
            self.avg_facs.append(avg_fac)
            self.avg_funcs.append(avg_func)
        return output_ID

    def register_set(self, outputs):
        if outputs is None:
            return [], []
        else:
            gpu_outputs = list()
            output_IDs = list()
            if not isinstance(outputs, (list, tuple)):
                outputs = (outputs,)
            for var in outputs:
                output_ID = self.register(var)
                output_IDs.append(output_ID)
                gpu_outputs.append(self.gpu_vars[output_ID])
            return gpu_outputs, output_IDs

    def set_avg_facs(self, n_gpu):
        for avg_fac, var in zip(self.avg_facs, self.vars):
            if "int" in var.type.dtype:
                avg_fac.set_value(1)
            else:
                avg_fac.set_value(1 / n_gpu)


###############################################################################
#                                                                             #
#                     Base Function (master & workers)                        #
#                                                                             #
###############################################################################


class BaseFunction(object):

    _create = False
    _n_gpu = None
    _rank = None

    def __init__(self, ID, theano_function):
        self._ID = ID
        self._f = theano_function
        self._n_input = len(self._f.inv_finder) - len(self._f.get_shared())
        self._n_output = len(self._f.outputs)
        self._build_sync()

    def _build_sync(self):
        tag_ID = str(self._ID)
        if self._n_output == 0:
            output_subset = []
            collect_modes = []
            reduce_ops = []
        else:
            output_subset = ShmemRawArray(
                c_bool, [True] * self._n_output, OTPT_SBST_TAG + tag_ID, self._create)
            collect_modes = NpShmemArray(
                'uint8', self._n_output, CLCT_MD_TAG + tag_ID, self._create)
            reduce_ops = NpShmemArray(
                'uint8', self._n_output, RDC_OP_TAG + tag_ID, self._create)
        if self._n_input == 0:
            data_IDs = []
        else:
            data_IDs = NpShmemArray(
                'uint32', self._n_input, IN_DATA_ID_TAG + tag_ID, self._create)
        self._sync = struct(
            output_subset=output_subset,
            collect_modes=collect_modes,
            reduce_ops=reduce_ops,
            data_IDs=data_IDs,
        )

    def _get_my_inputs(self, sync_scat, g_synk_datas):
        if self._n_input == 0:
            return ()
        my_idxs = get_my_scat_idxs(sync_scat, self._rank)
        my_inputs = list()
        for data_ID in self._sync.data_IDs:
            synk_data = g_synk_datas[data_ID]
            if synk_data._scatter:
                my_inputs.append(synk_data.data[my_idxs])
            else:
                my_inputs.append(synk_data.data)
        return tuple(my_inputs)
