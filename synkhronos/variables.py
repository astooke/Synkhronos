
"""
Classes for managing variables.
Inputs and Shareds used in both master and workers.
Outputs used only in master.
SynkFunction is base class for master and worker Function classes.
"""

# import gtimer as gt

import numpy as np
import theano

from .common import PID
from .shmemarray import NpShmemArray


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


###############################################################################
#                                                                             #
#                    Inputs, Shareds   (master & workers)                     #
#                                                                             #
###############################################################################


PRE = "/synk_" + PID

SHRD_IDS_TAG = PRE + "_shared_ids_"
ASGN_IDX_TAG = PRE + "_assign_idx_"
OTPT_SBST_TAG = PRE + "_output_subset_"
BCAST_BTS_TAG = PRE + "_scat_bts_"
BCAST_ID_TAG = PRE + "_scat_id_"
BCAST_IDXS_TAG = PRE + "_scat_idxs_"
BCAST_IDXS_SIZE_TAG = PRE + "_scat_idxs_size_"

INPT_TAG_PRE = PRE + "_inpt_"
SHRD_TAG_PRE = PRE + "_shrd_"
TAGS_TAG = "_tags_"
SHAPES_TAG = "_shapes_"


AVG_FAC_NAME = "__synk_avg_fac__"


class SynkVariables(struct):

    def __init__(self, create):
        super().__init__()
        self.create = create  # (True in master, False in workers)
        self.vars = list()
        self.names = list()
        self.dtypes = list()
        self.shmems = list()
        self.tags = list()
        self.ndims = list()
        self.num = 0
        self.sync = None

    def register_func(self, f):
        var_IDs = list()
        n_vars = len(f.inv_finder)
        for theano_In in [f.inv_finder[f.finder[i]] for i in range(n_vars)]:
            if theano_In.implicit == self.implicit:
                var_IDs.append(self.include(theano_In.variable))
        return tuple(var_IDs)

    def include(self, var):
        is_new_var = var not in self.vars
        if not is_new_var:
            var_ID = self.vars.index(var)
        else:
            var_ID = self.num
            self.vars.append(var)
            self.names.append(var.name)
            dtype = var.type.dtype
            self.dtypes.append(dtype)
            self.ndims.append(var.type.ndim)
            self.tags.append(0)
            self.shmems.append(None)
            self.num += 1
        return var_ID

    def alloc_shmem(self, var_ID, shape=None, rank=None):
        """ rank currently unused: maybe for chunking shared memory by worker"""
        if self.create:  # (only in master)
            self.sync.tags[var_ID] += 1
            self.sync.shapes[var_ID][:] = shape
        else:  # (only in worker)
            self.tags[var_ID] = self.sync.tags[var_ID]
            shape = self.sync.shapes[var_ID]
        tag = self.tag_pre + str(var_ID) + "_" + str(self.sync.tags[var_ID])
        if rank is not None:
            tag += "_" + str(rank)
        shmem = NpShmemArray(self.ctypes[var_ID], shape, tag, self.create)
        self.shmems[var_ID] = shmem
        return shmem

    def build_sync(self):
        if self.sync is not None:
            raise RuntimeError("Tried to build variables sync a second time.")
        if self.num > 0:
            shape_tag = self.tag_pre + SHAPES_TAG
            tags_tag = self.tag_pre + TAGS_TAG
            shapes = [NpShmemArray('i', ndim, shape_tag + str(idx), self.create)
                        for idx, ndim in enumerate(self.ndims)]
            self.sync = struct(
                tags=NpShmemArray('i', self.num, tags_tag, self.create),
                shapes=shapes,
            )

    def update_shmem(self, var_ID, data_arr, oversize=None):
        """ Master-only """
        data_arr = self.check_data_array(var_ID, data_arr)
        shmem = self.shmems[var_ID]
        memory_OK, data_OK = check_memory(shmem, data_arr)
        if not memory_OK:
            shape = oversize_shape(data_arr.shape, oversize)
            shmem = self.alloc_shmem(var_ID, shape)
        if not data_OK:
            shmem[:data_arr.shape[0]] = data_arr
        return shmem

    def check_data_array(self, var_ID, data_arr, idx=""):
        dtype = self.dtypes[var_ID]
        ndim = self.ndims[var_ID]
        if not isinstance(data_arr, np.ndarray):
            data_arr = np.asarray(data_arr)  # TODO: force type?
        if data_arr.dtype != dtype:
            common_dtype = np.find_common_type([data_arr.dtype, dtype], [])
            if common_dtype == dtype:
                data_arr = data_arr.astype(dtype)  # TODO: avoid recast?
            else:
                raise TypeError("Non up-castable data type provided for input "
                    "{}, received: {}, expected: {}".format(idx, data_arr.dtype,
                    dtype))
        if data_arr.ndim != ndim:
            raise TypeError("Wrong data ndim provided for input "
                "{}: {}".format(idx, data_arr.ndim))
        return data_arr

    def update_shmems(self, vars_data, variables=None, oversize=None):
        """
        vars_data is a dict with keys: var or name, values: data array
        variables is optional argument to use subset of vars_data
        """
        variables = vars_data.keys() if variables is None else variables
        var_IDs = self.get_IDs(variables)
        shmems = dict()
        for var_ID, var in zip(var_IDs, variables):
            try:
                shmem = self.update_shmem(var_ID, vars_data[var], oversize)
            except Exception as exc:
                msg = "Error when processing data under key: {}".format(var)
                raise Exception(msg) from exc
            shmems[var] = shmem
        return shmems

    # TODO: clean up all these helpers...just keep the ones ended up using.
    def get_shmems_vars(self, variables):
        var_IDs = self.get_IDs(variables)
        return self.get_shmems(var_IDs)

    def is_member(self, variable):
        if variable in self.names or variable in self.vars:
            return True
        else:
            return False

    def get_shmems(self, var_IDs):
        shmems = list()
        for var_ID in var_IDs:
            shmems.append(self.shmems[var_ID])
        return shmems

    def get_IDs(self, variables):
        if not isinstance(variables, (list, tuple)):
            variables = (variables,)
        var_IDs = list()
        for var in variables:
            if var is None:
                raise TypeError("Recieved NoneType for at least one variable.")
            elif var in self.vars:
                var_IDs.append(self.vars.index(var))
            elif var in self.names:
                var_IDs.append(self.names.index(var))
            else:
                raise ValueError("Unrecognized variable instance or name: ", var)
        if len(set(var_IDs)) != len(var_IDs):
            raise ValueError("Redundant variables provided.")
        return tuple(var_IDs)


def oversize_shape(shape, oversize):
    if oversize is not None:
        if oversize < 1 or oversize > 2:
            raise ValueError("Param 'oversize' must be in range 1 to 2"
                " (direct multiplicative factor on 0-th index size).")
        shape = list(shape)
        shape[0] = int(np.ceil(shape[0] * oversize))
    return shape


def check_memory(shmem, data_arr):
    memory_OK = False
    data_OK = False
    if shmem is not None:
        if data_arr.shape[1:] == shmem.shape[1:] and \
                data_arr.shape[0] <= shmem.shape[0]:
            memory_OK = True  # (existing memory big enough)
            input_addr, _ = data_arr.__array_interface__["data"]
            shmem_addr, _ = shmem.__array_interface__["data"]
            if input_addr == shmem_addr:
                if data_arr.__array_interface__["strides"] is not None:
                    print("Warning: Cannot keep strided view of memory as "
                        "input, will copy data into shmem array.")
                else:
                    data_OK = True  # (existing memory passed as input)
    return memory_OK, data_OK


###############################################################################
#                               Inputs                                        #


class Inputs(SynkVariables):

    implicit = False
    tag_pre = INPT_TAG_PRE


###############################################################################
#                               Shareds                                       #


class Shareds(SynkVariables):

    implicit = True
    tag_pre = SHRD_TAG_PRE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_funcs = list()
        self.avg_facs = list()

    def get_gpuarray(self, idx):
        """ Re-reference the variable in case GPU allocation has changed. """
        return self.vars[idx].container.data

    def include(self, var):
        var_ID = super().include(var)
        if self.create and var_ID >= len(self.avg_funcs):  # (only in master)
            avg_fac = theano.shared(np.array(1, dtype=self.dtypes[var_ID]),
                                    name=AVG_FAC_NAME)
            avg_func = theano.function([], updates={var: var * avg_fac})
            self.avg_facs.append(avg_fac)
            self.avg_funcs.append(avg_func)
        return var_ID

    # NOTE: in future, may bring back separate shmem for each rank, but not now
    # def build_shmem(self, shared_ID, n_gpu, master_rank):
    #     """ Only in master """
    #     shape = self.vars[shared_ID].container.data.shape
    #     shmems = list()
    #     for rank in range(n_gpu):
    #         if rank != master_rank:
    #             shmems.append(self.alloc_shmem(shared_ID, shape, rank))
    #         else:
    #             shmems.append(None)
    #     self.shmems[shared_ID] = shmems

    def build_sync(self):
        super().build_sync()
        if self.sync is not None:
            self.sync.shared_IDs = \
                NpShmemArray('i', self.num, SHRD_IDS_TAG, self.create)

    def set_avg_facs(self, n_gpu):
        for avg_fac, dtype in zip(self.avg_facs, self.dtypes):
            if "int" in dtype:
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
                    raise RuntimeError("Could not identify shared var's \
                        average factor.")


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
        self.dtypes = list()
        self.to_cpu = list()
        self.avg_funcs = list()
        self.avg_facs = list()
        self.num = 0

    def include(self, var):
        if var in self.vars:  # (already have this var, just retrieve it)
            output_ID = self.vars.index(var)
        else:
            from theano.gpuarray.type import GpuArrayVariable
            output_ID = self.num
            self.vars.append(var)
            to_cpu = False if isinstance(var, GpuArrayVariable) else True
            self.to_cpu.append(to_cpu)
            gpu_var = var.transfer(None)
            self.gpu_vars.append(gpu_var)
            self.dtypes.append(var.type.dtype)
            avg_fac = theano.shared(np.array(1, dtype=var.type.dtype))
            avg_otpt = (avg_fac * gpu_var).transfer(None)
            avg_func = theano.function([gpu_var], avg_otpt)
            self.avg_facs.append(avg_fac)
            self.avg_funcs.append(avg_func)
            self.num += 1
        return output_ID

    def register(self, outputs):
        if outputs is None:
            return [], []
        else:
            gpu_outputs = list()
            output_IDs = list()
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            for var in outputs:
                output_ID = self.include(var)
                output_IDs.append(output_ID)
                gpu_outputs.append(self.gpu_vars[output_ID])
            return gpu_outputs, output_IDs

    def set_avg_facs(self, n_gpu):
        for avg_fac, dtype in zip(self.avg_facs, self.dtypes):
            if "int" in dtype:
                avg_fac.set_value(1)
            else:
                avg_fac.set_value(1 / n_gpu)


###############################################################################
#                                                                             #
#                     Base Function (master & workers)                        #
#                                                                             #
###############################################################################


class SynkFunction(object):

    _n_gpu = None

    def __init__(self,
                 ID,
                 theano_function,
                 input_IDs,
                 collect_modes,
                 reduce_ops,
                 ):
        self._ID = ID
        self._theano_function = theano_function
        self._input_IDs = input_IDs
        self._collect_modes = collect_modes
        self._reduce_ops = reduce_ops

    def _build_sync(self):
        n_outputs = len(self._collect_modes)
        tag_ID = str(self._ID)
        if n_outputs == 0:
            output_subset = []
        else:
            output_subset = NpShmemArray(
                np.bool,
                len(self._collect_modes),  # (n_outputs)
                OTPT_SBST_TAG + tag_ID,
                self._create,
            )
            output_subset[:] = True
        self.sync = struct(
            output_subset=output_subset,
        )

    @property
    def theano_function(self):
        """ Read-only: returns the underlying Theano function. """
        return self._theano_function

    @property
    def inputs_scatter(self):
        """ Read-only: lists whether inputs are scattered (`0-th` dimension);
        otherwise broadcast. """
        return self._inputs_scatter

    @property
    def collect_modes(self):
        """ Read-only: lists the output collection modes. """
        return self._collect_modes

    @property
    def reduce_ops(self):
        """ Read-only: lists the output reduce operations. """
        return self._reduce_ops

    def _call_theano_function(self, inputs, output_subset=None):
        results = self._theano_function(*inputs, output_subset=output_subset)
        if not isinstance(results, list):
            results = [results]
        return results  # (always returns a list, even if length 1)
