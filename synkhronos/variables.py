
"""
Classes for managing variables.
Inputs and Shareds used in both master and workers.
Outputs used only in master.
"""

import ctypes
import numpy as np
import theano

from .common import PID
from .shmemarray import ShmemRawArray


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


###############################################################################
#                                                                             #
#                    Inputs, Shareds   (master & workers)                     #
#                                                                             #
###############################################################################


NP_TO_C_TYPE = {'float64': ctypes.c_double,
                'float32': ctypes.c_float,
                'float16': None,
                'int8': ctypes.c_byte,
                'int16': ctypes.c_short,
                'int32': ctypes.c_int,
                'int64': ctypes.c_longlong,
                'uint8': ctypes.c_ubyte,
                'uint16': ctypes.c_ushort,
                'uint32': ctypes.c_uint,
                'uint64': ctypes.c_ulonglong,
                'bool': ctypes.c_bool,
                }

PRE = "/synk_" + PID
SHRD_ARRAY_TAG = PRE + "_active_theano_shareds"  # (shouldn't be a conflict!)
INPUT_TAGS_TAG = PRE + "_input_tag_IDs"
ASGN_IDX_TAG = PRE + "_assign_idx_"
SHAPES_TAG = PRE + "_shapes_"
MAX_INPT_IDX_TAG = PRE + "_max_idx"
INPT_SHMEM_TAG_PRE = PRE + "_INPT_"
SHRD_SHMEM_TAG_PRE = PRE + "_SHRD_"
OTPT_SBST_TAG_PRE = PRE + "_output_subset_"

AVG_FAC_NAME = "__synk_avg_fac__"


class SynkVariables(struct):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vars = list()
        self.names = list()
        self.dtypes = list()
        self.ctypes = list()
        self.shmems = list()
        self.num = 0
        self.sync = None
        self.shmem_tag_pre = None

    def _include(self, var):
        is_new_var = var not in self.vars
        if not is_new_var:
            var_ID = self.vars.index(var)
        else:
            var_ID = self.num
            self.vars.append(var)
            self.names.append(var.name)
            dtype = var.type.dtype
            self.dtypes.append(dtype)
            ctype = NP_TO_C_TYPE.get(dtype, None)
            if ctype is None:
                raise TypeError("Numpy/Theano type: ", dtype, " not supported.")
            self.ctypes.append(ctype)
            self.shmems.append(None)
            self.num += 1
        return is_new_var, var_ID

    def _alloc_shmem(self, var_ID, shape, tag_ID, create):
        shmem = np.ctypeslib.as_array(
            ShmemRawArray(
                self.ctypes[var_ID],
                int(np.prod(shape)),
                self.shmem_tag_pre + str(var_ID) + "_" + str(tag_ID),
                create,
            )
        ).reshape(shape)
        return shmem


###############################################################################
#                               Inputs                                        #


class Inputs(SynkVariables):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tags = list()
        self.ndims = list()
        self.shmem_tag_pre = INPT_SHMEM_TAG_PRE

    def include(self, var):
        is_new_var, var_ID = self._include(var)
        if is_new_var:
            self.tags.append(var_ID)
            self.ndims.append(var.type.ndim)
        return var_ID

    def register_func(self, theano_function):
        input_IDs = list()
        for theano_In in theano_function.inv_finder.values():
            if not theano_In.implicit:  # (then it is explicit input)
                var = theano_In.variable
                input_IDs.append(self.include(var))
        return tuple(input_IDs)

    def alloc_shmem(self, input_ID, shape, tag_ID=None, create=True):
        _tag_ID = self.tags[input_ID] + 1 if tag_ID is None else tag_ID
        shmem = self._alloc_shmem(input_ID, shape, _tag_ID, create)
        self.shmems[input_ID] = shmem
        self.tags[input_ID] = _tag_ID
        if tag_ID is None:
            return shmem, _tag_ID  # (in master, new tag made)
        else:
            return shmem  # (in worker)

    def build_sync(self, n_func, n_gpu, create=True):
        if self.sync is not None:
            raise RuntimeError("Tried to build inputs sync a second time.")
        if self.num == 0:
            sync = None
        else:
            assign_idx = [ShmemRawArray('i', n_gpu + 1, ASGN_IDX_TAG + str(idx),
                                        create)
                            for idx in range(n_func)]
            shapes = [ShmemRawArray('i', ndim, SHAPES_TAG + str(idx), create)
                        for idx, ndim in enumerate(self.ndims)]
            max_idx = ShmemRawArray('i', self.num, MAX_INPT_IDX_TAG, create)
            sync = struct(
                tags=ShmemRawArray('i', self.num, INPUT_TAGS_TAG, create),
                assign_idx=assign_idx,
                shapes=shapes,
                max_idx=max_idx,
            )
        self.sync = sync
        return sync

    def update_shmem(self, input_ID, input_data):
        """ Master-only """
        shmem = self.shmems[input_ID]
        if not check_memory(shmem, input_data):
            shape = list(input_data.shape)
            shape[0] = int(np.ceil(shape[0] * 1.05))   # (a little extra)
            shmem, tag_ID = self.alloc_shmem(input_ID, shape)
            self.sync.tags[input_ID] = tag_ID
            self.sync.shapes[input_ID][:] = shape
        shmem[:input_data.shape[0]] = input_data
        self.sync.max_idx[input_ID] = input_data.shape[0]  # (in case broadcast)
        return shmem

    def check_inputs(self, input_IDs, ordered_inputs):
        """ Master-only """
        for idx, (input_ID, input_data) in enumerate(zip(input_IDs, ordered_inputs)):
            if not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data, dtype=self.dtypes[input_ID])
                ordered_inputs[idx] = input_data
            elif input_data.dtype != self.dtypes[input_ID]:
                raise TypeError("Wrong data type provided for input ", idx,
                    ": ", input_data.dtype)
            if input_data.ndim != self.ndims[input_ID]:
                raise TypeError("Wrong data ndim provided for input ", idx,
                    ": ", input_data.ndim)
        return ordered_inputs  # (now as numpy arrays)


def check_memory(shmem, input_data):
    memory_OK = False
    if shmem is not None:
        input_addr, _ = input_data.__array_interface__["data"]
        shmem_addr, _ = shmem.__array_interface__["data"]
        if input_addr == shmem_addr:
            if input_data.__array_interface__["strides"] is not None:
                print("Warning: Cannot use strided view of memory as input, "
                    "will copy into new shmem array.")
            elif input_data.shape[1:] == shmem.shape[1:] and \
                    input_data.shape[0] <= shmem.shape[0]:
                memory_OK = True
    return memory_OK


###############################################################################
#                               Shareds                                       #


class Shareds(SynkVariables):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gpuarrays = list()
        self.shapes = list()
        self.avg_funcs = list()
        self.avg_facs = list()
        self.shmem_tag_pre = SHRD_SHMEM_TAG_PRE

    def include(self, var, build_avg_func):
        is_new_var, var_ID = self._include(var)
        if is_new_var:
            self.gpuarrays.append(var.container.data)
            self.shapes.append(var.container.data.shape)
            if build_avg_func:  # (only in master)
                dtype = self.dtypes[var_ID]
                avg_fac = theano.shared(np.array(1, dtype=dtype),
                                        name=AVG_FAC_NAME)
                avg_func = theano.function([], updates={var: var * avg_fac})
                self.avg_facs.append(avg_fac)
                self.avg_funcs.append(avg_func)
        return var_ID

    def register_func(self, theano_function, build_avg_func=True):
        shared_IDs = list()
        shareds = theano_function.get_shared()
        for var in shareds:
            shared_IDs.append(self.include(var, build_avg_func))
        return tuple(shared_IDs)

    def alloc_shmem(self, shared_ID, rank, create=True):
        shape = self.shapes[shared_ID]
        shmem = self._alloc_shmem(shared_ID, shape, rank, create)
        if not create:  # (will be organized differently in master)
            self.shmems[shared_ID] = shmem
        return shmem

    def build_shmems(self, shared_ID, n_gpu, master_rank):
        shmems = list()
        for rank in range(n_gpu):
            if rank == master_rank:
                shmems.append(None)
            else:
                shmems.append(self.alloc_shmem(shared_ID, rank))
        self.shmems[shared_ID] = shmems
        return shmems

    def build_sync(self, create=True):
        if self.sync is not None:
            raise RuntimeError("Tried to build sync on shareds a second time.")
        if self.num == 0:
            sync = None
        else:
            sync = struct(
                shared_IDs=ShmemRawArray('i', self.num, SHRD_ARRAY_TAG, create),
            )
        self.sync = sync
        return sync

    def set_avg_facs(self, n_gpu):
        for avg_fac in self.avg_facs:
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
        for avg_fac in self.avg_facs:
            avg_fac.set_value(1 / n_gpu)


###############################################################################
#                                                                             #
#                     Base Function (master & workers)                        #
#                                                                             #
###############################################################################


class SynkFunction(object):

    def __init__(self,
                 ID,
                 theano_function,
                 input_IDs,
                 inputs_scatter,
                 collect_modes,
                 reduce_ops,
                 ):
        self._ID = ID
        self._theano_function = theano_function
        self._input_IDs = input_IDs
        self._inputs_scatter = inputs_scatter
        self._collect_modes = collect_modes
        self._reduce_ops = reduce_ops

    def _build_output_subset_shmem(self, create=True):
        n_outputs = len(self._collect_modes)
        if n_outputs == 0:
            self._output_subset_shmem = []
        else:
            self._output_subset_shmem = ShmemRawArray(
                ctypes.c_bool,
                [True] * len(self._collect_modes),  # (n_outputs)
                OTPT_SBST_TAG_PRE + str(self._ID),
                create,
            )

    @property
    def theano_function(self):
        return self._theano_function

    @property
    def inputs_scatter(self):
        return self._inputs_scatter

    @property
    def collect_modes(self):
        return self._collect_modes

    @property
    def reduce_ops(self):
        return self._reduce_ops

    def _call_theano_function(self, inputs, output_subset=None):
        results = self._theano_function(*inputs, output_subset=output_subset)
        if not isinstance(results, list):
            results = [results]
        return results  # (always returns a list, even if length 1)
