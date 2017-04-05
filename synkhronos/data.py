
import numpy as np

from .shmemarray import ShmemRawArray, NP_TO_C_TYPE
from .common import PRE, DATA_ALLOC, DATA_RESHAPE


###############################################################################
#                                                                             #
#          Base (& worker) Data Container for Inputs & Shareds                #
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
#                Helper Methods for Master Synk Data                          #
#                                                                             #
###############################################################################


class DataHelpers(BaseData):

    _create = True

    def _signal(self, *args, **kwargs):
        raise NotImplementedError  # (exct in master only)

    def _update_array(self, sync_data, shape, oversize):
        if shape != self.shape:
            size = int(np.prod(shape))
            if size > self._alloc_size:
                self._alloc_and_signal(sync_data, shape, float(oversize))
            else:
                self._shape_and_signal(sync_data, shape)

    def _alloc_and_signal(self, sync_data, shape, oversize):
        self._tag += 1
        if oversize < 1 or oversize > 2:
            raise ValueError("param 'oversize' must be in range [1, 2].")
        size = int(np.prod(shape) * oversize)
        self._alloc_shmem(size, self._tag)
        sync_data.alloc_size.value = size
        sync_data.tag.value = self._tag
        sync_data.ID.value = self._ID
        sync_data.shape[:self.ndim] = shape
        self._shape_data(shape)
        self._signal(DATA_ALLOC)

    def _shape_and_signal(self, sync_data, shape):
        sync_data.ID.value = self._ID
        sync_data.shape[:self.ndim] = shape
        self._shape_data(shape)
        self._signal(DATA_RESHAPE)

    def _condition_data(self, input_data, force_cast):
        """ takes in any data and returns numpy array """
        if force_cast:
            if not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data, dtype=self.dtype)
            elif input_data.dtype != self.dtype:
                input_data = input_data.astype(self.dtype)
        else:
            if not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data)
            if input_data.dtype != self.dtype:
                common_dtype = np.find_common_type([input_data.dtype, self.dtype], [])
                if common_dtype == self.dtype:
                    input_data = input_data.astype(self.dtype)
                else:
                    raise TypeError("Non up-castable data type provided for "
                        "input..., received: {}, expected: {}.  Could use param "
                        "'force_cast=True' to force to expected dtype.".format(
                            input_data.dtype, self.dtype))
        if input_data.ndim != self.ndim:
            raise TypeError("Wrong data ndim provided for data, received: "
                "{}, expected: {}".format(input_data.ndim, self.ndim))
        return input_data
