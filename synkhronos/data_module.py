
import numpy as np
from ctypes import c_char

from .shmemarray import ShmemRawArray
from .util import PREFIX
from . import exct


sync = None


###############################################################################
#                                                                             #
#          Base and Worker Data Container for System Shared Memory            #
#                                                                             #
###############################################################################


class BaseData(object):

    _create = False

    def __init__(self, ID, dtype, ndim, minibatch):
        self._ID = ID
        self._dtype = dtype.name if hasattr(dtype, "name") else dtype
        self._itemsize = np.dtype(self._dtype).itemsize
        self._ndim = ndim
        self._data = np.empty([0] * ndim, dtype=dtype)  # NOTE: why?
        self._minibatch = minibatch
        self._tag = 0
        self._shmem = None
        self._np_shmem = None
        self._alloc_size = 0

    def _alloc_shmem(self, size, tag):
        tag = PREFIX + "_data_" + str(self._ID) + "_" + str(tag)
        nbytes = size * self._itemsize
        self._shmem = ShmemRawArray(c_char, nbytes, tag, self._create)
        self._np_shmem = np.frombuffer(self._shmem, dtype=self._dtype, count=size)
        # self._np_shmem = np.ctypeslib.as_array(self._shmem)
        self._alloc_size = size

    def _shape_data(self, shape):
        self._data = self._np_shmem if len(shape) == 0 else \
            self._np_shmem[:int(np.prod(shape))].reshape(shape)

    def _free_shmem(self):
        self._data = np.empty([0] * self._ndim, dtype=self._dtype)  # NOTE: why?
        self._np_shmem = None
        self._shmem = None
        self._alloc_size = 0


class WorkerData(BaseData):

    def __init__(self, ID):
        assert ID == sync.ID.value
        dtype = sync.dtype.value.decode('utf-8')
        ndim = sync.ndim.value
        minibatch = sync.minibatch.value
        super().__init__(ID, dtype, ndim, minibatch)

    def alloc_shmem(self):
        size = sync.alloc_size.value
        tag = sync.tag.value
        self._alloc_shmem(size, tag)

    def shape_data(self):
        shape = sync.shape[:self._ndim]
        self._shape_data(shape)

    def free_memory(self):
        self._free_shmem()


###############################################################################
#                                                                             #
#                Helper Methods for Master Synk Data                          #
#                                                                             #
###############################################################################


class DataHelpers(BaseData):

    _create = True

    def __init__(self, ID, dtype, ndim, minibatch=False, name=None):
        super().__init__(ID, dtype, ndim, minibatch)
        sync.ID.value = ID
        sync.dtype.value = bytes(self._dtype, encoding='utf-8')
        sync.ndim.value = ndim
        sync.minibatch.value = minibatch
        exct.launch(exct.DATA, exct.CREATE)
        self._name = name
        exct.join()

    def _update_array(self, shape, oversize):
        if shape != self.shape:
            size = int(np.prod(shape))
            if size > self._alloc_size:
                self._alloc_and_signal(shape, float(oversize))
            else:
                self._shape_and_signal(shape)

    def _alloc_and_signal(self, shape, oversize):
        self._tag += 1
        if oversize < 1 or oversize > 2:
            raise ValueError("param 'oversize' must be in range [1, 2].")
        size = int(np.prod(shape) * oversize)
        self._alloc_shmem(size, self._tag)
        sync.alloc_size.value = size
        sync.tag.value = self._tag
        sync.ID.value = self._ID
        sync.shape[:self._ndim] = shape
        exct.launch(exct.DATA, exct.ALLOC)
        self._shape_data(shape)
        exct.join()

    def _shape_and_signal(self, shape):
        sync.ID.value = self._ID
        sync.shape[:self._ndim] = shape
        exct.launch(exct.DATA, exct.RESHAPE)
        self._shape_data(shape)
        exct.join()

    def _condition_data(self, input_data, force_cast):
        """ takes in any data and returns numpy array """
        if force_cast:
            if not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data, dtype=self._dtype)
            elif input_data.dtype != self._dtype:
                input_data = input_data.astype(self._dtype)
        else:
            if not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data)
            if input_data.dtype != self._dtype:
                common_dtype = np.find_common_type([input_data.dtype, self._dtype], [])
                if common_dtype == self._dtype:
                    input_data = input_data.astype(self._dtype)
                else:
                    raise TypeError("Non up-castable data type provided for "
                        "input..., received: {}, expected: {}.  Could use param "
                        "'force_cast=True' to force to expected dtype.".format(
                            input_data.dtype, self._dtype))
        if input_data.ndim != self._ndim:
            raise TypeError("Wrong data ndim provided for data, received: "
                "{}, expected: {}".format(input_data.ndim, self._ndim))
        return input_data


###############################################################################
#                                                                             #
#                       API for (Master) Synk Data                            #
#                                                                             #
###############################################################################


class Data(DataHelpers):
    """Type of object required as inputs to functions (instead of numpy
    arrays).  May also be used as inputs to some collectives.  Underlying
    memory is OS shared memory, for multi-process access.  Presents a similar
    interface as a numpy array, with some additions and restrictions.  The
    terms "numpy wrapper" and "underlying numpy array" refer to the same
    object, which is a view to the underlying memory allocation.
    """

    def __getitem__(self, k):
        """Provided to read from underlying numpy data array, as in array[k].
        Full numpy indexing is supported.
        """
        return self._data[k]

    def __setitem__(self, k, v):
        """Provided to write to underlying numpy data array, as in array[k] =
        v.  Full numpy indexing is supported.
        """
        self._data[k] = v

    def __len__(self):
        """Returns len() of underlying numpy data array."""
        return len(self._data)

    @property
    def dtype(self):
        """Returns data type."""
        return self._dtype

    @property
    def ndim(self):
        """Returns number of dimensions."""
        return self._data.ndim

    @property
    def shape(self):
        """Returns shape of underlying numpy data array."""
        return self._data.shape

    @property
    def size(self):
        """Returns size of underlying numpy data array."""
        return self._data.size

    @property
    def data(self):
        """Returns underlying numpy data array.  In general, it is not
        recommended to manipulate this object directly, aside from reading
        from or writing to it (without changing shape).  It may be passed 
        to other python processes and used as shared memory.
        """
        return self._data

    @property
    def alloc_size(self):
        """Returns the size of the underlying memory allocation (units: number
        of items).  This may be larger than the size of the underlying numpy
        array, which may occupy only a portion of the allocation (always
        starting at the same memory address as the allocation).
        """
        return self._alloc_size

    @property
    def name(self):
        """Returns the name (may be None)"""
        return self._name

    @property
    def minibatch(self):
        """Returns whether this data is treated as a minibatch.  When using the
        ``batch`` kwarg in a function call, all minibatch data inputs will have their
        selection indexes shifted so that the lowest overall index present in ``batch``
        corresponds to index 0.  (It is enforced that all data is long enough to meet
        all requested indexes.)
        """
        return self._minibatch

    def set_value(self, input_data, force_cast=False, oversize=1):
        """(Over)Write data values.  Change length, reshape, and/or reallocate shared
        memory if necessary (applies eagerly in workers).  

        Args:
            input_data: e.g. numpy array, fed into numpy.asarray()
            force_cast (bool, optional): force input data to existing dtype of data 
                object, without error or warning
            oversize (int, [1-2]): Factor for oversizing memory allocation
                relative to input data size

        Oversize applies only to underlying shared memory.  The numpy array wrapper
        will have the exact shape of ``input_data``.  
        """
        input_data = self._condition_data(input_data, force_cast)
        self._update_array(input_data.shape, oversize)
        self._data[:] = input_data

    def set_length(self, length, oversize=1):
        """Change length of underlying numpy array.  Will induce memory
        reallocation if necessary (for length larger than current memory).

        Args:
            length (int): New length of underlying numpy array
            oversize (int, [1,2]): Used only if reallocating memory
        
        Warning:
            Currently, memory reallocation loses old data.
        """
        length = int(length)
        if length < 1:
            raise ValueError("Length must be a positive integer.")
        shape = list(self.shape)
        shape[0] = length
        self._update_array(shape, oversize)

    def set_shape(self, shape, oversize=1):
        """Change shape of underlying numpy array.  Will induce memory
        reallocation if necessary (for shape larger than current memory).
        
        Args:
            shape (list, tuple): New shape of underlying numpy array
            oversize (int, [1,2]): Used only if reallocating memory
        
        Warning:
            Currently, memory reallocation loses old data.
        """
        if len(shape) != self._ndim:
            raise ValueError("Cannot change number of dimensions.")
        self._update_array(shape, oversize)

    def condition_data(self, input_data, force_cast=False):
        """Test the conditioning done to input data when calling ``set_value``
        or ``synkhronos.data()``
        
        Args:
            input_data: e.g., numpy array
            force_cast (bool, optional): force data type
        
        Returns:
            TYPE: numpy array of shape and dtype that would be used
        """
        return self._condition_data(input_data, force_cast)

    def free_memory(self):
        """ Removes all references to underlying memory (and numpy wrapper) in
        master and workers; the only way to shrink the allocation size.
        """
        self._free_shmem()
        sync.ID.value = self._ID
        exct.launch(exct.DATA, exct.FREE)
        exct.join()
