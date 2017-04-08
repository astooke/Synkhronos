"""
Classes and functions used by master but which don't MODIFY globals.
(Might still read from globals passed explicitly as parameter.)
"""


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def init_gpu(rank):
    import theano.gpuarray
    theano.gpuarray.use("cuda" + str(rank))


def init_cpu(rank):
    raise NotImplementedError


###############################################################################
#                           User                                              #

def make_slices(data_collection):
    """Make a set of slice objects according to lengths of data subsets.

    Example:
        >>> slice_1, slice_2 = make_slices([[0, 1, 2, 3], [10, 11, 12]])
        >>> slice_1
        slice(0, 3, None)
        >>> slice_2
        slice(3, 6, None)

    Args:
        data_collection (list): collection of data arrays whose lengths to use

    Returns:
        slice: slice objects

    Raises:
        TypeError: if input is not list or tuple
    """
    if not isinstance(data_collection, (list, tuple)):
        raise TypeError("Expected list or tuple for input.")
    endings = [0]
    for data_arr in data_collection:
        endings.append(endings[-1] + len(data_arr))
    slices = list()
    for i in range(len(data_collection)):
        slices.append(slice(endings[i], endings[i + 1]))
    return tuple(slices)

