
import numpy as np

from .scatterer import scatterer
from .data import Data

###############################################################################
#                                                                             #
#                       API for building Synk Data                            #
#                                                                             #
###############################################################################


def data(var_or_arr=None, dtype=None, ndim=None, shape=None,
         minibatch=False, force_cast=False, oversize=1, name=None):
    """ Returns a Data object, which is the only type that synkhronos
    functions can receive for Theano inputs.
    """
    if var_or_arr is not None:
        try:
            dtype = var_or_arr.dtype
            ndim = var_or_arr.ndim
        except AttributeError as exc:
            raise exc("Input 'var_or_arr' must have dtype and ndim attributes.")
    elif dtype is None or (ndim is None and shape is None):
        raise TypeError("Must provide either 1) variable or array, or 2) dtype "
            "and either ndim or shape.")
    if shape is not None:
        if ndim is not None:
            if ndim != len(shape):
                raise ValueError("Received inconsistent shape and ndim values.")
        ndim = len(shape)
    synk_data = Data(len(scatterer), dtype, ndim, minibatch, name)
    scatterer.append(synk_data)

    if isinstance(var_or_arr, np.ndarray):
        synk_data.set_value(var_or_arr, force_cast, oversize)
    elif shape is not None:
        synk_data.set_shape(shape, oversize)
    return synk_data
