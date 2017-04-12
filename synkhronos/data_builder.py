
from .scatterer import scatterer
from .data_module import Data


###############################################################################
#                                                                             #
#                       API for building Synk Data                            #
#                                                                             #
###############################################################################


def data(var=None, value=None, dtype=None, ndim=None, shape=None,
         minibatch=False, force_cast=False, oversize=1, name=None):
    """ Returns a Data object, which is the only type that synkhronos
    functions can receive for Theano inputs.
    """
    if var is not None:
        dtype = var.dtype
        ndim = var.ndim
    elif value is not None:
        try:
            dtype = value.dtype
            ndim = value.ndim
        except AttributeError:
            pass
    if dtype is None or (ndim is None and shape is None):
        raise TypeError("Must provide either 1) variable, 2) value "
            "object with dtype and ndim attributes, or 3) dtype and "
            "either ndim or shape.")
    if ndim is None:
        ndim = len(shape)

    synk_data = Data(len(scatterer), dtype, ndim, minibatch, name)
    scatterer.append(synk_data)

    if value is not None:
        synk_data.set_value(value, force_cast, oversize)
    elif shape is not None:
        synk_data.set_shape(shape, oversize)
    return synk_data
