
from .scatterer import scatterer
from .data_module import Data


###############################################################################
#                                                                             #
#                       API for building Synk Data                            #
#                                                                             #
###############################################################################


def data(value=None, var=None, dtype=None, ndim=None, shape=None,
         minibatch=False, force_cast=False, oversize=1, name=None):
    """Returns a ``synkhronos.Data`` object, for data input to functions.
    Similar to a Theano variable, Data objects have fixed ndim and dtype.  It
    is optional to populate this object with actual data or assign a shape 
    (induces memory allocation) at instantiation.
    
    Args:
        value: Data values to be stored (e.g. numpy array)
        var (Theano variable): To infer dtype and ndim
        dtype: Can specify dtype (if not implied by var)
        ndim: Can specify ndim (if not implied if not implied)
        shape: Can specify shape (if not implied by value)
        minibatch (bool, optional): Use for minibatch data inputs (compare
            to full dataset inputs)
        force_cast (bool, optional): If True, force value to specified dtype
        oversize (int, [1,2], optional):  Factor for OS shared memory 
            allocation in excess of given value or shape
        name: As in Theano variables
    
    Returns:
        synkhronos.Data: used for data input to functions
    
    Raises:
        TypeError: If incomplete specification of dtype and ndim.
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
