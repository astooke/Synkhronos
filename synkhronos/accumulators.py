
import pickle
from common import PKL_PATH


MODES = ["reduce", "gather", "avg_output", "avg_shared"]
OPS = ["sum", "min", "max", "prod"]


def make_name(mode, dtype, ndim, op=None):
    name = mode + "_" + dtype + "_" + ndim
    name = name if op is None else name + "_" + op
    return name


def make_accum_f(mode, dtype, ndim, op=None):
    import theano
    import theano.tensor as T

    if mode == "avg_shared":
        import numpy as np
        arr = np.zeros([1] * ndim, dtype=dtype)
        s = theano.shared(arr, 's')
        y = T.scalar('avg_fac', dtype=dtype)
        name = make_name(mode, dtype, ndim, op)
        return theano.function([y], updates={s: s * y}, name=name)

    t_type = T.TensorType(dtype, [False] * ndim)
    x = t_type('accum').transfer(None)
    if mode == "reduce":
        y = t_type('slice').transfer(None)
        T_op = getattr(T, op)
        x_pad = T.shape_padaxis(x, axis=0)
        y_pad = T.shape_padaxis(y, axis=0)
        z = T_op(T.concatenate([x_pad, y_pad], axis=0), axis=0)
    elif mode == "gather":
        y = t_type('slice').transfer(None)
        z = T.concatenate([x, y])
    elif mode == "avg_output":
        y = T.scalar('avg_fac', dtype=dtype)
        z = x * y
    else:
        raise ValueError("Unrecognized mode: ", mode)
    name = make_name(mode, dtype, ndim, op)
    return theano.function([x, y], z.tranfser(None), name=name)


def check_inputs(mode, dtype, ndim, op):
    if mode not in MODES:
        raise KeyError("Invalid accumulator mode: ", mode)
    if mode == "avg" and "int" in dtype:
        raise KeyError("Cannot average integer dtype: ", dtype)
    if op is not None and op not in OPS:
        raise KeyError("Invalid accumulator operation: ", op)
    ndim = int(ndim)
    if ndim < 0:
        raise KeyError("Invalid number of dimensions: ", ndim)


class Accumulators(object):

    def __init__(self):
        self.accum_fs = dict()  # functions cached in nested dictionaries

    def get_function(self, mode, dtype, ndim, op=None, check_args=True):
        """
        Search unpickled cache; if not, search pickled cache; if not, build.
        """
        if check_args:
            check_inputs(mode, dtype, ndim, op)

        # Try to find existing unpickled function.
        this_mode = self.accum_fs.get(mode, None)
        if this_mode is not None:
            this_dtype = this_mode.get(dtype, None)
            if this_dtype is not None:
                this_ndim = this_dtype.get(ndim, None)
                if this_ndim is not None:
                    if mode == "reduce":
                        this_op = this_ndim.get(op, None)
                        if this_op is not None:
                            return this_op
                    else:
                        return this_ndim

        # Did not find it unpickled.
        filepath = PKL_PATH + make_name(mode, dtype, ndim, op) + ".pkl"
        try:
            # Try to find it pickled.
            with open(filepath, "rb") as f:
                accum_f = pickle.load(f)
        except FileNotFoundError:
            # Did not find it pickled; create it.  (Need to be on GPU.)
            # (class is used so that only master ever does this)
            accum_f = make_accum_f(mode, dtype, ndim, op)
            with open(filepath, "wb") as f:
                pickle.dump(accum_f, f, pickle.HIGHEST_PROTOCOL)

        # Put the function in the unpickled cache.
        this_mode = self.accum_fs.get(mode, None)
        if this_mode is None:
            self.accum_fs[mode] = dict()
            this_mode = self.accum_fs[mode]
        this_dtype = this_mode.get(dtype, None)
        if this_dtype is None:
            this_mode[dtype] = dict()
            this_dtype = this_mode[dtype]
        if mode == "reduce":
            this_ndim = this_dtype.get(ndim, None)
            if this_ndim is None:
                this_dtype[ndim] = dict()
                this_ndim = this_dtype[ndim]
            this_ndim[op] = accum_f
        else:
            this_dtype[ndim] = accum_f

        return accum_f
