
import pickle
from .common import PKL_PATH


MODES = ["reduce", "gather", "avg_output", "avg_shared"]
OPS = ["sum", "min", "max", "prod"]


def make_name(mode, dtype, bcast, op=None):
    name = mode + "_" + dtype + "_" + bcast
    name = name if op is None else name + "_" + op
    return name


def make_accum_f(mode, var, op=None):
    import theano
    import theano.tensor as T
    dtype = var.dtype
    broadcastable = var.broadcastable
    bcast = broadcastable_string(broadcastable)
    ndim = var.ndim

    if mode == "avg_shared":
        import numpy as np
        arr = np.zeros([1] * ndim, dtype=dtype)
        s = theano.shared(arr, 's', broadcastable=broadcastable)
        y = T.scalar('avg_fac', dtype=dtype)
        name = make_name(mode, dtype, bcast, op)
        return theano.function([y], updates=[(s, s * y)], name=name)

    t_type = T.TensorType(dtype=dtype, broadcastable=broadcastable)
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
    name = make_name(mode, dtype, bcast, op)
    return theano.function([x, y], z.transfer(None), name=name)


def check_inputs(mode, op, dtype):
    if mode not in MODES:
        raise KeyError("Invalid accumulator mode: {}".format(mode))
    if mode == "avg" and "int" in dtype:
        raise KeyError("Cannot average integer dtype: {}".format(dtype))
    if op is not None and op not in OPS:
        raise KeyError("Invalid accumulator operation: {}".format(op))


def broadcastable_string(broadcastable):
    bcast = ""
    for b in broadcastable:
        if b:
            bcast += "T"
        else:
            bcast += "F"
    return bcast


class Accumulators(object):

    def __init__(self):
        self.accum_fs = dict()  # functions cached in nested dictionaries

    def get_function(self, mode, var, op=None, check_args=True):
        """
        Search unpickled cache; if not, search pickled cache; if not, build.
        """
        if check_args:
            check_inputs(mode, op, var.dtype)

        dtype = var.dtype
        bcast = broadcastable_string(var.broadcastable)

        # Try to find existing unpickled function.
        this_mode = self.accum_fs.get(mode, None)
        if this_mode is not None:
            this_dtype = this_mode.get(dtype, None)
            if this_dtype is not None:
                this_bcast = this_dtype.get(bcast, None)
                if this_bcast is not None:
                    if mode == "reduce":
                        this_op = this_bcast.get(op, None)
                        if this_op is not None:
                            return this_op
                    else:
                        return this_bcast

        # Did not find it unpickled.
        filepath = PKL_PATH + make_name(mode, dtype, bcast, op) + ".pkl"
        try:
            # Try to find it pickled.
            with open(filepath, "rb") as f:
                accum_f = pickle.load(f)
        except FileNotFoundError:
            # Did not find it pickled; create it.  (Need to be on GPU.)
            # (class is used so that only master ever does this)
            accum_f = make_accum_f(mode, var, op)
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
            this_bcast = this_dtype.get(bcast, None)
            if this_bcast is None:
                this_dtype[bcast] = dict()
                this_bcast = this_dtype[bcast]
            this_bcast[op] = accum_f
        else:
            this_dtype[bcast] = accum_f

        # accum_f.trust_input = True
        return accum_f
