
import theano
import theano.tensor as T

# TODO: Support for CPU-based variables.


def make_avg_f(dtype, ndim):
    y = T.scalar('avg_fact', dtype=dtype)
    t_type = T.TensorType(dtype=dtype, broadcastable=[False] * ndim)
    x = t_type('accum').transfer(None)
    z = x * y
    return theano.function([x, y], z.transfer(None), name='avg', allow_input_downcast=True)


def make_reduce_f(mode, dtype, ndim):
    t_type = T.TensorType(dtype=dtype, broadcastable=[False] * ndim)
    x = t_type('accum').transfer(None)
    y = t_type('slice').transfer(None)
    if mode == "gather":
        z = T.concatenate([x, y])
    else:
        T_op = getattr(T, mode)
        x_pad = T.shape_padaxis(x, axis=0)
        y_pad = T.shape_padaxis(y, axis=0)
        z = T_op(T.concatenate([x_pad, y_pad], axis=0), axis=0)
    name = mode + "_" + str(dtype)
    return theano.function([x, y], z.transfer(None), name=name, allow_input_downcast=True)


class Reducers(object):

    def __init__(self):
        self.reduce_fs = dict()  # functions cached in nested dictionaries
        self.avg_fs = dict()

    def get_reduce_f(self, var_or_arr, mode):

        if mode is None:
            return lambda x, y: y

        mode = mode.lstrip("c_")
        dtype = var_or_arr.dtype
        ndim = var_or_arr.ndim
        if mode == "avg" and "int" in dtype:
            raise TypeError("Cannot average integer dtype: {}".format(dtype))
        mode = "sum" if mode == "avg" else mode

        # Try to find existing function.
        this_mode = self.reduce_fs.get(mode, None)
        if this_mode is not None:
            this_dtype = this_mode.get(dtype, None)
            if this_dtype is not None:
                this_ndim = this_dtype.get(ndim, None)
                if this_ndim is not None:
                    return this_ndim

        # Did not find it; make it.
        reduce_f = make_reduce_f(mode, dtype, ndim)

        # Put the function in the cache.
        this_mode = self.reduce_fs.get(mode, None)
        if this_mode is None:
            self.reduce_fs[mode] = dict()
            this_mode = self.reduce_fs[mode]
        this_dtype = this_mode.get(dtype, None)
        if this_dtype is None:
            this_mode[dtype] = dict()
            this_dtype = this_mode[dtype]
        this_dtype[ndim] = reduce_f

        return reduce_f

    def get_avg_f(self, var_or_arr):

        dtype = var_or_arr.dtype
        ndim = var_or_arr.ndim

        this_dtype = self.avg_fs.get(dtype, None)
        if this_dtype is not None:
            this_ndim = this_dtype.get(ndim, None)
            if this_ndim is not None:
                return this_ndim

        avg_f = make_avg_f(dtype, ndim)

        this_dtype = self.avg_fs.get(dtype, None)
        if this_dtype is None:
            self.avg_fs[dtype] = dict()
            this_dtype = self.avg_fs[dtype]
        this_dtype[ndim] = avg_f

        return avg_f


reducers = Reducers()
