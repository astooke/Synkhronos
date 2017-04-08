
import theano
import theano.tensor as T

# TODO:  Use inplace operators for all of these.


def make_avg_f(var):
    y = T.scalar('avg_fact', dtype=var.dtype)
    t_type = T.TensorType(dtype=var.dtype, broadcastable=var.broadcastable)
    x = t_type('accum').transfer(None)
    z = x * y  # * (1 / y)
    return theano.function([x, y], z.transfer(None), name='avg')


def make_accum_f(var, mode):
    dtype = var.dtype
    bcast = var.broadcastable
    t_type = T.TensorType(dtype=dtype, broadcastable=bcast)
    x = t_type('accum').transfer(None)
    y = t_type('slice').transfer(None)
    if mode == "gather":
        z = T.concatenate([x, y])
    else:
        T_op = getattr(T, mode)
        x_pad = T.shape_padaxis(x, axis=0)
        y_pad = T.shape_padaxis(y, axis=0)
        z = T_op(T.concatenate([x_pad, y_pad], axis=0), axis=0)
    name = mode + "_" + str(dtype) + broadcastable_string(bcast)
    return theano.function([x, y], z.transfer(None), name=name)


def broadcastable_string(broadcastable):
    bcast = ""
    for b in broadcastable:
        bcast += "T" if b else "F"
    return bcast


class Accumulators(object):

    def __init__(self):
        self.accum_fs = dict()  # functions cached in nested dictionaries
        self.avg_fs = dict()

    def get_accum_f(self, var, mode):

        if mode is None:
            return lambda x, y: y

        mode = mode.lstrip("c_")
        dtype = var.dtype
        if mode == "avg" and "int" in dtype:
            raise TypeError("Cannot average integer dtype: {}".format(dtype))
        mode = "sum" if mode == "avg" else mode
        bcast = broadcastable_string(var.broadcastable)

        # Try to find existing function.
        this_mode = self.accum_fs.get(mode, None)
        if this_mode is not None:
            this_dtype = this_mode.get(dtype, None)
            if this_dtype is not None:
                this_bcast = this_dtype.get(bcast, None)
                if this_bcast is not None:
                    return this_bcast

        # Did not find it; make it.
        accum_f = make_accum_f(var, mode)

        # Put the function in the cache.
        this_mode = self.accum_fs.get(mode, None)
        if this_mode is None:
            self.accum_fs[mode] = dict()
            this_mode = self.accum_fs[mode]
        this_dtype = this_mode.get(dtype, None)
        if this_dtype is None:
            this_mode[dtype] = dict()
            this_dtype = this_mode[dtype]
        this_dtype[bcast] = accum_f

        return accum_f

    def get_avg_f(self, var):

        dtype = var.dtype
        bcast = broadcastable_string(var.broadcastable)

        this_dtype = self.avg_fs.get(dtype, None)
        if this_dtype is not None:
            this_bcast = this_dtype.get(bcast, None)
            if this_bcast is not None:
                return this_bcast

        avg_f = make_avg_f(var)

        this_dtype = self.avg_fs.get(dtype, None)
        if this_dtype is None:
            self.avg_fs[dtype] = dict()
            this_dtype = self.avg_fs[dtype]
        this_dtype[bcast] = avg_f

        return avg_f


accumulators = Accumulators()
