
import theano
import theano.tensor as T
import numpy as np


###############################################################################
#
# Utilities for reshaping variables, to make only one call to NCCL all_reduce.
#
###############################################################################


def flatten_vars(variables):
    """
    input: list or tuple of Theano tensor or shared variables
    output: Theano tensor variable, concatenation of all variables flattened
    """
    return T.concatenate([T.flatten(v) for v in variables])


def vars_flat_size(variables):
    """
    input: list or tuple of Theano shared variables
    output: int, sum of sizes of all variables
    """
    return sum([v.get_value(borrow=True).size for v in variables])


def unflatten_vars(flat_var, variables):
    """
    inputs: 1) flat (1-dim) Theano shared variable from which to draw slices
            2) list of Theano shared variables from which to draw shapes
    outputs: list of variables, shaped as inputs, referencing flat_var slices
    """
    unflat_params = list()
    i = 0
    for v in variables:
        val = v.get_value(borrow=True)
        unflat_params.append(T.reshape(flat_var[i:i + val.size], val.shape))
        i += val.size
    return unflat_params


###############################################################################
#
# Utilities for computing gradients, flattening and unflattening (or not)
#
###############################################################################


def flat_unflat_grads(loss, params):
    """ use this one to make the raw gradient one vector for all-reduce """
    grads = T.grad(loss, wrt=params)
    flat_grad = flatten_vars(grads)
    grad_shared_flat = theano.shared(np.zeros(vars_flat_size(params),
                                              dtype=theano.config.floatX),
                                     name='flat_grad_s')
    unflat_grads = unflatten_vars(grad_shared_flat, params)
    return grad_shared_flat, flat_grad, unflat_grads


def nonflat_grads(loss, params):
    """ not recommended to use this; demonstrates the pattern w/o flattening """
    grads = T.grad(loss, wrt=params)
    grads_shared = list()
    for p in params:
        value = p.get_value(borrow=True)
        g = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                          broadcastable=p.broadcastable)
        grads_shared.append(g)
    return grads_shared, grads


###############################################################################
#
# Update rules, adapted from Lasagne/lasagne/updates.py (Aug 2017)
#
# To be used as in Synkhronos/extensions/train.py
#
# All of these update rule generates return three outputs:
# 1) Update rules used (by workers) to compute raw gradient and store result
#    to a single vector Theano shared variable.
# 2) Update rules used (by workers) to update network parameters; use after
#    calling synk.all_reduce() on the raw gradient shared variable and all
#    all workers will produce the same (deterministic) parameter update.
# 3) Theano shared variable (or list of them, if not flattened), which holds
#    the result of the raw gradient computation; to be all-reduced before
#    parameter updates.  (all-reduce is faster on single, large variable)
#
###############################################################################


def sgd(loss, params, lr):
    grad_shared_flat, flat_grad, unflat_grads = flat_unflat_grads(loss, params)
    grad_updates = [(grad_shared_flat, flat_grad)]
    param_updates = [(p, p - lr * g) for p, g in zip(params, unflat_grads)]
    return grad_updates, param_updates, grad_shared_flat


def apply_momentum(param_updates, momentum=0.9):
    updates = list()
    for p, update in param_updates:
        value = p.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=p.broadcastable)
        x = momentum * velocity + update
        updates += [(velocity, x - p), (p, x)]
    return updates


def apply_nesterov_momentum(param_updates, momentum=0.9):
    updates = list()
    for p, update in param_updates:
        value = p.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=p.broadcastable)
        x = momentum * velocity + update - p
        updates += [(velocity, x), (p, momentum * x + update)]
    return updates


def momentum(loss, params, lr, momentum=0.9):
    grad_updates, param_updates, grads_shared = sgd(lr, params, loss)
    param_updates = apply_momentum(param_updates, momentum=momentum)
    return grad_updates, param_updates, grads_shared


def nesterov_momentum(loss, params, lr, momentum=0.9):
    grad_updates, param_updates, grads_shared = sgd(lr, params, loss)
    param_updates = apply_nesterov_momentum(param_updates, momentum=momentum)
    return grad_updates, param_updates, grads_shared


def adagrad(loss, params, lr=1.0, epsilon=1e-6):
    grad_shared_flat, flat_grad, unflat_grads = flat_unflat_grads(loss, params)
    grad_updates = [(grad_shared_flat, flat_grad)]
    param_updates = list()
    for p, g in zip(params, unflat_grads):
        value = p.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=p.broadcastable)
        accu_new = accu + g ** 2
        param_updates += [(accu, accu_new)]
        param_updates += [(p, p - (lr * g / T.sqrt(accu_new + epsilon)))]
    return grad_updates, param_updates, grad_shared_flat


def rmsprop(loss, params, lr=1.0, rho=0.9, epsilon=1e-6):
    grad_shared_flat, flat_grad, unflat_grads = flat_unflat_grads(loss, params)
    grad_updates = [(grad_shared_flat, flat_grad)]
    one = T.constant(1)
    param_updates = list()
    for p, g in zip(params, unflat_grads):
        value = p.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=p.broadcastable)
        accu_new = rho * accu + (one - rho) * g ** 2
        param_updates += [(accu, accu_new)]
        param_updates += [(p, p - (lr * g / T.sqrt(accu_new + epsilon)))]
    return grad_updates, param_updates, grad_shared_flat


def adadelta(loss, params, lr=1.0, rho=0.95, epsilon=1e-6):
    grad_shared_flat, flat_grad, unflat_grads = flat_unflat_grads(loss, params)
    grad_updates = [(grad_shared_flat, flat_grad)]
    one = T.constant(1)
    param_updates = list()
    for p, g in zip(params, unflat_grads):
        value = p.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=p.broadcastable)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=p.broadcastable)
        accu_new = rho * accu + (one - rho) * g ** 2
        update = g * T.sqrt(delta_accu + epsilon) / T.sqrt(accu_new + epsilon)
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        param_updates += [(accu, accu_new)]
        param_updates += [(p, p - lr * update)]
        param_updates += [(delta_accu, delta_accu_new)]
    return grad_updates, param_updates, grad_shared_flat


def adam(loss, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    grad_shared_flat, flat_grad, unflat_grads = flat_unflat_grads(loss, params)
    grad_updates = [(grad_shared_flat, flat_grad)]
    t_prev = theano.shared(np.array(0, dtype=theano.config.floatX))
    one = T.constant(1)
    t = t_prev + one
    a_t = lr * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)
    param_updates = list()
    for p, g in zip(params, unflat_grads):
        value = p.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=p.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=p.broadcastable)
        m_t = beta1 * m_prev + (one - beta1) * g
        v_t = beta2 * v_prev + (one - beta2) * g ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)
        param_updates += [(m_prev, m_t), (v_prev, v_t), (p, p - step)]
        param_updates += [(t_prev, t)]
    return grad_updates, param_updates, grad_shared_flat


def adamax(loss, params, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    grad_shared_flat, flat_grad, unflat_grads = flat_unflat_grads(loss, params)
    grad_updates = [(grad_shared_flat, flat_grad)]
    t_prev = theano.shared(np.array(0, dtype=theano.config.floatX))
    one = T.constant(1)
    t = t_prev + one
    a_t = lr / (one - beta1 ** t)
    param_updates = list()
    for p, g in zip(params, unflat_grads):
        value = p.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=p.broadcastable)
        u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=p.broadcastable)
        m_t = beta1 * m_prev + (one - beta1) * g
        u_t = T.maximum(beta2 * u_prev, abs(g))
        step = a_t * m_t / (u_t + epsilon)
        param_updates += [(m_prev, m_t), (u_prev, u_t), (p, p - step)]
        param_updates += [(t_prev, t)]
    return grad_updates, param_updates, grad_shared_flat


def sgd_nonflat(loss, params, lr):
    """ Provided for speed test to measure improvement from flattening """
    grads, grads_shared = nonflat_grads(loss, params)
    grad_updates = [(g_s, g) for g_s, g in zip(grads_shared, grads)]
    param_updates = [(p, p - lr * g_s) for p, g_s in zip(params, grads_shared)]
    return grad_updates, param_updates, grads_shared
