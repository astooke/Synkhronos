

import theano
import synkhronos as synk


###############################################################################
#
# Utilities for building training functions using the update functions in
# Synkhronos/extensions/updates.py
# (valid arg for "update_rule" includes the functions sgd, rmsprop, etc.)
#
###############################################################################


def make_train_function(loss, params, x, y, update_rule, *args, **kwargs):
    grad_updates, param_updates, grad_shared = \
        update_rule(loss, params, *args, **kwargs)
    f_grad_shared = synk.function(inputs=[x, y],
                                  outputs=loss,  # (assumes this is an avg)
                                  updates=grad_updates)
    f_param_update = synk.function(inputs=[], updates=param_updates)

    def train_minibatch(x_data, y_data):
        train_loss = f_grad_shared(x_data, y_data)
        synk.all_reduce(grad_shared, op="avg")  # (assumes loss is an avg)
        f_param_update()
        return train_loss

    return train_minibatch


def make_train_function_theano(loss, params, x, y, update_rule, *args, **kwargs):
    grad_updates, param_updates, grad_shared = \
        update_rule(loss, params, *args, **kwargs)
    f_grad_shared = theano.function(inputs=[x, y],
                                    outputs=loss,
                                    updates=grad_updates)
    f_param_update = theano.function(inputs=[], updates=param_updates)

    def train_minibatch(x_data, y_data):
        train_loss = f_grad_shared(x_data, y_data)
        f_param_update()
        return train_loss

    return train_minibatch
