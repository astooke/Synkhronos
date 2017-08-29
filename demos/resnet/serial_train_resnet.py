
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import time

from demos.resnet.common import build_resnet, load_data, unflatten_params


def simple_sgd(lr, tparams, grads, x, y, cost):
    """ Was just for a little speed test...can ignore"""
    updates = [(p, p - lr * g) for p, g in zip(tparams, grads)]
    f_grad = theano.function([x, y, lr], cost, updates=updates)
    return f_grad


def sgd_vec(lr, tparams, grads, x, y, cost):
    """ like sgd but combines all the params into one vector (anticipate nccl)
    """
    flat_grad = T.concatenate([T.flatten(g) for g in grads])
    param_shapes = [p.get_value(borrow=True).shape for p in tparams]
    param_sizes = [p.get_value(borrow=True).size for p in tparams]
    flat_grad_size = np.sum(param_sizes)
    gshared_flat = theano.shared(np.empty(flat_grad_size,
                                 dtype=theano.config.floatX), name='gshared_flat')
    grad_flat_update = [(gshared_flat, flat_grad)]

    f_grad_shared = theano.function([x, y], cost, updates=grad_flat_update,
                                    name='sgd_f_grad_shared_flat')

    unflatten_updates = list()
    last_sz = 0
    for p, sz, sh in zip(tparams, param_sizes, param_shapes):
        inc = T.reshape(gshared_flat[last_sz:last_sz + sz], sh)
        upd = (p, p - lr * inc)
        unflatten_updates.append(upd)
        last_sz += sz

    f_update = theano.function([lr], [], updates=unflatten_updates,
                                name='sgd_f_update_unflatten')

    return f_grad_shared, f_update, gshared_flat


def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent
    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.
    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % p.name,
                             broadcastable=p.broadcastable)
               for p in tparams]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams, gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update, gshared


def iter_mb_idxs(mb_size, data_length, shuffle=False):
    if shuffle:
        indices = np.arange(data_length)
        np.random.shuffle(indices)
    for low in range(0, data_length - mb_size + 1, mb_size):
        if shuffle:
            mb_idxs = indices[low:low + mb_size]
        else:
            mb_idxs = slice(low, low + mb_size)
        yield mb_idxs


def train_resnet(
    batch_size=32,  # batch size on each GPU
    validFreq=1,
    lrate=1e-4,
    optimizer=sgd_vec,
    n_epoch=2,
    n_gpu=1,  # later get this from synk.fork
):

    t_0 = time.time()
    print("Loading data (pretend)")
    train, valid, test = load_data()
    x_train, y_train = train
    x_valid, y_valid = valid
    x_test, y_test = test

    print("Building model")
    resnet = build_resnet()
    params = L.get_all_params(resnet.values(), trainable=True)

    x = T.ftensor4('x')
    y = T.imatrix('y')

    prob = L.get_output(resnet['prob'], x, deterministic=False)
    loss = T.nnet.categorical_crossentropy(prob, y.flatten()).mean()

    print("Building update rules")
    grads = T.grad(loss, wrt=params)
    lr = T.scalar(name='lr')
    f_grad_shared, f_update, gshared = optimizer(lr, params, grads, x, y, loss)
    # f_grad = simple_sgd(lr, params, grads, x, y, loss)

    print("Building validation functions")
    v_prob = L.get_output(resnet['prob'], x, deterministic=True)
    v_loss = T.nnet.categorical_crossentropy(v_prob, y.flatten()).mean()
    v_mc = T.mean(T.neq(T.argmax(v_prob, axis=1), y.flatten()))
    f_pred = theano.function([x, y], [v_loss, v_mc])

    t_1 = time.time()
    print("Total setup time: {:,.1f} s".format(t_1 - t_0))
    print("Starting training")
    full_mb_size = batch_size * n_gpu
    t_last = t_1
    for ep in range(n_epoch):
        train_loss = 0.
        i = 0
        for mb_idxs in iter_mb_idxs(full_mb_size, len(x_train), shuffle=True):
            # train_loss += f_grad(x_train[mb_idxs], y_train[mb_idxs], lrate)
            train_loss += f_grad_shared(x_train[mb_idxs], y_train[mb_idxs])
            f_update(lrate)
            i += 1
        train_loss /= i

        print("\nEpoch: ", ep)
        print("Training Loss: {:.3f}".format(train_loss))

        if ep % validFreq == 0:
            valid_loss = valid_mc = 0.
            i = 0
            for mb_idxs in iter_mb_idxs(full_mb_size, len(x_valid), shuffle=False):
                mb_loss, mb_mc = f_pred(x_valid[mb_idxs], y_valid[mb_idxs])
                valid_loss += mb_loss
                valid_mc += mb_mc
                i += 1
            valid_loss /= i
            valid_acc = 1 - (valid_mc / i)
            print("Validation Loss: {:3f},   Accuracy: {:3f}".format(valid_loss, valid_acc))

        t_2 = time.time()
        print("(epoch total time: {:,.1f} s)".format(t_2 - t_last))
        t_last = t_2
    print("\nTotal training time: {:,.1f} s".format(t_last - t_1))


if __name__ == "__main__":
    import theano.gpuarray
    theano.gpuarray.use("cuda", preallocate=1)
    train_resnet()
