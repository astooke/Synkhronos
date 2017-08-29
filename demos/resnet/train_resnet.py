
# import theano
import theano.tensor as T
# import lasagne
import lasagne.layers as L
import time

import synkhronos as synk
from synkhronos.extensions import updates

from demos.resnet.common import build_resnet, iter_mb_idxs, load_data

def build_funcs(resnet, params, update_rule, **update_kwargs):

    print("Building training functions")
    x = T.ftensor4('x')
    y = T.imatrix('y')

    prob = L.get_output(resnet['prob'], x, deterministic=False)
    loss = T.nnet.categorical_crossentropy(prob, y.flatten()).mean()

    grad_updates, param_updates, grad_shared = \
        update_rule(loss, params, **update_kwargs)
    # make a function to compute and store the worker's raw gradient
    f_grad_shared = synk.function(inputs=[x, y],
                                  outputs=loss,  # (assumes this is an avg)
                                  updates=grad_updates)
    # make a function to update worker's parameters using stored gradient
    f_param_update = synk.function(inputs=[], updates=param_updates)

    def f_train_minibatch(x_data, y_data, batch):
        # compute worker gradient; average across GPUs; update worker params
        # (alternatively, could update parameters only in master, then broadcast,
        # but that costs more communication)
        train_loss = f_grad_shared(x_data, y_data, batch=batch)
        synk.all_reduce(grad_shared, op="avg")  # (assumes loss is an avg)
        f_param_update()
        return train_loss

    print("Building validation / test function")
    v_prob = L.get_output(resnet['prob'], x, deterministic=True)
    v_loss = T.nnet.categorical_crossentropy(v_prob, y.flatten()).mean()
    v_mc = T.mean(T.neq(T.argmax(v_prob, axis=1), y.flatten()))
    f_predict = synk.function(inputs=[x, y], outputs=[v_loss, v_mc])

    return f_train_minibatch, f_predict


def train_resnet(
        batch_size=32,  # batch size on each GPU
        validFreq=1,
        learning_rate=1e-3,
        update_rule=updates.nesterov_momentum,
        n_epoch=3,
        n_gpu=None,  # later get this from synk.fork
        **update_kwargs):

    n_gpu = synk.fork(n_gpu)  # (n_gpu==None will use all)

    t_0 = time.time()
    print("Loading data (pretend)")
    train, valid, test = load_data()

    x_train, y_train = [synk.data(d) for d in train]
    x_valid, y_valid = [synk.data(d) for d in valid]
    x_test, y_test = [synk.data(d) for d in test]

    full_mb_size = batch_size * n_gpu
    lr = learning_rate * n_gpu  # (one technique for larger minibatches)
    num_valid_slices = len(x_valid) // n_gpu // batch_size
    print("Will compute validation using {} slices".format(num_valid_slices))

    print("Building model")
    resnet = build_resnet()
    params = L.get_all_params(resnet.values(), trainable=True)

    f_train_minibatch, f_predict = \
        build_funcs(resnet, params, update_rule, lr=lr, **update_kwargs)

    synk.distribute()
    synk.broadcast(params)  # (ensure all GPUs have same values)

    t_last = t_1 = time.time()
    print("Total setup time: {:,.1f} s".format(t_1 - t_0))
    print("Starting training")

    for ep in range(n_epoch):
        train_loss = 0.
        i = 0
        for mb_idxs in iter_mb_idxs(full_mb_size, len(x_train), shuffle=True):
            train_loss += f_train_minibatch(x_train, y_train, batch=mb_idxs)
            i += 1
        train_loss /= i

        print("\nEpoch: ", ep)
        print("Training Loss: {:.3f}".format(train_loss))

        if ep % validFreq == 0:
            valid_loss, valid_mc = f_predict(x_valid, y_valid, num_slices=num_valid_slices)
            print("Validation Loss: {:3f},   Accuracy: {:3f}".format(float(valid_loss), float(1 - valid_mc)))

        t_2 = time.time()
        print("(epoch total time: {:,.1f} s)".format(t_2 - t_last))
        t_last = t_2
    print("\nTotal training time: {:,.1f} s".format(t_last - t_1))


if __name__ == "__main__":
    train_resnet()
