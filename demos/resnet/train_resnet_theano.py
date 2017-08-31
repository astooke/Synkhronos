
import theano
import theano.tensor as T
import theano.gpuarray
# import lasagne
import lasagne.layers as L
import time

from synkhronos.extensions import updates
from demos.resnet.common import build_resnet, iter_mb_idxs, load_data


def build_training(resnet, params, update_rule, **update_kwargs):

    print("Building training functions")
    x = T.ftensor4('x')
    y = T.imatrix('y')

    prob = L.get_output(resnet['prob'], x, deterministic=False)
    loss = T.nnet.categorical_crossentropy(prob, y.flatten()).mean()

    grad_updates, param_updates, grad_shared = \
        update_rule(loss, params, **update_kwargs)
    # make a function to compute and store the raw gradient
    f_grad_shared = theano.function(inputs=[x, y],
                                    outputs=loss,  # (assumes this is an avg)
                                    updates=grad_updates)
    # make a function to update parameters using stored gradient
    f_param_update = theano.function(inputs=[], updates=param_updates)

    def f_train_minibatch(x_data, y_data):
        train_loss = f_grad_shared(x_data, y_data)
        # No all-reduce here; single-GPU
        f_param_update()
        return train_loss

    print("Building validation / test function")
    v_prob = L.get_output(resnet['prob'], x, deterministic=True)
    v_loss = T.nnet.categorical_crossentropy(v_prob, y.flatten()).mean()
    v_mc = T.mean(T.neq(T.argmax(v_prob, axis=1), y.flatten()))
    f_predict = theano.function(inputs=[x, y], outputs=[v_loss, v_mc])

    return f_train_minibatch, f_predict


def train_resnet(
        batch_size=64,  # batch size on each GPU
        validFreq=1,
        do_valid=False,
        learning_rate=1e-3,
        update_rule=updates.sgd,  # updates.nesterov_momentum,
        n_epoch=3,
        **update_kwargs):

    # Initialize single GPU.
    theano.gpuarray.use("cuda")

    t_0 = time.time()
    print("Loading data (synthetic)")
    train, valid, test = load_data()

    x_train, y_train = train
    x_valid, y_valid = valid
    x_test, y_test = test

    print("Building model")
    resnet = build_resnet()
    params = L.get_all_params(resnet.values(), trainable=True)

    f_train_minibatch, f_predict = \
        build_training(resnet, params, update_rule, lr=learning_rate, **update_kwargs)

    t_last = t_1 = time.time()
    print("Total setup time: {:,.1f} s".format(t_1 - t_0))
    print("Starting training")

    for ep in range(n_epoch):
        train_loss = 0.
        i = 0
        for mb_idxs in iter_mb_idxs(batch_size, len(x_train), shuffle=True):
            train_loss += f_train_minibatch(x_train[mb_idxs], y_train[mb_idxs])
            i += 1
        train_loss /= i

        print("\nEpoch: ", ep)
        print("Training Loss: {:.3f}".format(train_loss))

        if do_valid and ep % validFreq == 0:
            valid_loss = valid_mc = 0.
            i = 0
            for mb_idxs in iter_mb_idxs(batch_size, len(x_valid), shuffle=False):
                mb_loss, mb_mc = f_predict(x_valid[mb_idxs], y_valid[mb_idxs])
                valid_loss += mb_loss
                valid_mc += mb_mc
                i += 1
            valid_loss /= i
            valid_mc /= i
            print("Validation Loss: {:3f},   Accuracy: {:3f}".format(valid_loss, 1 - valid_mc))

        t_2 = time.time()
        print("(epoch total time: {:,.1f} s)".format(t_2 - t_last))
        t_last = t_2
    print("\nTotal training time: {:,.1f} s".format(t_last - t_1))


if __name__ == "__main__":
    train_resnet()
