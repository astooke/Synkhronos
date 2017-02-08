import ipdb
from timeit import default_timer as timer

import numpy as np
import theano
import theano.tensor as T

import synkhronos as synk


def make_data(input_shape, batch_size):
    x_data = np.random.randn(batch_size, *input_shape).astype('float32')
    y_data = np.random.randint(low=0, high=5, size=batch_size, dtype='int32')
    return x_data, y_data


def build_mlp(input_var=None):
    import lasagne
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. XXIt applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.XX

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # # Apply 20% dropout to the input data:
    # l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # # We'll now add dropout of 50%:
    # l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # # 50% dropout again:
    # l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_cnn(input_var=None):
    import lasagne
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network



def main():

    B_SIZE = 10000
    MID = B_SIZE // 2

    synk.fork()
    import lasagne

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = build_mlp(input_var)
    # network = build_cnn(input_var)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)

    grads = theano.grad(loss, wrt=params)
    flat_grad = T.concatenate(list(map(T.flatten, grads)))

    f_loss = synk.function([input_var, target_var], loss, collect_modes=[None], reduce_ops="sum")
    f_grad = synk.function([input_var, target_var], flat_grad, collect_modes=[None])

    synk.distribute()

    x_data, y_data = make_data([1, 28, 28], B_SIZE)

    loss_1 = f_loss(x_data, y_data)
    grad_1 = f_grad(x_data, y_data)

    x_shmem, y_shmem = f_loss.get_input_shmems()
    x_dat_sh = x_shmem[:B_SIZE]
    y_dat_sh = y_shmem[:B_SIZE]
    x_data_1 = x_data[:MID]
    x_data_2 = x_data[MID:]
    y_data_1 = y_data[:MID]
    y_data_2 = y_data[MID:]

    ITERS = 10
    t0 = timer()
    for _ in range(ITERS):
        loss_i = f_loss.as_theano(x_data_1, y_data_1)
        loss_j = f_loss.as_theano(x_data_2, y_data_2)
    loss_time = timer() - t0
    print("theano loss_time: ", loss_time)

    t0 = timer()
    for _ in range(ITERS):
        grad_i = f_grad.as_theano(x_data_1, y_data_1)
        grad_j = f_grad.as_theano(x_data_2, y_data_2)
    grad_time = timer() - t0
    print("theano grad_time: ", grad_time)


    t0 = timer()
    for _ in range(ITERS):
        loss_i = f_loss(x_dat_sh, y_dat_sh)
    loss_time = timer() - t0
    print("synk shmem loss_time: ", loss_time)

    t0 = timer()
    for _ in range(ITERS):
        grad_i = f_grad(x_dat_sh, y_dat_sh)
    grad_time = timer() - t0
    print("synk shmem grad_time: ", grad_time)

    t0 = timer()
    for _ in range(ITERS):
        loss_i = f_loss(x_data, y_data)
    loss_time = timer() - t0
    print("synk new input loss_time: ", loss_time)

    t0 = timer()
    for _ in range(ITERS):
        grad_i = f_grad(x_data, y_data)
    grad_time = timer() - t0
    print("synk new input grad_time: ", grad_time)

    # ipdb.set_trace()

if __name__ == '__main__':
    main()
