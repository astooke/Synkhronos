
import numpy as np
import theano
import theano.tensor as T
import theano.gpuarray
import lasagne
import time
import sys

from demos.lasagne_mnist.common import load_dataset, build_network

import synkhronos as synk
from synkhronos.extensions import updates


# ############################# Batch iterator ###############################

def iterate_minibatch_indices(data_len, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(data_len)
        np.random.shuffle(indices)
    for start_idx in range(0, data_len - batchsize + 1, batchsize):
        if shuffle:
            batch = indices[start_idx:start_idx + batchsize]
        else:
            batch = slice(start_idx, start_idx + batchsize)
        yield batch


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='mlp', batch_size=500, num_epochs=10):

    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    y_train = y_train.astype("int32")  # (some downstream type error on uint8)
    y_val = y_val.astype("int32")

    # Fork worker processes and initilize GPU before building variables.
    n_gpu = synk.fork()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_network(model, input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)

    grad_updates, param_updates, grad_shared = updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)


    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Make GPU variables to hold the data.
    s_input_train = theano.shared(X_train[:len(X_train) // n_gpu])
    s_target_train = theano.shared(y_train[:len(y_train) // n_gpu])
    s_input_val = theano.shared(X_val[:len(X_val) // n_gpu])
    s_target_val = theano.shared(y_val[:len(y_val) // n_gpu])

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_grad_fn = synk.function(inputs=[], outputs=loss,
                                  givens=[(input_var, s_input_train),
                                          (target_var, s_target_train)],
                                  sliceable_shareds=[s_input_train, s_target_train],
                                  updates=grad_updates)
    train_update_fn = synk.function([], updates=param_updates)
    # train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = synk.function(inputs=[],
                           givens=[(input_var, s_input_val),
                                   (target_var, s_target_val)],
                           sliceable_shareds=[s_input_val, s_target_val],
                           outputs=[test_loss, test_acc])
    # val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Don't bother to put test data on GPU ahead of time.
    test_fn = synk.function([input_var, target_var],
                            outputs=[test_loss, test_acc])

    # After building all functions, give them to workers.
    synk.distribute()

    # Put data into OS shared memory for worker access.
    X_test, y_test = test_fn.build_inputs(X_test, y_test)

    print("Scattering data to GPUs.")
    scatter_vars = [s_input_train, s_target_train, s_input_val, s_target_val]
    scatter_vals = [X_train, y_train, X_val, y_val]
    synk.scatter(scatter_vars, scatter_vals)
    train_worker_len = min(synk.get_lengths(s_target_train))
    worker_batch_size = batch_size // n_gpu

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        # for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
        for batch in iterate_minibatch_indices(train_worker_len, worker_batch_size, shuffle=True):
            train_err += train_grad_fn(batch_s=batch)
            synk.all_reduce(grad_shared)  # (averges)
            train_update_fn()
            train_batches += 1

        # And a full pass over the validation data:
        # val_err = 0
        # val_acc = 0
        # val_batches = 0
        # for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
        #     inputs, targets = batch
        #     err, acc = val_fn(inputs, targets)
        #     val_err += err
        #     val_acc += acc
        #     val_batches += 1
        val_err, val_acc = val_fn(num_slices=4)


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(float(val_err)))
        print("  validation accuracy:\t\t{:.2f} %".format(float(val_acc) * 100))

    # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
    #     inputs, targets = batch
    #     err, acc = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    test_err, test_acc = test_fn(X_test, y_test, num_slices=4)
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(float(test_err)))
    print("  test accuracy:\t\t{:.2f} %".format(float(test_acc) * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("BATCH_SIZE: number of data points to process for each")
        print("            gradient step (default: 500)")
        print("EPOCHS: number of training epochs to perform (default: 10)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['batch_size'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['num_epochs'] = int(sys.argv[3])
        main(**kwargs)
