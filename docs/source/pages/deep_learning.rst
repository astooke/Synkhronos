
Deep Learning Examples
======================

Two deep learning examples are provided in the ``Synkhronos/demos/`` directory, one using the MNIST dataset and the other training a ResNet-50 model.  Both build models using Lasagne, and this portion of the code remains unchanged.  The main change is the construction of the training function.

Update Rules
------------

All first-order update rules can be computed using multiple devices: 1) workers compute raw gradients on their local data, 2) all-reduce the raw gradient, 3) workers update parameters using the combined gradient.  The tasks of computing the gradient and updating the parameters are split into two Theano (Synkhronos) functions.  Lasagne's update rules have been adapted in this fashion in ``Synkhronos/extensions/updates.py``.  The MNIST and ResNet examples show how to use these updates, and a similar tool for making a single callable training function is provided in ``Synkhronos/extensions/train.py``

One speed enhancement for the above scheme is to write the gradients of all layers in a network into one flattened vector.  Calling NCCL collectives once on a large variable is faster than making many calls over smaller variables.  This flattening pattern is built in to the Synkhronos update rules, which automatically reshape the gradients in the parameter updates.

Data Management
---------------

A more subtle difference in the code is the data management.  All data must first be written to ``synkhronos.Data`` objects, to be made available to all worker processes.  Thereafter, the entire data objects can be passed to the training function, with the kwarg ``batch`` used to select the indexes to use.  It is possible to pass in a list of randomized indexes, and each process will build its own input data from its assigned subset of these indexes.  This is not only convenient for shuffling data, but also more efficient.  In parallel, each worker process will perform its own memory copy inherent in excerpting a list of indexes from a numpy array.  This is the pattern used in ``lasagne_mnist/train_mnist_cpu_data.py`` and ``resnet/train_resnet.py``.

It is also possible to scatter a data set to GPU memories in Theano shared variables.  The kwarg ``batch_s`` selects the indexes of these variables to use in a function call.  This can be either one list (slice), to be used in all workers, or a list of lists (slices), one for each.  All Theano shared variables to be subject to ``batch_s`` or slicing must be declared in function creation under the kwarg ``sliceable_shareds``.  The ``lasagne_mnist/train_mnist_gpu_data.py`` demo follows this pattern.  Scaling and overall speed should improve.  

Note that the ``batch`` and ``batch_s`` kwargs work differently, because ``batch`` applies before scattering, and ``batch_s`` applies after scattering.  See the basic demos for examples.

Easy Validation and Test
------------------------

Validation and test functions can be performed conveniently by passing the entire validation or test set to the function call.  Use the ``num_slices`` kwarg to automatically accumulate the results over multiple calls to the underlying Theano function, to avoid out-of-memory errors.

.. _lasagne_import:

Importing Lasagne & GpuArray
----------------------------

Use the Theano flags ``device=cpu,force_device=True`` when placing ``import lasagne`` or ``import theano.gpuarray`` in file headers.

The reason this is needed is that it is not possible to fork after initializing a CUDA context and then use GPU functions in a sub-processes.  Without the ``force_device=True`` flag, importing ``theano.gpuarray`` creates a CUDA context.  ``import theano.gpuarray`` is called within ``import lasagne``.