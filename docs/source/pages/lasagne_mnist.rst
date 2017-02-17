
Lasagne MNIST Example
=====================

This example shows how to modify the Lasagne MNIST example to use Synkhronos.  Both the original and modified files are in the "demos" folder.  It is suggested to familiarize with the original one now; only the key differences are shown below.

Managing Input Shared Memory
----------------------------

The main difference in the modified version is in management of the input data.  One key to good performance is to eliminate extraneous data-copying operations, specifically in regards to the system shared memory used to provide data to the workers.  In this example, the entire data set, including training, validation, and test sets, are written to shared memory once at the beginning of training and thereafter remain available to all workers.  Different minibatches are effectively input during each Synkhronos function call by merely passing the desired indices.

Code Changes
~~~~~~~~~~~~

Populate the CPU shared memory, as if inputing all data into the function at once, inside ``main()``:

.. literalinclude:: /examples/lasagne_mnist.py
    :lines: 313-319

Now to iterate over minibatches, instead of calling the Synkhronos function with data inputs, use the ``batch`` keyword to indicate the data points to use from the existing shared memory store (calling with fresh data at each minibatch will still compute the correct result but will be slower than single GPU computation):

.. literalinclude:: /examples/lasagne_mnist.py
    :lines: 329-332

The ``all_reduce()`` call by default averages the requested Theano shared variables--this is required to keep the model parameters the same on all GPUs after the training function updates the values separately in each.

The iteration generator has been modified to operate over indices only and not the data:

.. literalinclude:: /examples/lasagne_mnist.py
    :lines: 233-242

Notice that the input parameter ``batch`` can be either a slice object or a collection of integers.  In either case, the designated set of indices is divided evenly among the GPUs for processing.  Internally the Theano function is called with normal input arrays formed from these indices used to slice the system shared memory array.

Speed Results
-------------

Running the MNIST example using the CNN option on a two-GPU (GTX-1080) workstation yielded the following average epoch times:

Theano: **0.72** s  ( **1** x)

Synkhronos (no `train_loss` collect, no `all_reduce`): **0.46** s  ( **1.6** x)

Synkhronos (no `train_loss` collect): **0.53** s  ( **1.4** x)

Synkhronos: **0.66** s  ( **1.1** x)

When the training loss output is not collected and the model parameters are never synchronized, the maximum possible speedup is achieved, for reference.  In this case, it is only a factor of **1.6**, well short of **2**.  The more compute-bound the program is, the better the speedup will be.  Note that the size of the minibatch and hence the number of calls per epoch is `not` altered--the same batch is simply scattered.

Synchronizing model parameters at every step results in a mild slow-down to **1.4** x.  Synchronizing on a less frequent schedule would decrease this effect and could be traded against iteration performance.

It is unclear why collection of the training loss output--a scalar--induces such a dramatic slow-down, to only **1.1** x.  A better paradigm would be to build accumulator Theano shared variables for the loss and accuracy and only reduce them once at the conclusion of each epoch.  (In the future, either the cause of this will be determined, hopefully, or CPU-based collection may become an attractive alternative for some outputs.)

Inputs as Theano Shareds -- Scatter
-----------------------------------

Use of multiple GPUs may allow a significant portion of a data set to fit in GPU memory.  In this case, computation may be further accelerated by building function inputs as Theano shared variables.  The ``scatter()`` collective can be used to push a distinct subset of the data onto each GPU.  Management of shared memory in this case can be performed just as with inputs: all data may be placed in system shared memory associated with a Theano shared variable using ``set_shmems()``.  Subsequently passing a slice or list of indices to ``scatter()`` via the ``batch`` keyword will result in that subset being evenly divided up and pushed to GPU memories.

.. _lasagne_import:

Importing Lasagne & GpuArray
----------------------------

Use the Theano flags ``device=cpu,force_device=True`` when placing ``import lasagne`` or ``import theano.gpuarray`` in file headers.

The reason this is needed is that it is not possible to fork after initializing a CUDA context and then use GPU functions in a sub-processes.  Without the ``force_device=True`` flag, importing ``theano.gpuarray`` creates a CUDA context.  ``import theano.gpuarray`` is called within ``import lasagne``.
