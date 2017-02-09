Introductory Example
====================

.. literalinclude:: /examples/example_0.py

Which returns, on an example, dual-GPU machine:

.. literalinclude:: /examples/example_0.txt
    :language: text

The program flow is:

1. Import Theano and Sykhronos in file headers as desired.
2. Call ``synkhronos.fork()``.
3. Build Theano variables and graphs.
4. Build functions through Synkhronos instead of Theano.
5. Call ``synkhronos.distribute()``.
6. Run remainder of program as normal.

In More Detail
--------------

Call ``synkhronos.fork()`` to fork a python subprocess for each additional GPU in the computer.  Theano must be initialized on CPU-only before this point.  During ``fork()``, a single GPU is initialized in the master process, and all Theano variables thereafter are built in reference to it.  Inputs to a Synkhronos function can be either scattered (by default, ``x`` in the example), or broadcast (``y`` in the example).  Scattered inputs are split evenly among all workers along the `0-th` dimension.  An input set small enough to run on a single GPU can be computed using a synkhronos function's ``as_theano()`` method, which is equivalent to calling a normal Theano function and executes only in the master process and device.  The user is free to create Theano functions at any time to run only in the master process, including functions influencing Theano shared variables present in Synkhronos functions.

By default, outputs are reduced and averaged, so the assertions pass.  Outputs may be gathered instead, or reduced by other operations: sum, product, max, or min.  Currently, all output collections are executed using NVIDIA's NCCL via PyGPU.


Distribute
----------

After all functions are constructed, calling ``synkhronos.distribute()`` pickles all functions (and their shared variable data) in the master and unpickles them in all workers.  (This may take a few moments.)


Pickling all functions together preserves correspondences among variables used in multiple functions.  Workers register the unpickled functions in the same fashion as they were registered in the master, giving all processes the same mapping of variables for efficient use of memory.

Currently, ``distribute()`` can only be called once.  In the future, this function will be callable unlimited times and will be executed lazily and automatically when functions are called.  It will remain available to optionally be called manually.
