Introductory Example
====================

.. literalinclude:: /examples/example_0.py

.. literalinclude:: /examples/example_0.txt

Call ``synkhronos.fork()`` to fork a python subprocess for each additional GPU in the computer.  Theano must be initialized on CPU-only before this point.  During ``fork``, a single GPU is initialized in the master process, and all Theano variables thereafter are built in reference to it.  Inputs to a synkhronos function can be either scattered (by default, ``x`` in the example), or broadcast (``y`` in the example).  Scattered inputs are split evenly among all workers along the `0-th` dimension.  An input set small enough to run on a single GPU can be computed using a synkhronos function's ``as_theano`` method, which is equivalent to calling a normal Theano function and executes only in the master process and device.

After all functions are constructed, calling ``synkhronos.distribute()`` pickles all functions (and their shared variable data) in the master and unpickles them in all workers.  (This may take a few moments.)

By default, outputs are reduced and averaged, so the assertion passes.  Outputs may be gathered instead, or reduced by other operations: sum, product, max, or min.

Pickling all functions together preserves correspondences among variables used in multiple functions.  Workers in-process the unpickled functions in the same fashion as they were registered in the master, giving all processes the same mapping of variables for efficient use of memory.  Currently, ``distribute`` can only be called once.  In the future, this function will be callable unlimited times and will be executed lazily and automatically when functions are called.  It will remain available to optionally be called manually.
