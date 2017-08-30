Introductory Example
====================

.. literalinclude:: /examples/example_0.py

Which returns, on a dual-GPU machine:

.. literalinclude:: /examples/example_0.txt
    :language: text

The program flow is:

1. Call ``synkhronos.fork()``.
2. Build Theano variables and graphs.
3. Build functions through Synkhronos instead of Theano.
4. Call ``synkhronos.distribute()``.
5. Manage input data / run program with functions.

In More Detail
--------------

Import Theano in CPU-mode, and ``fork()`` will initialize the master GPU in the main process and additional GPUs in other processes.  All Theano variables thereafter are built in the master, as in single-GPU programs.  ``distribute()`` replicates all functions, and their variables, in the additional processes and their GPUs.

A function's ``inputs`` will be scattered by splitting evenly along the 0-th dimension.  In this example, data parallelism applies across the 0-th dimensions of the variable ``x``.  A function's ``bcast_inputs`` are broadcast and used wholly in all workers, as the variable ``y`` in the example.

All explicit inputs to functions must be of type ``synkhronos.Data``, rather than numpy arrays.  The underlying memory of these objects is in OS shared memory, so all processes have access to it.  They present an interface similar to numpy arrays, demonstrated later.

The Synkhronos function is computed simultaneously on all GPUs, including the master.  By default, outputs are reduced and averaged, so the comparison to the single-GPU Theano function result passes.  Other operations are possible: `sum`, `prod`, `max`, `min`, or ``None`` for no reduction.


Distribute
----------

After all functions are constructed, calling ``distribute()`` pickles all functions (and their shared variable data) in the master and unpickles them in all workers.  This may take a few moments.  Pickling all functions together preserves correspondences among variables used in multiple functions in each worker.  

Currently, ``distribute()`` can only be called once.  In the future it could be automated or made possible to call multiple times.  Synkhronos data objects can be made before or after distributing, but only after forking.
