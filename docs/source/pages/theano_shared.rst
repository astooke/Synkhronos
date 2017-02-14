
Theano Shared Variable Example
==============================

This example demonstrates management of Theano shared variables in the master and workers.  First, the setup (again dual-GPU):

.. literalinclude:: /examples/example_1.py
    :lines: 6-19

The resulting output is the correct answer:

.. literalinclude:: /examples/example_1.txt
    :language: text
    :lines: 1-5

Continuing with a reset and a call to the Synkhronos function, we investigate results using ``gather()``, which is the way to collect all shared values into the master:

.. literalinclude:: /examples/example_1.py
    :lines: 20-28

.. literalinclude:: /examples/example_1.txt
    :language: text
    :lines: 7-19

Lastly, to propagate the result to all workers and observe this effect, call the following:

.. literalinclude:: /examples/example_1.py
    :lines: 29-

.. literalinclude:: /examples/example_1.txt
    :language: text
    :lines: 20-

Notice the use of ``broadcast()`` to reset the values in the workers according to the master's values.

Notes on Collectives
--------------------

Currently, all Theano shared variable collectives must be called directly, through Synkhronos--no rules for associating such actions with a function exist yet.  Broadcast, reduce, all-reduce, gather, and all-gather operate through NVIDIA's NCCL collectives via PyGPU.  Scatter is a CPU shared-memory operation.  See the function reference page for more details specific to each operation.

In the future, CPU-based analogs of all collectives are planned to be provided.  Beside being necessary for CPU-only usage, these might actually be faster in some uses on some computer architectures.

Averaging
---------

The reduce operation "average" is not present in NCCL.  This is implemented in Synkhronos by a NCCL "sum" reduction and then calling an internally built Theano function to multiply the value (shared variable or function output, while it's still on GPU) by the reciprocal of the number of GPUs.
