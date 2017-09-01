
Theano Shared Variable Example
==============================

This example demonstrates management of Theano shared variables in the master and workers.  First, the setup, again dual-GPU:

.. literalinclude:: /examples/example_1.py
    :lines: 6-19

The resulting output is the correct answer:

.. literalinclude:: /examples/example_1.txt
    :language: text
    :lines: 1-5

Continuing with a reset and a call to the Synkhronos function, we investigate results using ``gather()``, one way to collect all shared values into the master:

.. literalinclude:: /examples/example_1.py
    :lines: 21-32

.. literalinclude:: /examples/example_1.txt
    :language: text
    :lines: 7-19

Lastly, to propagate the result to all workers and observe this effect, call the following:

.. literalinclude:: /examples/example_1.py
    :lines: 34-

.. literalinclude:: /examples/example_1.txt
    :language: text
    :lines: 20-

Notice the use of ``broadcast()`` to set the same values in all GPUs.

Notes on Collectives
~~~~~~~~~~~~~~~~~~~~

Collectives can be called on any Theano shared variable used in a Synkhronos function.  CPU- and GPU-based collectives are available through the same interface.  Results of a GPU collective communication may be returned as a new GPU array in the master, but no collective can create a new array (not associated with a Theano shared variable) in a worker.

Synkhronos provides the averaging reduction operation.  The reduce operation ``avg`` is not present in NCCL; Synkhronos uses ``sum`` and then multiplies by the reciprocal number of GPUs.

Theano Shared Variable Sizes
----------------------------

Beware that the ``nccl`` collectives assume the same shape variable on each GPU, but it is possible to have different shapes in Synkhronos.  In particular, ``gather`` and ``all_gather`` may leave off data or add extra data without raising an exception--in this case use CPU-based gather operations.  See ``demos/demo_3.py`` for more about manipulating GPU-variables in workers.

