.. synkhronos documentation master file, created by
   sphinx-quickstart on Tue Feb  7 17:51:00 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Synkhronos
==========
A Python Extension for Data Parallelism
---------------------------------------

Synkhronos is a Python tool for accelerating computation of Theano functions whenever data parallelism applies.  For example, when computing an expectation value over a data set, computation can potentially accelerate by `N` times when the data is split over `N` devices.  Careful control over communication among devices is necessary to achieve favorable speeds; this package includes effective communication routines.

Acceleration is possible with multiple GPUs or even in multi-core CPUs, where explicit use of data parallelism outperforms multi-threaded BLAS operations such as in MKL.  The current version supports only multi-GPU use (CPU support forthcoming), and requires use of Theano's new GPU back-end.

The aim of this package is to minimize change in user code while leveraging multiple compute devices.  Construct Theano variables and graphs as usual, and simply build functions through this package rather than directly through Theano.  All parallelism is automated and hidden from view.  The only additional algorithmic consideration is communication over shared variables.  A distinct version of each one exists on each GPU, so collective communications, such as broadcast, all-reduce, and scatter, are provided to manage workers' values.  This must be done manually, as Theano function updates apply only locally within the worker.

Achieving the best speedup may require one implementation change.  Distributing data to the workers is achieved through shared memory (akin to multiprocessing's shared memory); it can be advantageous to manually write data to this memory at an opportune time, rather than waiting for synkhronos to do it internally (examples on the following pages).

This package is for single-node computing; underlying it is ``multiprocessing``, not ``MPI``.

Contents:

.. toctree::
   :maxdepth: 2
   :numbered:

   pages/installation.rst
   pages/intro_example.rst
   pages/functions.rst




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

