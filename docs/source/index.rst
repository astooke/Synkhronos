.. synkhronos documentation master file, created by
   sphinx-quickstart on Tue Feb  7 17:51:00 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



==========
Synkhronos
==========
A Python Extension for Data Parallelism
---------------------------------------

Synkhronos is a Python package for accelerating computation of Theano functions whenever data parallelism applies.  For example, when computing an expectation value over a data set, computation can potentially accelerate by `N` times when the data is split over `N` devices.  Careful control over communication among devices is necessary to achieve favorable speeds; this package includes an effective and straightforward communication framework.

Acceleration is possible with multiple GPUs or even in multi-core CPUs, where the use of multiple Python processes with explicit data parallelism outperforms multi-threaded BLAS operations such as in MKL.  The current version supports only multi-GPU use (CPU support forthcoming), and requires use of Theano's new GPU back-end.  To date, use of multiple processes is still required to ensure full concurrency of GPU-based functions.

The aim of this package is to minimize change in user code while leveraging multiple compute devices.  Theano variables and graphs are constructed as usual.  Then, functions are built through this package rather than directly through Theano.  All parallelism is automated and hidden from view.  Worker processes simply wait for signals from the master, and they only perform the same function the master is performing, and always all at the same time.

The only additional algorithmic consideration is communication over Theano shared variables.  A distinct version of each one exists on each GPU, so collective communications, such as `broadcast`, `all-reduce`, and `scatter`, are provided to manage workers' values.  This must be done explicitly with Synkhronos, as Theano function updates apply only locally within the worker.

Achieving the best speedup usually requires one additional implementation consideration.  Distributing data to the workers is achieved through system shared memory (e.g. Multiprocessing's shared memory).  It can be advantageous to write data to this memory before calling a function, thus altering the scheme for Theano inputs to be more like Theano shared variables.  Examples in the following pages show how to do this.

This package is for single-node computing; underlying it is Multiprocessing, not MPI.

.. hint::  Use Theano flags ``device=cpu`` and ``force_device=True`` (see :ref:`lasagne_import`).

Contents:

.. toctree::
   :maxdepth: 2
   :numbered:

   pages/installation.rst
   pages/intro_example.rst
   pages/theano_shared.rst
   pages/lasagne_mnist.rst
   pages/functions.rst




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

