Installation
============

To view the source code and / or install::

    git clone https://github.com/astooke/Synkhronos
    cd Synkhronos
    pip install .

Third party dependencies include ``Theano`` with its new GPU back-end, ``libgpuarray``, ``nccl`` (v1) for collective GPU communications, ``posix_ipc`` for allocating shared memory, and ``pyzmq`` for other CPU-based communications.

PyPI package (possibly) forthcoming.

Currently Python3 compatible only.

The use of ``posix_ipc`` limits operating system compatibility--Windows is not supported.

.. hint::  Use Theano flags ``device=cpu`` and ``force_device=True`` (see :ref:`lasagne_import`).

.. hint::  Compile-lock contention that slows down multi-GPU initialization can be avoided by modifying ``Theano\theano\gpuarray\dnn.py``.  Where it initializes ``version.v = None``, change to ``version.v = 6020`` or the installed version of cuDNN.
