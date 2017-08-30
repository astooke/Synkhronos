Installation
============

To view the source code and / or install::

    git clone https://github.com/astooke/Synkhronos
    cd Synkhronos
    pip install .

Third party dependencies include ``Theano`` with its new GPU back-end, ``libgpuarray``, ``nccl`` for collective GPU communications, ``posix_ipc`` for allocating shared memory, and ``pyzmq`` for other CPU-based communications.

PyPI package (possibly) forthcoming.

Currently Python3 compatible only.

The use of ``posix_ipc`` limits operating system compatibility--Windows is not supported.
