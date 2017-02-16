Installation
============

To view the source code and / or install::

    git clone https://github.com/astooke/synkhronos
    cd synkhronos
    pip install .

Third party dependencies include ``Theano`` with its new GPU back-end, and ``posix_ipc`` for allocating shared memory.

PyPI package forthcoming.

Currently Python3 compatible only.

The use of ``posix_ipc`` limits operating system compatibility--Windows is not supported.
