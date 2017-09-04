
import numpy as np
import multiprocessing as mp
import theano
try:
    from pygpu.gpuarray import GpuArray
except ImportError:
    pass  # (needed for autodocs to build without pygpu)


def get_n_gpu():
    detected_n_gpu = mp.RawValue('i', 0)
    p = mp.Process(target=n_gpu_subprocess, args=(detected_n_gpu,))
    p.start()
    p.join()
    n_gpu = int(detected_n_gpu.value)
    if n_gpu == -1:
        raise ImportError("Must be able to import pygpu to use GPUs.")
    return n_gpu


def n_gpu_subprocess(mp_n_gpu):
    """
    Call in a subprocess because it prevents future subprocesses from using GPU.
    """
    try:
        from pygpu import gpuarray
        mp_n_gpu.value = gpuarray.count_devices("cuda", 0)
    except ImportError:
        mp_n_gpu.value = -1


def alloc_gpu_arr(dtype, shape):
    # NOTE: can probably do this through pygpu instead
    s = theano.shared(np.empty(*shape, dtype=dtype))
    assert isinstance(s.container.data, GpuArray)
    return s.container.data
