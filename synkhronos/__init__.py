
from .function_builder import function, distribute
from .data_builder import data
from .collectives import *
from .forking import fork, close
from .gpu_utils import get_n_gpu

# workaround for pickling some functions
import sys
min_recursion_limit = 50000
if sys.getrecursionlimit() < min_recursion_limit:
    sys.setrecursionlimit(min_recursion_limit)
del sys, min_recursion_limit

# clean up dir
del function_builder, data_builder, collectives, forking, gpu_utils
del comm, data_module, exct, function_module, reducers, scatterer
del shmemarray, synchronize, worker