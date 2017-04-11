
from .function_builder import function, distribute
from .data_builder import data
from .collectives import *
from .forking import fork, close
from .gpu_utils import get_n_gpu
from .util import make_slices


import sys
sys.setrecursionlimit(50000)  # (workaround for pickling some functions)
del sys

# clean up dir
del function_builder, data_builder, collectives, forking, gpu_utils, util

