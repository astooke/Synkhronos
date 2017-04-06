
from .function_builder import function, distribute
from .data_builder import data
from .collectives import *
from .master import *
from .util import make_slices

import sys
sys.setrecursionlimit(50000)  # (workaround for pickling some functions)
del sys

del function_builder, data_builder, collectives, master, util  # clean up dir

