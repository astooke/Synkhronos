

from .master import function, data
from .master import broadcast, gather, reduce, all_reduce, all_gather
from .master import scatter
from .master import fork, distribute, close
from .util import make_slices


import sys
sys.setrecursionlimit(50000)  # (workaround for pickling some functions)
del sys
