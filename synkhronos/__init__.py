

from synkhronos.master import function
from synkhronos.master import broadcast, gather, reduce, all_reduce, all_gather
from synkhronos.master import scatter
from synkhronos.master import fork, distribute, close


import sys
sys.setrecursionlimit(50000)  # (workaround for pickling functions)
del sys
