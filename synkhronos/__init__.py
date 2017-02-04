

from .master import function
from .master import broadcast, gather, reduce, all_reduce, all_gather
from .master import scatter
from .master import fork, distribute, close

import sys
sys.setrecursionlimit(50000)  # (workaround for pickling functions)

del sys
