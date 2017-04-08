
"""
Constants and utilities used in across master and workers.
"""

import os

import inspect
import synkhronos
PKL_PATH = inspect.getfile(synkhronos).rsplit("__init__.py")[0] + "pkl/"


PID = str(os.getpid())

PRE = "/synk_" + PID


# Where to put functions on their way to workers
PKL_FILE = PKL_PATH + "synk_f_dump_" + PID + ".pkl"





