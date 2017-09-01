
import os

import inspect
import synkhronos
PKL_PATH = inspect.getfile(synkhronos).rsplit("__init__.py")[0] + "pkl/"
PID = str(os.getpid())
PKL_FILE = PKL_PATH + "synk_f_dump_" + PID + ".pkl"
PREFIX = "/synk_" + PID


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def init_cpu(rank):
    raise NotImplementedError
