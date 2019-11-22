from multiprocessing import *
from patterns import *

from deco import *

class AutoProcess(object):
    '''Uses Deco Library To Parallalize Functions using @concurrent decorator'''
        __metaclass__ = Singleton
