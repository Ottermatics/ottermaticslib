from contextlib import contextmanager
import attr
import typing

# from engforge.logging import LoggingMixin, log
from engforge.tabulation import TabulationMixin, system_property
from engforge.configuration import forge, Configuration
from engforge.solver import SolveableMixin
from engforge.properties import class_cache
from engforge.dynamics import DynamicsMixin


import os, sys
import inspect

# import pathlib
import random
import matplotlib.pyplot as plt


@forge
class SolveableInterface(Configuration,TabulationMixin,SolveableMixin):
    """common base betwewn solvable and system"""
    parent: typing.Union["Component", "System"] = attr.ib(default=None)
    _last_context: "ProblemExec"

    @property
    def last_context(self):
        """get the last context run, or the parent's"""
        if hasattr(self,"_last_context"):
            #cleanup parent context
            if hasattr(self,'_parent_context') and not self.parent.last_context:
                del self._last_context
                del self._parent_context
            else:
                return self._last_context 
        elif hasattr(self,'parent') and self.parent and (ctx:=self.parent.last_context):
            self._last_context = ctx
            self._parent_context = True
            return ctx
        return None

#NOTE: components / systems not interchangable, systems are like components but are have solver capabilities
#TODO: justify separation of components and systems, and not making system a subclass of component. Otherwise make system a subclass of component
@forge
class Component(SolveableInterface,DynamicsMixin):
    """Component is an Evaluatable configuration with tabulation, and solvable functionality"""
    pass

