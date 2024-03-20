from contextlib import contextmanager
import attr
import typing

# from engforge.logging import LoggingMixin, log
from engforge.tabulation import TabulationMixin, system_property
from engforge.configuration import forge, Configuration
from engforge.solver import SolveableMixin
from engforge.properties import class_cache


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
        if hasattr(self,"_last_context"):
            return self._last_context

#NOTE: components / systems not interchangable, systems are like components but are have solver capabilities
#TODO: justify separation of components and systems, and not making system a subclass of component. Otherwise make system a subclass of component
@forge
class Component(SolveableInterface):
    """Component is an Evaluatable configuration with tabulation, and solvable functionality"""
    pass

