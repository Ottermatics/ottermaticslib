"""A System is a Configuration that orchestrates dataflow between components, as well as solving systems of equations in the presense of limits, as well as formatting results of each Component into reporting ready dataframe. System's solver behavior is inspired by NASA's numerical propulsion simulation system (NPSS) to solve systems of inequalities in complex systems.

Component or other subsystems are added to a System class with `SLOTs`:
    ```
    class CustomSystem(System):
        slot_name = System.defineSLOT(Component,ComponentSubclass,System)
    ```

Component's data flow is established via `SIGNALS` that are defined:
    ```
    class CustomSystem(System):
        signal_name = SIGNAL.define(source_attr_or_property, target_attr)
        control_signal = SIGNAL.define(source_attr_or_property, target_attr,control_with='system.attr or slot.attr`)
    ```
    - source_attr: can reference a locally defined slot attribute (a la attr's fields) or any locally defined slot system property
    - target_attr: must be a locally defined slot attribute or system attribute.

update description to include solver

A system calculates its state upon calling `System.run()`. This executes `pre_execute()` first which will directly update any attributes based on their `SIGNAL` definition between `SLOT` components. Once convergence is reached target_attr's are updated in `post_execute()` for cyclic SIGNALS.

If the system encounters a subsystem in its solver routine, the subsystem is evaluated() and its results used as static in that iteration,ie it isn't included in the system level dependents if cyclic references are found.


The solver uses the root or cobla scipy optimizer results on quick references to internal component references. Upon solving the system

SIGNALS can be limited with constrains via `min or max` values on `NumericProperty` which can be numeric values (int or float) or functions taking one argument of the component it is defined on. Additionally signals may take arguments of `min` or `max` which are numeric values or callbacks which take the system instance as an argument.
"""
import attrs

from ottermatics.properties import *
from ottermatics.logging import LoggingMixin
from ottermatics.configuration import Configuration, otterize
from ottermatics.tabulation import TabulationMixin
from ottermatics.solver import SolverMixin, SOLVER, TRANSIENT
from ottermatics.plotting import PlottingMixin

import copy
import collections
import typing
import numpy


# make a module logger
class SystemsLog(LoggingMixin):
    pass


log = SystemsLog()


@otterize
class System(TabulationMixin, SolverMixin, PlottingMixin):
    """A system defines SLOTS for Components, and data flow between them using SIGNALS

    The system records all attribues to its subcomponents via system_references with scoped keys to references to set or get attributes, as well as observe system properties. These are cached upon first access in an instance.

    The table is made up of these system references, allowing low overhead recording of systems with many variables.

    When solving by default the run(revert=True) call will revert the system state to what it was before the system began.
    """

    _anything_changed_ = False

    parent: typing.Union['Component','System'] = attrs.field(default=None)

    # Properties!
    @system_property
    def converged(self) -> int:
        return int(self._converged)

    @system_property
    def run_id(self) -> int:
        return self._run_id

    @classmethod
    def subclasses(cls, out=None):
        """
        return all subclasses of components, including their subclasses
        :param out: out is to pass when the middle of a recursive operation, do not use it!
        """

        # init the set by default
        if out is None:
            out = set()

        for cls in cls.__subclasses__():
            out.add(cls)
            cls.subclasses(out)

        return out

    def clone(self):
        """returns a clone of this system, often used to iterate the system without affecting the input values at the last convergence step."""
        return copy.deepcopy(self)

    @property
    def identity(self):
        return f"{self.name}_{self._run_id}"

    @property
    def _anything_changed(self):
        """looks at internal components as well as flag for anything chagned."""
        if self._anything_changed_:
            return True
        elif any([c.anything_changed for k, c in self.comp_references.items()]):
            return True
        return False

    @_anything_changed.setter
    def _anything_changed(self, inpt):
        """allows default functionality with new property system"""
        self._anything_changed_ = inpt

    #@instance_cached
    @property
    def comp_references(self):
        """A cached set of recursive references to any slot component"""
        out = {}
        for key, lvl, comp in self.go_through_configurations(parent_level=1):
            if not isinstance(comp, TabulationMixin):
                continue
            out[key] = comp
        return out

    #@property
    def system_references(self,recache=False):
        """gather a list of references to attributes and"""
        if recache == False and hasattr(self,'_prv_system_references'):
            return self._prv_system_references
        
        out = self.internal_references(recache)
        tatr = out["attributes"]
        tprp = out["properties"]

        # component iternals
        for key, comp in self.comp_references.items():
            sout = comp.internal_references(recache)
            satr = sout["attributes"]
            sprp = sout["properties"]

            # Fill in
            for k, v in satr.items():
                tatr[f"{key}.{k}"] = v

            for k, v in sprp.items():
                tprp[f"{key}.{k}"] = v
        
        self._prv_system_references = out
        return out

    @instance_cached
    def all_references(self) -> dict:
        out = {}
        sysref = self.system_references()
        out.update(**sysref["attributes"])
        out.update(**sysref["properties"])
        return out

    @solver_cached
    def data_dict(self):
        """records properties from the entire system, via system references cache"""
        out = collections.OrderedDict()
        sref = self.system_references()
        for k, v in sref["attributes"].items():
            val = v.value()
            if isinstance(val, TABLE_TYPES):
                out[k] = val
            else:
                out[k] = numpy.nan
        for k, v in sref["properties"].items():
            val = v.value()
            if isinstance(val, TABLE_TYPES):
                out[k] = val
            else:
                out[k] = numpy.nan
        return out

    @property
    def system_state(self):
        """records all attributes"""
        out = collections.OrderedDict()
        sref = self.system_references()
        for k, v in sref["attributes"].items():
            out[k] = v.value()
        self.debug(f"recording system state: {out}")
        return out

    def set_system_state(self, ignore=None, **kwargs):
        """accepts parital input scoped from system references"""
        sref = self.system_references()
        satr = sref["attributes"]
        self.debug(f"setting system state: {kwargs}")
        for k, v in kwargs.items():
            if ignore and k in ignore:
                continue
            if k not in satr:
                self.debug(f'skipping {k} not in attributes')
                continue
            ref = satr[k]
            ref.set_value(v)
