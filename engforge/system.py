"""A System is a Configuration that orchestrates dataflow between components, as well as solving systems of equations in the presense of limits, as well as formatting results of each Component into reporting ready dataframe. System's solver behavior is inspired by NASA's numerical propulsion simulation system (NPSS) to solve systems of inequalities in complex systems.

Component or other subsystems are added to a System class with `Slots`:
    ```
    class CustomSystem(System):
        slot_name = Slot.define(Component,ComponentSubclass,System)
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

A system calculates its state upon calling `System.run()`. This executes `pre_execute()` first which will directly update any attributes based on their `SIGNAL` definition between `Slot` components. Once convergence is reached target_attr's are updated in `post_execute()` for cyclic SIGNALS.

If the system encounters a subsystem in its solver routine, the subsystem is evald() and its results used as static in that iteration,ie it isn't included in the system level dependents if cyclic references are found.


The solver uses the root or cobla scipy optimizer results on quick references to internal component references. Upon solving the system

SIGNALS can be limited with constrains via `min or max` values on `NumericProperty` which can be numeric values (int or float) or functions taking one argument of the component it is defined on. Additionally signals may take arguments of `min` or `max` which are numeric values or callbacks which take the system instance as an argument.
"""
import attrs

from engforge.properties import *
from engforge.logging import LoggingMixin
from engforge.configuration import Configuration, forge
from engforge.components import SolveableInterface
from engforge.solver import SolverMixin
from engforge.attr_plotting import PlottingMixin
from engforge.dynamics import GlobalDynamics

import copy
import collections
import typing
import numpy


# make a module logger
class SystemsLog(LoggingMixin):
    pass


log = SystemsLog()

#NOTE: solver must come before solvable interface since it overrides certain methods
@forge
class System(SolverMixin, SolveableInterface, PlottingMixin,GlobalDynamics):
    """A system defines SlotS for Components, and data flow between them using SIGNALS

    The system records all attribues to its subcomponents via system_references with scoped keys to references to set or get attributes, as well as observe system properties. These are cached upon first access in an instance.

    The table is made up of these system references, allowing low overhead recording of systems with many variables.

    When solving by default the run(revert=True) call will revert the system state to what it was before the system began.
    """
    
    #default to nothing
    dynamic_input_vars: list = attrs.field(factory=list)
    dynamic_state_vars: list = attrs.field(factory=list)
    dynamic_output_vars: list = attrs.field(factory=list)
    

    _anything_changed_ = True
    _solver_override: bool = False #this comp will run with run_internal_systems when True, otherwise it resolves to global solver behavior, also prevents the solver from reaching into this system


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
        if self._run_id:
            return f"{self.name}_{self._run_id}"
        else:
            return f"{self.name}"

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
