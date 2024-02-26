"""solver defines a SolverMixin for use by System.

Additionally the SOLVER attribute is defined to add complex behavior to a system as well as add constraints and transient integration.
"""

import attrs
import uuid
import numpy
import scipy.optimize as scopt
from contextlib import contextmanager
import copy
import datetime

from engforge.attributes import ATTR_BASE,AttributeInstance
from engforge.properties import *

import itertools

INTEGRATION_MODES = ["euler", "trapezoid", "implicit"]
SOLVER_OPTIONS = ["root", "minimize"]


class AttrSolverLog(LoggingMixin):
    pass


log = AttrSolverLog()


# Solver minimizes residual by changing independents
class SolverInstance(AttributeInstance):
    """A decoupled signal instance to perform operations on a system instance"""

    system: "System"
    solver: "SOLVER"
    
    # compiled info
    dependent: "Ref"
    independent: "Ref"

    __slots__ = ["system", "solver", "dependent", "independent"]

    def __init__(self, solver: "SOLVER", system: "System") -> None:
        self.solver = solver
        self.system = system
        self.compile()

    def compile(self):
        self.dependent = self.system.locate_ref(self.solver.dependent)
        self.independent = self.system.locate_ref(self.solver.independent)
        self.system.debug(f"solving {self.dependent} with {self.independent}")


class Solver(ATTR_BASE):
    """solver creates subclasses per solver balance"""

    dependent: str
    independent: str

    constraints: dict

    attr_prefix = 'SOLVER'

    instance_class = SolverInstance

    @classmethod
    def define(
        cls, dependent: "system_property", independent: "attrs.Attribute"=None
    ):
        """Defines a new dependent and independent variable for the system solver. The error term will not necessiarily be satisfied by changing this particular independent"""

        # Create A New Signals Class
        new_name = f"SOLVER_indep_{dependent}_{independent}".replace(".", "_")
        constraints = {"min": None, "max": None}
        new_dict = dict(
            name=new_name,
            dependent=dependent,
            independent=independent,
            constraints=constraints,
            default_options=cls.default_options.copy(),
        )
        new_slot = type(new_name, (SOLVER,), new_dict)
        #new_slot.default_options['default'] = new_slot.make_factory()
#        new_slot.default_options['validator'] = new_slot.configure_instance
        return new_slot

    #TODO: update all code with this
    solve_root = define
    #TODO: add minimize / maximize / objective options
    

    @classmethod
    def class_validate(cls,instance,**kwargs):
        from engforge.properties import system_property

        system = cls.config_cls

        parm_type = system.locate(cls.independent)
        if parm_type is None:
            raise Exception(f"independent not found: {cls.independent}")
        assert isinstance(
            parm_type, attrs.Attribute
        ), f"bad parm {cls.independent} not attribute: {parm_type}"
        assert parm_type.type in (
            int,
            float,
        ), f"bad parm {cls.independent} not numeric"

        driv_type = system.locate(cls.dependent)
        if parm_type is None:
            raise Exception(f"dependent not found: {cls.dependent}")
        assert isinstance(
            driv_type, system_property
        ), f"bad dependent {cls.dependent} type: {driv_type}"
        assert driv_type.return_type in (
            int,
            float,
        ), f"bad parm {cls.dependent} not numeric"

    @classmethod
    def create_instance(cls, system: "System") -> SolverInstance:
        return SolverInstance(cls, system)

    @classmethod
    def add_constraint(cls, type_, value):
        """adds a `type` constraint to the solver. If value is numeric it is used as a bound with `scipy` optimize.

        If value is a function it should be of the form value(Xarray) and will establish an inequality constraint that independent parameter must be:
            1. less than for max
            2. more than for min

        During the evaluation of the limit function system.X should be set, and pre_execute() have already run.

        :param type: str, must be either min or max with independent value comparison, or with a function additionally eq,ineq (same as max(0)) can be used
        :value: either a numeric (int,float), or a function, f(system)
        """
        assert cls is not SOLVER, f"must set constraint on SOLVER subclass"
        assert (
            cls.constraints[type_] is None
        ), f"existing constraint for {type in {cls.__name__}}"
        assert isinstance(value, (int, float)) or callable(
            value
        ), f"only int,float or callables allow. Callables must take system as argument"
        cls.constraints[type_] = value

    #this means the independent parameter is the only thing that can change
    add_input_constraint = add_constraint

    @classmethod
    def add_constraints(cls, **kwargs):
        for k, v in kwargs.items():
            cls.add_constraint(k, v)


#Support Previous SnakeCase
SOLVER = Solver