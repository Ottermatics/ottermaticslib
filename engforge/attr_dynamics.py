from engforge.attributes import ATTR_BASE, AttributeInstance

import attrs, attr, uuid


# Instance & Attribute definition for integration parameters
# Solver minimizes residual by changing independents
class IntegratorInstance(AttributeInstance):
    """A decoupled signal instance to perform operations on a system instance"""

    system: "System"
    transient: "Time"

    # compiled info
    parameter: "Ref"
    derivative: "Ref"

    __slots__ = ["system", "solver", "parameter", "derivative"]

    # TODO: add forward implicit solver

    def __init__(self, solver: "Time", system: "System") -> None:
        self.solver = solver
        self.system = system
        self.compile()

    def compile(self):
        self.parameter = self.system.locate_ref(self.solver.parameter)
        self.derivative = self.system.locate_ref(self.solver.derivative)
        self.system.info(f"integrating {self.parameter} with {self.derivative}")

    def as_ref_dict(self):
        return {'rate':self.derivative,'parm':self.parameter}


#TODO: depriciate modes and update for dynamicmixin strategies
class Time(ATTR_BASE):
    """Transient is a base class for integrators over time"""

    mode: str
    parameter: str
    derivative: str
    instance_class = IntegratorInstance

    @classmethod
    def integrate(
        cls,
        parameter: "attrs.Attribute",
        derivative: "system_property",
        mode: str = "euler",
    ):
        """Defines an ODE like integrator that will be integrated over time with the defined integration rule.

        Input should be of strings to look up the particular property or field
        """
        # Create A New Signals Class
        new_name = f"TRANSIENT_{mode}_{parameter}_{derivative}".replace(
            ".", "_"
        )
        new_name = new_name + "_" + str(uuid.uuid4()).replace("-", "")[0:16]
        new_dict = dict(
            mode=mode,
            name=new_name,
            parameter=parameter,
            derivative=derivative,
            # type=cls,
        )
        return cls._setup_cls(new_name, new_dict)

    # make define the same as integrate
    # @classmethod
    # def subcls_compile(cls,**kwargs):
    #     cls.define = cls.integrate

    @classmethod
    def class_validate(cls, instance, **kwargs):
        from engforge.properties import system_property
        from engforge.solver import SolveableMixin

        system = cls.config_cls
        assert issubclass(system, SolveableMixin), f"must be a solveable system"

        parm_type = instance.locate(cls.parameter)
        if parm_type is None:
            raise Exception(f"parameter not found: {cls.parameter}")
        assert isinstance(
            parm_type, attrs.Attribute
        ), f"bad parm {cls.parameter} not attribute: {parm_type}"
        assert parm_type.type in (
            int,
            float,
        ), f"bad parm {cls.parameter} not numeric"

        driv_type = instance.locate(cls.derivative)
        if driv_type is None:
            raise Exception(f"derivative not found: {cls.derivative}")
        assert isinstance(
            driv_type, (system_property, attrs.Attribute)
        ), f"bad derivative {cls.derivative} type: {driv_type}"
        if isinstance(driv_type, system_property):
            assert driv_type.return_type in (
                int,
                float,
            ), f"bad parm {cls.derivative} not numeric"

        # else: attributes are not checked, youre in command


# Support Previous API
TRANSIENT = Time
