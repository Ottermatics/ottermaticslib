from engforge.attributes import ATTR_BASE, AttributeInstance

import attrs, attr, uuid


# Instance & Attribute definition for integration vars
# Solver minimizes residual by changing independents
class IntegratorInstance(AttributeInstance):
    """A decoupled signal instance to perform operations on a system instance"""

    system: "System"

    # compiled info
    var_ref: "Ref"
    rate_ref: "Ref"

    __slots__ = ["system", "solver", "var_ref", "rate_ref"]

    def __init__(self, solver: "Time", system: "System") -> None:
        self.class_attr = self.solver = solver
        self.system = system
        self.compile()

    def compile(self):
        self.var_ref = self.system.locate_ref(self.solver.var)
        self.rate_ref = self.system.locate_ref(self.solver.rate)
        self.system.debug(f"integrating {self.var_ref} with {self.rate_ref}")

    def as_ref_dict(self):
        return {"rate": self.rate_ref, "var": self.var_ref}

    @property
    def rates(self):
        return {self.name: self.class_attr.rate}

    @property
    def integrated(self):
        return {self.name: self.class_attr.var}

    @property
    def var(self):
        return self.class_attr.var

    @property
    def rate(self):
        return self.class_attr.rate

    @property
    def current_rate(self):
        return self.rate_ref.value(self.system, self.system.last_context)

    @property
    def constraint_refs(self):
        if self.solver.slvtype in ["eq", "ineq"]:
            return self.const_f
        return None

    @property
    def constraints(self):
        if hasattr(self, "_constraints"):
            return self._constraints

        return self.solver.constraints

        # return [{'type':self.slvtype,'var':self.solver.lhs,'value':self.const_f}]
        # if self.solver.slvtype in ['var','obj']:
        #    return self.solver.constraints
        # else:
        # just me

    @property
    def slvtype(self):
        return self.solver.slvtype

    @property
    def combos(self):
        return self.solver.combos

    @property
    def normalize(self):
        return self.solver.normalize

    @property
    def active(self):
        if hasattr(self, "_active"):
            return self._active


# TODO: depriciate modes and update for dynamicmixin strategies
# TODO: add Time.add_profile(parm, time:parameter_values, combos,active) to add a transient profile to be run on the system that is selectable by the user


class Time(ATTR_BASE):
    """Transient is a base class for integrators over time"""

    mode: str
    var: str
    rate: str
    constraints: list
    allow_constraint_override: bool = True
    instance_class = IntegratorInstance

    @classmethod
    def integrate(
        cls,
        var: str,
        rate: "system_property",
        mode: str = "euler",
        active=True,
        combos="default",
    ):
        """Defines an ODE like integrator that will be integrated over time with the defined integration rule.

        Input should be of strings to look up the particular property or field
        """
        # Create A New Signals Class
        new_name = f"TRANSIENT_{mode}_{var}_{rate}".replace(".", "_")
        # new_name = new_name
        new_dict = dict(
            mode=mode,
            name=new_name,
            var=var,
            rate=rate,
            active=active,
            constraints=[],  # TODO: parse kwargs for limits
            combos=cls.process_combos(combos, add_combos=f"time,{var}"),
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

        var_type = instance.locate(cls.var)
        if var_type is None:
            raise Exception(f"var not found: {cls.var}")
        assert isinstance(
            var_type, attrs.Attribute
        ), f"bad var {cls.var} not attribute: {var_type}"
        assert var_type.type in (
            int,
            float,
        ), f"bad var {cls.var} not numeric"

        driv_type = instance.locate(cls.rate)
        if driv_type is None:
            raise Exception(f"rate not found: {cls.rate}")
        assert isinstance(
            driv_type, (system_property, attrs.Attribute)
        ), f"bad rate {cls.rate} type: {driv_type}"
        if isinstance(driv_type, system_property):
            assert driv_type.return_type in (
                int,
                float,
            ), f"bad var {cls.rate} not numeric"

        # else: attributes are not checked, youre in command

    @classmethod
    def add_var_constraint(cls, value, kind="min", **kwargs):
        """adds a `type` constraint to the solver. If value is numeric it is used as a bound with `scipy` optimize.

        :param type: str, must be either min or max with var value comparison, or with a function additionally eq,ineq (same as max(0)) can be used
        :value: either a numeric (int,float), or a function, f(system)
        """
        assert cls is not Time, f"must set constraint on Time Attribute"
        # assert not cls.constraint_exists(type=kind,var=var), f"constraint already exists!"
        assert isinstance(value, (int, float)) or callable(
            value
        ), f"only int,float or callables allow. Callables must take system as argument"

        var = cls.var
        assert var is not None, "must provide var on non-var solvers"
        assert kind in ("min", "max")
        combo_dflt = "default,lim"
        cmbs = kwargs.get("combos", "")
        combos = cls.process_combos(cmbs, combo_dflt, combo_dflt)
        if isinstance(combos, str):
            combos = combos.split(",")
        active = kwargs.get("active", True)
        const = {
            "type": kind,
            "value": value,
            "var": var,
            "active": active,
            "combos": combos,
            "combo_var": cls.name,
        }
        # print(const,cls.__dict__)

        cinx = cls.constraint_exists(type=kind, var=var)
        inix = cinx is not None
        if cls.allow_constraint_override and inix:
            # print(f'replacing constraint {cinx} with {kind} {value} {var}')
            constraint = cls.constraints[cinx]
            constraint.update(const)
        elif not cls.allow_constraint_override and inix:
            cnx = cls.constraints[cinx]
            raise Exception(f"constraint already exists! {cnx}")
        else:
            cls.constraints.append(const)

    @classmethod
    def constraint_exists(cls, **kw):
        """check constraints on the system, return its index position if found, else None."""
        for i, c in enumerate(cls.constraints):
            if all([c[k] == v for k, v in kw.items()]):
                return i
        return None


# Support Previous API
TRANSIENT = Time
