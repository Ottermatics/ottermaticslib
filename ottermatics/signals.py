"""This module defines the slot attrs attribute to define the update behavior of a component or between components in an analysis"""
"""Signals define data flow in the solver system. These updates can happen before or after a solver execution as defined in SolverMixin

"""


import attrs
from engforge.slots import SLOT_TYPES

VALID_MODES = ["pre", "post", "both"]


class SIGNAL(attrs.Attribute):
    """A base class that handles initalization in the attrs meta class scheme by ultimately createing a SignalInstance"""

    name: str
    mode: str
    target: str
    source: str
    on_system: "System"

    @classmethod
    def define(cls, target: str, source: str, mode="pre"):
        """taking a component or system class as possible input valid input is later validated as an instance of that class or subclass"""
        assert mode in VALID_MODES, f"invalid mode: {mode}"

        # Create A New Signals Class
        new_name = f"SIGNAL_{mode}_{source}_to_{target}".replace(".", "_")
        new_dict = dict(
            name=new_name,
            mode=mode,
            target=target,
            source=source,
        )
        new_slot = type(new_name, (SIGNAL,), new_dict)
        return new_slot

    @classmethod
    def configure_for_system(cls, name, system):
        cls.name = name
        cls.on_system = system

    # FIXME: move to
    @classmethod
    def validate_parms(cls):
        from engforge.properties import system_property

        system = cls.on_system

        parm_type = system.locate(cls.target)
        if parm_type is None:
            raise Exception(f"target not found: {cls.target}")
        assert isinstance(
            parm_type, attrs.Attribute
        ), f"bad parm {cls.target} not attribute: {parm_type}"

        driv_type = system.locate(cls.source)
        if parm_type is None:
            raise Exception(f"source not found: {cls.source}")

    @classmethod
    def make_signal_factory(cls):
        return attrs.Factory(cls.signal_factory_func, takes_self=True)

    @classmethod
    def signal_factory_func(cls, instance):
        return SignalInstance(cls, instance)


class SignalInstance:
    """A decoupled signal instance to perform operations on a system instance"""

    system: "System"
    signal: "SIGNAL"

    # compiled info
    target: "Ref"
    source: "Ref"

    __slots__ = ["system", "signal", "target", "source"]

    def __init__(self, signal, system) -> None:
        self.signal = signal
        self.system = system
        self.compile()

    def compile(self):
        self.source = self.system.locate_ref(self.signal.source)
        self.target = self.system.locate_ref(self.signal.target)
        self.system.info(f"setting {self.target} with {self.source}")

    def apply(self):
        """sets `target` from `source`"""
        val = self.source.value()
        if self.system.log_level < 5:
            self.system.debug(f"applying {self.source}|{val} to {self.target}")
        self.target.set_value(val)

    @property
    def mode(self) -> str:
        return self.signal.mode
