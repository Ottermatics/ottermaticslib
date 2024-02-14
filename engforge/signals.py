"""This module defines the slot attrs attribute to define the update behavior of a component or between components in an analysis"""
"""Signals define data flow in the solver system. These updates can happen before or after a solver execution as defined in SolverMixin

"""


import attrs
from engforge.attributes import ATTR_BASE, AttributeInstance
from engforge.slots import SLOT_TYPES

VALID_MODES = ["pre", "post", "both"]


class SignalInstance(AttributeInstance):
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

class SIGNAL(ATTR_BASE):
    """A base class that handles initalization in the attrs meta class scheme by ultimately createing a SignalInstance"""

    name: str
    mode: str
    target: str
    source: str
    config_obj: "System"
    attr_prefix = 'SIGNAL'
    instance_class = SignalInstance

#     @classmethod
#     def define(cls, target: str, source: str, mode="pre"):
#         """taking a component or system class as possible input valid input is later validated as an instance of that class or subclass"""
#         assert mode in VALID_MODES, f"invalid mode: {mode}"
# 
#         # Create A New Signals Class
#         new_name = f"SIGNAL_{mode}_{source}_to_{target}".replace(".", "_")
#         new_dict = dict(
#             name=new_name,
#             mode=mode,
#             target=target,
#             source=source,
#         )
#         new_slot = type(new_name, (SIGNAL,), new_dict)
#         return new_slot

    @classmethod
    def define_validate(cls, **kwargs):
        """A method to validate the kwargs passed to the define method"""
        assert 'target' in kwargs
        assert 'source' in kwargs
        assert 'mode' in kwargs
        mode = kwargs['mode']
        assert mode in VALID_MODES, f"invalid mode: {mode}"

    # FIXME: move to
    @classmethod
    def instance_validate(cls,instance,**kwargs):
        from engforge.properties import system_property

        system = cls.config_obj

        parm_type = system.locate(cls.target)
        if parm_type is None:
            raise Exception(f"target not found: {cls.target}")
        assert isinstance(
            parm_type, attrs.Attribute
        ), f"bad parm {cls.target} not attribute: {parm_type}"

        driv_type = system.locate(cls.source)
        if parm_type is None:
            raise Exception(f"source not found: {cls.source}")



