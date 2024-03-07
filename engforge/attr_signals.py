"""This module defines the slot attrs attribute to define the update behavior of a component or between components in an analysis"""
"""Signals define data flow in the solver system. These updates can happen before or after a solver execution as defined in SolverMixin

"""


import attrs, uuid
from engforge.attributes import ATTR_BASE, AttributeInstance, DEFAULT_COMBO
from engforge.attr_slots import SLOT_TYPES

VALID_MODES = ["pre", "post", "both"]


class SignalInstance(AttributeInstance):
    """A decoupled signal instance to perform operations on a system instance"""

    system: "System"
    signal: "SIGNAL"

    # compiled info
    target: "Ref"
    source: "Ref"
    class_attr: "SIGNAL"

    def __init__(self, signal, system) -> None:
        self.class_attr = self.signal = signal
        self.system = system
        self.compile()

    def compile(self, **kwargs):
        self.source = self.system.locate_ref(self.signal.source)
        self.target = self.system.locate_ref(self.signal.target)
        self.system.debug(f"SIGNAL|setting {self.target} with {self.source}")

    def apply(self):
        """sets `target` from `source`"""
        val = self.source.value()
        if self.system.log_level < 15:
            self.system.info(
                f"SIGNAL|applying {self.source}|{val} to {self.target}"
            )
        self.target.set_value(val)

    @property
    def mode(self) -> str:
        return self.signal.mode
    
    def as_ref_dict(self)->dict:
        return dict(
            target=self.target,
            source=self.source,
        )
    
    def get_alias(self,path):
        return path.split('.')[-1]    


class Signal(ATTR_BASE):
    """A base class that handles initalization in the attrs meta class scheme by ultimately createing a SignalInstance"""

    name: str
    mode: str
    target: str
    source: str
    config_cls: "System"
    attr_prefix = "SIGNAL"
    instance_class = SignalInstance

    @classmethod
    def define(cls, target: str, source: str, mode="pre",**kw):
        """taking a component or system class as possible input valid input is later validated as an instance of that class or subclass"""
        assert mode in VALID_MODES, f"invalid mode: {mode}"

        active = kw.get("active", True)
        combos = kw.get("combos", 'default')

        # Create A New Signals Class
        new_name = f"SIGNAL_{mode}_{source}_to_{target}".replace(".", "_")
        new_name = new_name + "_" + str(uuid.uuid4()).replace("-", "")[0:16]
        new_dict = dict(
            name=new_name,
            mode=mode,
            target=target,
            source=source,
            active=active,
            combos=combos,
            default_options=cls.default_options.copy(),
        )
        new_slot = type(new_name, (SIGNAL,), new_dict)
        new_slot.default_options["default"] = new_slot.make_factory()
        new_slot.default_options["validator"] = new_slot.configure_instance
        new_sig = cls._setup_cls(new_name, new_dict)
        return new_sig

    # FIXME: move to
    @classmethod
    def class_validate(cls, instance, **kwargs):
        from engforge.properties import system_property

        system = cls.config_cls

        parm_type = system.locate(cls.target)
        if parm_type is None:
            raise Exception(f"target not found: {cls.target}")
        assert isinstance(
            parm_type, attrs.Attribute
        ), f"bad parm {cls.target} not attribute: {parm_type}"

        driv_type = system.locate(cls.source)
        if driv_type is None:
            raise Exception(f"source not found: {cls.source}")


# Support Previous API
SIGNAL = Signal
