"""This module defines the slot attrs attribute to ensure the type of component added is correct and to define behavior,defaults and argument passing behavio"""

import attrs
import uuid
import typing as typ

SLOT_TYPES = typ.Union["Component", "System"]


class SLOT(attrs.Attribute):
    """Slot defines a way to accept different components or systems in a system"""

    # These are added on System signals_slots_handler aka attrs field_transformer
    name: str
    accepted: SLOT_TYPES
    on_system: "System"
    none_ok:bool

    @classmethod
    def define(cls, *component_or_systems: SLOT_TYPES,none_ok=False):
        """taking a component or system class as possible input valid input is later validated as an instance of that class or subclass
        
        :param none_ok: will allow no component on that item, oterwise will fail
        """
        from ottermatics.components import Component
        from ottermatics.system import System

        # Format THe Accepted Component Types
        assert (
            len(component_or_systems) == 1
        ), "only one slot allowed, try making a subclass"
        assert all(
            [
                issubclass(c, Component) or issubclass(c, System)
                for c in component_or_systems
            ]
        ), "Not System Or Component Input"

        # FIXME: come up with a better name :)
        new_name = f"SLOT_{str(uuid.uuid4()).replace('-','')[0:16]}"
        new_slot = type(
            new_name,
            (SLOT,),
            dict(name=new_name, accepted=component_or_systems,none_ok=none_ok),
        )
        return new_slot

    # Create a validator function
    @classmethod
    def validate_slot(cls, instance, attribute, value):
        if value in cls.accepted or all([value is None, cls.none_ok]):
            return True

        if any([isinstance(value, a) for a in cls.accepted]):
            return True

        raise ValueError(
            f"value {value} is not an accepted type for slot: {cls.on_system.__name__}.{cls.name}"
        )

    @classmethod
    def configure_for_system(cls, name, system):
        cls.name = name
        cls.on_system = system
