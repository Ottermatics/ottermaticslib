"""This module defines the slot attrs attribute to ensure the type of component added is correct and to define behavior,defaults and argument passing behavio"""

import attrs
import uuid
import typing as typ

SLOT_TYPES = typ.Union["Component", "System"]
ITERATOR = typ.Union["ComponentIterator"]


class SLOT(attrs.Attribute):
    """Slot defines a way to accept different components or systems in a system"""

    # These are added on System signals_slots_handler aka attrs field_transformer
    name: str
    accepted: SLOT_TYPES
    on_system: "System"
    none_ok: bool
    default_ok = True

    is_iter: bool
    wide: bool  # only for component iterator

    @classmethod
    def define(
        cls, *component_or_systems: SLOT_TYPES, none_ok=False, default_ok=True
    ):
        """taking a component or system class as possible input valid input is later validated as an instance of that class or subclass

        :param none_ok: will allow no component on that item, oterwise will fail
        :param default_ok: will create the slot class with no input if true, which is the behavior by default
        #TODO: add default_args,default_kwargs
        """
        from ottermatics.components import Component
        from ottermatics.component_collections import ComponentIter
        from ottermatics.system import System

        # Format THe Accepted Component Types
        assert (
            len(component_or_systems) == 1
        ), "only one slot allowed, try making a subclass"
        assert not any(
            [issubclass(c, ComponentIter) for c in component_or_systems]
        ), f"`ComponentIter` slot should be defined with `define_iterator` "
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
            dict(
                name=new_name,
                accepted=component_or_systems,
                none_ok=none_ok,
                default_ok=default_ok,
            ),
        )
        return new_slot

    @classmethod
    def define_iterator(
        cls,
        *component_or_systems: ITERATOR,
        none_ok=False,
        default_ok=True,
        wide=True,
    ):
        """taking a type of component iterator, defines an interface that can be 'wide' where all items are executed in the same row on `System.run()`.

        Conversely if `wide` is false the system will loop over each item as if it was included in System.run(). Multiple ComponentIterators with wide=False will result in a `outer join` of the items.

        :param none_ok: will allow no component on that item, otherwise will fail
        :param default_ok: will create the slot class with no input if true, which is the behavior by default
        :param wide: default is true, will determine if wide dataframe format, or outerproduct format when `System.run()` is called
        """
        from ottermatics.components import Component
        from ottermatics.component_collections import ComponentIter
        from ottermatics.system import System

        # Format THe Accepted Component Types
        assert (
            len(component_or_systems) == 1
        ), "only one slot allowed, try making a subclass"
        assert all(
            [issubclass(c, ComponentIter) for c in component_or_systems]
        ), "Not System Or Component Input"

        # FIXME: come up with a better name :)
        new_name = f"SLOTITER_{str(uuid.uuid4()).replace('-','')[0:16]}"
        new_slot = type(
            new_name,
            (SLOT,),
            dict(
                name=new_name,
                accepted=component_or_systems,
                none_ok=none_ok,
                default_ok=default_ok,
                is_iter=True,
                wide=wide,
            ),
        )
        return new_slot

    # Create a validator function
    @classmethod
    def validate_slot(cls, instance, attribute, value):
        from ottermatics.component_collections import ComponentIter

        

        # apply wide behavior to componentiter instance
        if isinstance(value, ComponentIter) and attribute.type.wide == False:
            #print(f'validate {instance} {attribute} {value}')
            value.wide = False

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
