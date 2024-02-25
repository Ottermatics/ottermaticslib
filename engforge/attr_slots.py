"""This module defines the slot attrs attribute to ensure the type of component added is correct and to define behavior,defaults and argument passing behavio"""

import attrs
import uuid
from engforge.attributes import ATTR_BASE,AttributeInstance
import typing as typ

SLOT_TYPES = typ.Union["Component", "System"]
ITERATOR = typ.Union["ComponentIterator"]


class Slot(ATTR_BASE):
    """Slot defines a way to accept different components or systems in a system"""

    # These are added on System signals_slots_handler aka attrs field_transformer
    name: str
    accepted: SLOT_TYPES
    config_cls: "System"
    attr_prefix = 'Slot'
    none_ok: bool
    
    dflt_kw: dict = None #a dictionary of input in factory for custom inits
    default_ok = True #otherwise accept class with defaults

    is_iter: bool
    wide: bool  # only for component iterator
    #default_options = ATTR_BASE.default_options.copy()
    default_options = dict( repr=True,
                            validator=None,
                            cmp=None,
                            hash=None,
                            init=True,
                            metadata=None,
                            converter=None,
                            kw_only=True,
                            eq=None,
                            order=None,
                            on_setattr=None,
                            inherited=False)

    @classmethod
    def define(
        cls, *component_or_systems: SLOT_TYPES, none_ok=False, default_ok=True,
        dflt_kw:dict=None
    ):
        """taking a component or system class as possible input valid input is later validated as an instance of that class or subclass

        :param none_ok: will allow no component on that item, oterwise will fail
        :param default_ok: will create the slot class with no input if true
        :param dflt_kw: a dictionary of input in factory for custom inits overrides defaults_ok
        #TODO: add default_args,default_kwargs
        """
        from engforge.components import Component
        from engforge.component_collections import ComponentIter
        from engforge.system import System
        from engforge.eng.costs import CostModel

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

        ### Cost models should always default to None
        if any([issubclass(c, CostModel) for c in component_or_systems]):
            default_ok = False 
            none_ok = True

        # FIXME: come up with a better name :)
        new_name = f"Slot_{str(uuid.uuid4()).replace('-','')[0:16]}"
        new_slot = type(
            new_name,
            (Slot,),
            dict(
                name=new_name,
                accepted=component_or_systems,
                none_ok=none_ok,
                default_ok=default_ok,
                dflt_kw=dflt_kw,
                default_options = cls.default_options.copy()
            ),
        )
        new_slot.default_options['validator'] = new_slot.configure_instance
        new_slot.default_options['default'] = new_slot.make_factory()
        #print(new_slot)
        
        return new_slot

    @classmethod
    def define_iterator(
        cls,
        *component_or_systems: ITERATOR,
        none_ok:bool=False,
        default_ok:bool=True,
        wide:bool=True,
        dflt_kw:dict=None,
    ):
        """taking a type of component iterator, defines an interface that can be 'wide' where all items are executed in the same row on `System.run()`.

        Conversely if `wide` is false the system will loop over each item as if it was included in System.run(). Multiple ComponentIterators with wide=False will result in a `outer join` of the items.

        :param none_ok: will allow no component on that item, otherwise will fail
        :param default_ok: will create the slot class with no input if true
        :param dflt_kw: a dictionary of input in factory for custom inits
        :param wide: default is true, will determine if wide dataframe format, or outerproduct format when `System.run()` is called
        """
        from engforge.components import Component
        from engforge.component_collections import ComponentIter
        from engforge.system import System

        # Format THe Accepted Component Types
        assert (
            len(component_or_systems) == 1
        ), "only one slot allowed, try making a subclass"
        assert all(
            [issubclass(c, ComponentIter) for c in component_or_systems]
        ), "Not System Or Component Input"

        # FIXME: come up with a better name :)
        new_name = f"SlotITER_{str(uuid.uuid4()).replace('-','')[0:16]}"
        new_slot = type(
            new_name,
            (Slot,),
            dict(
                #default=cls.make_factory(),
                name=new_name,
                accepted=component_or_systems,
                none_ok=none_ok,
                default_ok=default_ok,
                dflt_kw=dflt_kw,
                is_iter=True,
                wide=wide,
                default_options = cls.default_options.copy()
            ),
        )
        #new_slot.default_options['validator'] = new_slot.configure_instance
        #new_slot.default_options['default'] =  new_slot.make_factory()
        return new_slot

    # Create a validator function
    @classmethod
    def configure_instance(cls, instance, attribute, value):
        from engforge.component_collections import ComponentIter

        comp_cls = cls.config_cls

        # apply wide behavior to componentiter instance
        if isinstance(value, ComponentIter) and attribute.type.wide == False:
            #print(f'validate {instance} {attribute} {value}')
            value.wide = False

        if value in cls.accepted or all([value is None, cls.none_ok]):
            return True

        if any([isinstance(value, a) for a in cls.accepted]):
            return True
        
        if cls.default_ok and value is None:
            return True

        raise ValueError(
            f"{instance} value {value} is not an accepted type for slot: {comp_cls.__name__}.{cls.name}"
        )

    @classmethod
    def make_factory(cls,**kwargs):
        accepted = cls.accepted
        #print(f'slot instance factory: {cls} {accepted}, {kwargs}')

        if isinstance(accepted,(tuple,list)) and len(accepted) > 0:
            accepted = accepted[0]

        #print(accepted,cls.dflt_kw,cls.default_ok)
        if cls.dflt_kw:
            return attrs.Factory(lambda: accepted(**cls.dflt_kw),False)
        elif cls.default_ok:
            return attrs.Factory(accepted,False)
        else:
            return None

#Support Previous SnakeCase
Slot = Slot