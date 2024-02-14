"""define a collection of components that will propigate to its parents dataframe

When `wide` is set each component's references are reported to the system's table, otherwise only one component's references are reported, however the system will iterate over the components by calling `system.iterable_components` 

Define a Iterable Component slot in a system by calling `SLOT.define_iterable(...,wide=True/False)`

CostModel isonly supported in wide mode at this time.

Types: 
1. ComponentList, ordered by index
2. ComponentDict, ordered by key
3. ComponentGraph,  ?#TODO:
"""

from collections import UserDict, UserList
from engforge.components import Component
from engforge.slots import SLOT
from engforge.configuration import forge
from engforge.typing import *
from engforge.tabulation import Ref, system_property
from engforge.properties import *


import attrs


def check_comp_type(instance, attr, value):
    """ensures the input component type is a Component"""
    from engforge.eng.costs import CostModel

    if  not instance.wide and isinstance(value, type) and issubclass(value, CostModel):
        raise TypeError(f"Cost Mixin Not Supported As Iter Type! {value}")

    if isinstance(value, type) and issubclass(value, Component):
        return

    raise TypeError(f"Not A Component Class! {value}")


class iter_tkn:
    """ambigious type to keep track of iterable position form system reference"""

    pass


@forge
class ComponentIter(Component):
    """Iterable components are designed to evaluate a large selection of components either one-by-one or all at once at the system level depending on if `wide` property is set."""

    _ref_cache: dict = None
    _item_refs: dict = None

    wide: bool = True

    # what holds the components
    data: iter

    # current item keu, non table type, this triggers `anything_changed` in `system._iterate_component()`
    current_item: iter_tkn = attr.ib(factory=lambda: None,hash=False,eq=False)
    _first_item_key: iter_tkn

    @property
    def current(self):
        """returns all data in wide format and the active key in _current_item"""
        if self.wide:
            return self.data
        else:
            if self.current_item is None:
                if not hasattr(self, "_first_item_key"):
                    next(self._item_gen())
                return self.data[self._first_item_key]
            return self.data[self.current_item]

    def _item_gen(self):
        for key, item in self.data.items():
            if not hasattr(self, "_first_item_key"):
                self._first_item_key = key
            yield key, item

    def __setitem__(self, key, value):
        assert isinstance(
            value, self.component_type
        ), f"{value} is not of type: {self.component_type}"
        super().__setitem__(key, value)
        self.reset()

    def reset(self):
        # reset reference cache
        self._prv_internal_references = None
        self._item_refs = None
        self.current_item = None

    def _item_key(self, itkey, item):
        """override this to customize data access to self.data or other container name"""
        return itkey

    @property
    def _internal_references(self) -> dict:
        """considers wide format to return active references"""
        if self.wide:
            return self._prv_internal_references
        else:
            if self.current_item is None:
                return self._item_refs[self._first_item_key]
            return self._item_refs[self.current_item]

    @instance_cached
    def comp_references(self):
        """Returns this components global references"""
        out = {}
        out["attributes"] = at = {}
        out["properties"] = pr = {}

        for key in self.classmethod_system_properties():
            pr[key] = Ref(self, key,True,False)

        for key in self.input_fields():
            at[key] = Ref(self, key, False,True)

        return out

    def internal_references(self,recache=False):
        """lists the this_name.comp_key.<attr/prop key>: Ref format to override data_dict"""
        
        if recache == False and hasattr(self,'_prv_internal_references') and self._prv_internal_references:
            return self._internal_references  

        keeprefcopy = lambda d: {k: {**c} for k, c in d.items()}

        out = keeprefcopy(self.comp_references)
        at = out["attributes"]  # = at = {}
        pr = out["properties"]  # = pr = {}
        _item_refs = {}

        for itkey, item in self._item_gen():
            it_base_key = self._item_key(itkey, item)

            _item_refs[itkey] = ir = keeprefcopy(self.comp_references)
            atr = ir["attributes"]  # = atr = {}
            prr = ir["properties"]  # = prr = {}

            # set property refs
            for key in item.classmethod_system_properties():
                k = f"{it_base_key}.{key}"
                rc = Ref(item, key,True,False)
                pr[k] = rc
                prr[key] = rc

            # set attr refs
            for key in item.input_fields():
                k = f"{it_base_key}.{key}"
                ri = Ref(item, key, False,True)
                at[k] = ri
                atr[key] = ri

        # cache the references
        self._prv_internal_references = out
        self._item_refs = _item_refs
        return self._internal_references
    
    def __hash__(self):
        return hash(id(self))


@forge
class ComponentDict(ComponentIter, UserDict):
    """Stores components by name, and allows tabulation of them"""

    component_type: type = attrs.field(validator=check_comp_type)

    # tabulate_depth:int = attrs.field(default=1) #TODO: impement this

    _ref_cache: dict = None

    # Dict Setup
    def __on_init__(self):
        UserDict.__init__(self)

    def __str__(self):
        return f"{self.__class__.__name__}[{len(self.data)}]"

    def __repr__(self) -> str:
        return str(self)


@forge
class ComponentIterator(ComponentIter, UserList):
    """Stores components by name, and allows tabulation of them"""

    component_type: type = attrs.field(validator=check_comp_type)

    # Dict Setup
    def __on_init__(self):
        UserList.__init__(self)

    def _item_gen(self):
        for i, item in enumerate(self.data):
            if not hasattr(self, "_first_item_key"):
                self._first_item_key = i
            yield i, item

    def _item_key(self, itkey, item):
        return f"{self.component_type.__name__.lower()}.{itkey}"

    # def __setitem__(self, key, value):
    #     assert isinstance(value,self.component_type)
    #     super().__setitem__(key, value)
    #     #reset reference cache
    #     self._ref_cache = None


#     #Tabulation Override
#     @property
#     def internal_references(self):
#         """lists the this_name.comp_key.<attr/prop key>: Ref format to override data_dict"""
#
#         if self._ref_cache:
#             return self._ref_cache
#
#         out = {}
#         out["attributes"] = at = {}
#         out["properties"] = pr = {}
#
#         for itkey,item in self._item_gen():
#             it_base_key = f'{itkey}'
#
#             for key in item.classmethod_system_properties():
#                 k = f'{it_base_key}.{key}'
#                 pr[k] = Ref(item, key)
#
#             for key in item.input_fields():
#                 k = f'{it_base_key}.{key}'
#                 at[k] = Ref(item, key, False)
#
#         #cache the references
#         self._ref_cache = out
#         return out

#     #Tabulation Override
#     @property
#     def internal_references(self):
#         """lists the this_name.comp_key.<attr/prop key>: Ref format to override data_dict"""
#
#         if self._ref_cache:
#             return self._ref_cache
#
#         out = {}
#         out["attributes"] = at = {}
#         out["properties"] = pr = {}
#
#         for it,item in self._item_gen():
#             it_base_key = f'{self.component_type.__name__.lower()}.{it}'
#
#             for key in item.classmethod_system_properties():
#                 k = f'{it_base_key}.{key}'
#                 pr[k] = Ref(item, key)
#
#             for key in item.input_fields():
#                 k = f'{it_base_key}.{key}'
#                 at[k] = Ref(item, key, False)
#
#         #cache the references
#         self._ref_cache = out
#         return out
