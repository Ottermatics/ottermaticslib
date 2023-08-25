"""define a collection of components that will propigate to its parents dataframe

Types: #TODO
1. ComponentList, ordered by index
2. ComponentDict, ordered by key
3. ComponentGraph,  ?
"""

from collections import UserDict,UserList
from ottermatics.components import Component
from ottermatics.slots import SLOT
from ottermatics.configuration import otterize
from ottermatics.typing import *
from ottermatics.tabulation import Ref


import attrs

def check_comp_type(instance,attr,value):
    """ensures the input component type is a Component"""
    if isinstance(value,type) and issubclass(value,Component):
        return
    
    raise TypeError(f'Not A Component Class! {value}')

@otterize
class ComponentDict(Component,UserDict):
    """Stores components by name, and allows tabulation of them"""

    component_type:type = attrs.field(
                            validator= check_comp_type
                            )
    
    #tabulate_depth:int = attrs.field(default=1) #TODO: impement this

    _ref_cache: dict = None


    #Dict Setup
    def __on_init__(self):
        UserDict.__init__(self)

    def __setitem__(self, key, value):
        assert isinstance(value,self.component_type)
        super().__setitem__(key, value)
        #reset reference cache
        self._ref_cache = None 

    #Tabulation Override
    @property
    def internal_references(self):
        """lists the this_name.comp_key.<attr/prop key>: Ref format to override data_dict"""

        if self._ref_cache:
            return self._ref_cache

        out = {}
        out["attributes"] = at = {}
        out["properties"] = pr = {}

        for itkey,item in self.data.items():
            it_base_key = f'{itkey}'

            for key in item.classmethod_system_properties():
                k = f'{it_base_key}.{key}'
                pr[k] = Ref(item, key)

            for key in item.input_fields():
                k = f'{it_base_key}.{key}'
                at[k] = Ref(item, key, False)

        #cache the references
        self._ref_cache = out
        return out
