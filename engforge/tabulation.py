"""Tabulation Module:

Incrementally records attrs input values and system_properties per save_data() call.

save_data() is called after item.evaluate() is called.
"""

from contextlib import contextmanager
import attr

from engforge.common import inst_vectorize, chunks
#from engforge.configuration import Configuration, forge
from engforge.engforge_attributes import AttributedBaseMixin
from engforge.logging import LoggingMixin
from engforge.typing import *
from engforge.properties import *
from typing import Callable

import numpy
import pandas
import os
import collections
import uuid


class TableLog(LoggingMixin):
    pass


log = TableLog()

#Dataframe interrogation functions
def is_uniform(s:pandas.Series):
    a = s.to_numpy() # s.values (pandas<0.24)
    if (a[0] == a).all():
        return True
    try:
        if not numpy.isfinite(a).any():
            return True
    except:
        pass
    return False

#key_func = lambda kv: len(kv[0].split('.'))*len(kv[1])
#length of matches / length of key
key_func = lambda kv: len(kv[1])/len(kv[0].split('.'))
#key_func = lambda kv: len(kv[1])

    
#TODO: remove duplicate columns
# mtches = collections.defaultdict(set)
# dfv = ecs.dataframe_variants[0]
# for v1,v2 in itertools.combinations(dfv.columns,2):
#     if numpy.all(dfv[v1]==dfv[v2]):
#         
#         mtches[v1].add(v2)
#         mtches[v2].add(v1)
        
    
def determine_split(raw,top:int=1,key_f = key_func):
    parents = {}

    for rw in raw:
        grp = rw.split('.')
        for i in range(len(grp)):
            tkn = '.'.join(grp[0:i+1])
            parents[tkn] = set()
        
    for rw in raw:
        for par in parents:
            if rw.startswith(par):
                parents[par].add(rw)


    grps = sorted(parents.items(),key=key_f,reverse=True)[:top]
    return [g[0] for g in grps]

def split_dataframe(df:pandas.DataFrame)->tuple:
    """split dataframe into a dictionary of invariants and a dataframe of variable values
    
    :returns tuple: constants,dataframe
    """
    uniform = {}
    for s in df:
        c = df[s]
        if is_uniform(c):
            uniform[s] = c[0]

    df_unique = df.copy().drop(columns=list(uniform))
    return uniform,df_unique[0] if len(df_unique)>0 else df_unique

class DataframeMixin:
    dataframe: pandas.DataFrame

    _split_dataframe_func = split_dataframe
    _determine_split_func = determine_split

    def smart_split_dataframe(self,df=None,split_groups=0,key_f = key_func):
        """splits dataframe between constant values and variants"""
        if df is None:
            df = self.dataframe
        out = {}
        const,vardf = split_dataframe( df )
        out['constants'] = const
        columns = set(vardf.columns)
        split_groups = min(split_groups,len(columns)-1)
        if split_groups==0:
            out['variants'] = vardf
        else:
            nconst = {}
            cgrp = determine_split(const,min(split_groups,len(const)-1))
            for i,grp in enumerate(sorted(cgrp,reverse=True)):
                columns = set(const)
                bad_columns = [c for c in columns if not c.startswith(grp)]
                good_columns = [c for c in columns if c.startswith(grp)]
                nconst[grp] = {c:const[c] for c in good_columns}
                for c in good_columns:
                    if c in columns:
                        columns.remove(c)
            out['constants'] = nconst

            raw = sorted(set(df.columns))
            grps = determine_split(raw,split_groups,key_f=key_f)
            
            for i,grp in enumerate(sorted(grps,reverse=True)):
                columns = set(vardf.columns)
                bad_columns = [c for c in columns if not c.startswith(grp)]
                good_columns = [c for c in columns if c.startswith(grp)]
                out[grp] = vardf.copy().drop(columns=bad_columns)
                #remove columns from vardf
                vardf = vardf.drop(columns=good_columns) 
            if vardf.size>0:
                out['misc'] = vardf
        return out
    
    @solver_cached
    def _split_dataframe(self):
        """splits dataframe between constant values and variants"""
        return split_dataframe(self.dataframe)    
    
    @property
    def dataframe_constants(self):
        return self._split_dataframe[0]
    
    @property
    def dataframe_variants(self):
        return self._split_dataframe[1:]    

    #Plotting Interface
    @property
    def skip_plot_vars(self) -> list:
        """accesses '_skip_plot_vars' if it exists, otherwise returns empty list"""
        if hasattr(self,'_skip_plot_vars'):
            return [var.lower() for var in self._skip_plot_vars]
        return []

class TabulationMixin(AttributedBaseMixin,DataframeMixin):
    """In which we define a class that can enable tabulation"""

    # Super Special Tabulating Index
    index = 0  # Not an attr on purpose, we want pandas to provide the index

    # override per class:
    _skip_table_parms: list = None
    _skip_plot_parms: list

    # Cached and private
    _table: dict = None
    _anything_changed: bool
    _always_save_data = False
    _prv_internal_references: dict 

    # Data Tabulation - Intelligent Lookups
    def save_data(
        self,
        index=None,
        saved=None,
        force=False,
        subforce=False,
        save_internal=False,
    ):
        """We'll save data for this object and any other internal configuration if
        anything changed or if the table is empty This should result in at least one row of data,
        or minimal number of rows to track changes
        This should capture all changes but save data"""

        if log.log_level <= 10:
            log.debug(
                f"check save for {self}|{index}|save: {self.anything_changed or force}"
            )

        if saved is None:
            saved = set()

        if self.anything_changed or not self.TABLE or force:
            if index is not None:
                # only set this after we check if anything changed
                self.index = index
            else:
                # normal increiment
                self.index += 1
            self.TABLE[self.index] = self.data_dict
            self.debug("saving data {}".format(self.index))
            saved.add(self)

        # TODO: move to slots structure
        if save_internal:
            for config in self.internal_components().values():
                if config is None:
                    continue
                if config not in saved:
                    self.debug(f"saving {config.identity}")
                    config.save_data(index, saved=saved, force=subforce)
                else:
                    self.debug(f"skipping saved config {config.identity}")

        # reset value
        self._anything_changed = False

    def internal_components(self,recache=False) -> dict:
        """get all the internal components"""
        if recache == False and hasattr(self,'_prv_internal_components'):
            return self._prv_internal_components 
        from engforge.components import Component
        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, Component)}
        self._prv_internal_components = o
        return o
    
    def internal_systems(self,recache=False) -> dict:
        """get all the internal components"""
        if recache == False and hasattr(self,'_prv_internal_systems'):
            return self._prv_internal_systems 
        from engforge.system import System
        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, System)}
        self._prv_internal_systems = o
        return o    
    
    def internal_tabulations(self,recache=False) -> dict:
        """get all the internal tabulations"""
        
        if recache == False and hasattr(self,'_prv_internal_tabs'):
            return self._prv_internal_tabs 
            
        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, TabulationMixin)}
        self._prv_internal_tabs = o
        return o    

    @instance_cached
    def iterable_components(self) -> dict:
        """Finds ComponentIter internal_components that are not 'wide'"""
        from engforge.component_collections import ComponentIter

        return {
            k: v
            for k, v in self.internal_components().items()
            if isinstance(v, ComponentIter) and not v.wide
        }

    def internal_references(self,recache=False) -> dict:
        """get references to all internal attributes and values"""
        if recache == False and hasattr(self,'_prv_internal_references'):
            return self._prv_internal_references  
              
        out = self._gather_references()
        self._prv_internal_references = out
        return out
    
    def _gather_references(self) -> dict:
        out = {}
        out["attributes"] = at = {}
        out["properties"] = pr = {}

        for key in self.system_properties_classdef():
            pr[key] = Ref(self, key,True,False)

        for key in self.input_fields():
            at[key] = Ref(self, key, False,True)

        return out

    @property
    def anything_changed(self):
        """use the on_setattr method to determine if anything changed,
        also assume that stat_tab could change without input changes"""
        if not hasattr(self, "_anything_changed"):
            self._anything_changed = True

        if self._anything_changed or self.always_save_data:
            if self.log_level <= 5:
                self.msg(
                    f"change: {self._anything_changed}| always: {self.always_save_data}"
                )
            return True
        return False

    def reset_table(self):
        """Resets the table, and attrs label stores"""
        self.index = 0
        self._table = None
        cls = self.__class__
        if hasattr(cls,'_{cls.__name__}_system_properties'):
            return  setattr(cls,'_{cls.__name__}_system_properties',None)    

    @property
    def TABLE(self):
        """this should seem significant"""
        if self._table is None:
            self._table = {}
        return self._table

    @property
    def table(self):
        """alias for TABLE"""
        return self.TABLE

    @solver_cached
    def dataframe(self):
        """The table compiled into a dataframe"""
        data = [self.TABLE[v] for v in sorted(self.TABLE)]
        return pandas.DataFrame(data=data, copy=True)
    

    @property
    def plotable_variables(self):
        '''Checks columns for ones that only contain numeric types or haven't been explicitly skipped'''
        if self.dataframe is not None:
            check_type = lambda key: all([ isinstance(v, NUMERIC_TYPES) for v in self.dataframe[key] ])
            check_non_mono =  lambda key: len(set(self.dataframe[key])) > 1


            return [ var for var in self.dataframe.columns 
                         if var.lower() not in self.skip_plot_vars and check_type(var) and check_non_mono(var)]
        return []

    # Properties & Attribues
    def print_info(self):
        print(f"INFO: {self.name} | {self.identity}")
        print("#" * 80)
        for key, value in sorted(self.data_dict.items(), key=lambda kv: kv[0]):
            print(f"{key:>40} | {value}")

    @property
    def data_dict(self):
        """this is what is captured and used in each row of the dataframe / table"""

        out = collections.OrderedDict()
        sref = self.internal_references()
        for k, v in sref["attributes"].items():
            if k in self.attr_raw_keys:
                out[k] = v.value()
        for k, v in sref["properties"].items():
            out[k] = v.value()
        return out

    @instance_cached
    def skip_attr(self) -> list:
        base = list((self.internal_configurations()).keys())
        if self._skip_table_parms is None:
            return base
        return self._skip_table_parms + base
    
    def format_label_attr(self, k, attr_prop):
        if attr_prop.metadata and "label" in attr_prop.metadata:
            return self.format_label(attr_prop.metadata["label"])
        return self.format_label(attr_prop)

    def format_label(self, label):
        return label.replace("_", " ").replace("-", " ").title()

    @instance_cached
    def attr_labels(self) -> list:
        """Returns formated attr label if the value is numeric"""
        attr_labels = list(
            [
                k.lower()
                for k, v in attr.fields_dict(self.__class__).items()
                if k not in self.skip_attr
            ]
        )
        return attr_labels

    @solver_cached
    def attr_row(self) -> list:
        """Returns formated attr data if the value is numeric"""
        return list(
            [
                getattr(self, k)
                for k in self.attr_raw_keys
                if k not in self.skip_attr
            ]
        )

    @instance_cached
    def attr_raw_keys(self) -> list:
        good = set(self.table_fields())
        return [k for k in attr.fields_dict(self.__class__).keys() if k in good]

    # @solver_cached
    # def attr_dict(self) -> list:
    #     """Returns formated attr data if the value is numeric"""
    #     return {
    #         k.lower(): getattr(self, k)
    #         for k in self.attr_raw_keys
    #         if hasattr(self, k) and k not in self.skip_attr
    #     }

    def set_attr(self, **kwargs):
        assert set(kwargs).issubset(set(self.attr_raw_keys))
        # TODO: support subcomponents via slots lookup
        for k, v in kwargs.items():
            setattr(self, k, v)

    @instance_cached
    def always_save_data(self):
        """Checks if any properties are stochastic (random)"""
        return self._always_save_data

    @solver_cached
    def table_dict(self):
        # We use __get__ to emulate the property, we could call regularly from self but this is more straightforward
        return {
            k.lower(): obj.__get__(self)
            for k, obj in self.system_properties_def.items()
        }

    @solver_cached
    def system_properties(self):
        # We use __get__ to emulate the property, we could call regularly from self but this is more straightforward
        tabulated_properties = [
            obj.__get__(self) for k, obj in self.system_properties_def.items()
        ]
        return tabulated_properties

    @instance_cached
    def system_properties_labels(self) -> list:
        """Returns the labels from table properties"""
        class_dict = self.__class__.__dict__
        tabulated_properties = [
            obj.label.lower() for k, obj in self.system_properties_def.items()
        ]
        return tabulated_properties

    @instance_cached
    def system_properties_types(self) -> list:
        """Returns the types from table properties"""
        class_dict = self.__class__.__dict__
        tabulated_properties = [
            obj.return_type for k, obj in self.system_properties_def.items()
        ]
        return tabulated_properties

    @instance_cached
    def system_properties_keys(self) -> list:
        """Returns the table property keys"""
        tabulated_properties = [
            k for k, obj in self.system_properties_def.items()
        ]
        return tabulated_properties

    @instance_cached
    def system_properties_description(self) -> list:
        """returns system_property descriptions if they exist"""
        class_dict = self.__class__.__dict__
        tabulated_properties = [
            obj.desc for k, obj in self.system_properties_def.items()
        ]
        return tabulated_properties

    @classmethod
    def cls_all_property_labels(cls):
        return [
            obj.label for k, obj in cls.system_properties_classdef().items()
        ]

    @classmethod
    def cls_all_property_keys(cls):
        return [k for k, obj in cls.system_properties_classdef().items()]

    @classmethod
    def cls_all_attrs_fields(cls):
        return attr.fields_dict(cls)

    @solver_cached
    def system_properties_def(self):
        """Combine other classes table properties into this one, in the case of subclassed system_properties as a property that is cached"""
        return self.__class__.system_properties_classdef()

    @classmethod
    def system_properties_classdef(cls,recache=False):
        """Combine other classes table properties into this one, in the case of subclassed system_properties"""
        from engforge.tabulation import TabulationMixin
        #Use a cache for deep recursion
        if not recache and hasattr(cls,'_{cls.__name__}_system_properties'):
            res=getattr(cls,'_{cls.__name__}_system_properties')
            if res is not None:
                return res

        #otherwise make the cache
        __system_properties = {}
        for k, obj in cls.__dict__.items():
            if isinstance(obj, system_property):
                __system_properties[k] = obj

        #
        mrl = cls.mro()
        inx_comp = mrl.index(TabulationMixin)

        # Ensures everything is includes Tabulation Functionality
        mrvs = mrl[1:inx_comp]

        for mrv in mrvs:
            # Remove anything not in the user code
            log.debug(f"adding system properties from {mrv.__name__}")
            if (
                issubclass(mrv, TabulationMixin)
                # and "engforge" not in mrv.__module__
            ):
                for k, obj in mrv.__dict__.items():
                    if k not in __system_properties and isinstance(
                        obj, system_property
                    ):  # Precedent
                        # Assumes our instance has assumed this table property
                        prop = getattr(cls, k, None)
                        if prop and isinstance(prop, system_property):
                            __system_properties[k] = prop
                            log.debug(
                                f"adding system property {mrv.__name__}.{k}"
                            )

        setattr(cls,'_{cls.__name__}_system_properties',__system_properties)

        return __system_properties

    @classmethod
    def pre_compile(cls):
        cls._anything_changed = True  # set default on class
        if any(
            [
                v.stochastic
                for k, v in cls.system_properties_classdef(True).items()
            ]
        ):
            log.info(f"setting always save on {cls.__name__}")

    @classmethod
    def locate(cls, key, fail=True) -> type:
        """:returns: the class or attribute by key if its in this system class or a subcomponent. If nothing is found raise an error"""
        # Nested
        log.debug(f"locating {cls.__name__} | key: {key}")
        val = None

        if "." in key:
            args = key.split(".")
            comp, sub = args[0], ".".join(args[1:])
            assert comp in cls.slots_attributes(), f"invalid {comp} in {key}"
            comp_cls = cls.slots_attributes()[comp].type.accepted[0]
            val = comp_cls.locate(sub, fail=False)

        elif key in cls.input_fields():
            val = cls.input_fields()[key]

        elif key in cls.system_properties_classdef():
            val = cls.system_properties_classdef()[key]

        # Fail on comand but otherwise return val
        if val is None:
            if fail:
                raise Exception(f"key {key} not found")
            return None
        return val

    def locate_ref(self, key, fail=True):
        """:returns: the instance assigned to this system. If the key has a `.` in it the comp the lowest level component will be returned"""

        log.debug(f"locating {self.identity} | key: {key}")
        val = None

        if "." in key:
            args = key.split(".")
            comp, sub = args[0], ".".join(args[1:])
            assert comp in self.slots_attributes(), f"invalid {comp} in {key}"
            # comp_cls = cls.slots_attributes()[comp].type.accepted[0]
            comp = getattr(self, comp)
            if "." not in key:
                return Ref(comp, sub)
            return comp.locate_ref(sub, fail=False)

        elif key in self.input_fields():
            # val= cls.input_fields()[key]
            return Ref(self, key)

        elif key in self.system_properties_classdef():
            # val= cls.system_properties_classdef()[key]
            return Ref(self, key)
        
        elif key in self.internal_configurations() or key in self.slots_attributes():
            return Ref(self,key)

        # Fail on comand but otherwise return val
        if val is None:
            if fail:
                raise Exception(f"key {key} not found")
            return None
        return val

    @property
    def system_id(self) -> str:
        """returns an instance unique id based on id(self)"""
        idd = id(self)
        return f"{self.classname}.{idd}"
    


class Ref:
    """A way to create portable references to system's and their component's properties, ref can also take a key to a zero argument function which will be evaluated,
    
    A dictionary can be used
    """

    __slots__ = ["comp", "key", "use_call",'use_dict','allow_set','eval_f']
    comp: "TabulationMixin"
    key: str
    use_call: bool
    use_dict: bool
    allow_set: bool
    eval_f: callable

    def __init__(self, component, key, use_call=True,allow_set=True,eval_f=None):
        self.comp = component
        if isinstance(self.comp,dict):
            self.use_dict = True
        else:
            self.use_dict = False
        self.key = key
        self.use_call = use_call
        self.allow_set = allow_set
        self.eval_f = eval_f

    def value(self):
        if self.use_dict:
            o = self.comp.get(self.key)
        else:
            o = getattr(self.comp, self.key)
        if self.use_call and callable(o):
            o = o()
        if self.eval_f:
            return self.eval_f(o)
        return o

    def set_value(self, val):
        if self.allow_set:
            return setattr(self.comp, self.key, val)
        else:
            raise Exception(f'not allowed to set value on {self.key}')

    def __str__(self) -> str:
        if self.use_dict:
            return f"REF[DICT.{self.key}]"
        return f"REF[{self.comp.classname}.{self.key}]"
