from contextlib import contextmanager
import attr
import typing

# from ottermatics.logging import LoggingMixin, log
from ottermatics.tabulation import TabulationMixin, system_property
from ottermatics.configuration import otterize, Configuration
from ottermatics.properties import class_cache


import os, sys
import inspect

# import pathlib
import random
import matplotlib.pyplot as plt


@otterize
class Component(TabulationMixin):
    """Component is an Evaluatable configuration with tabulation and reporting functionality"""

    parent: typing.Union['Component','System'] = attr.ib(default=None)

    @classmethod
    def subclasses(cls, out=None):
        """return all subclasses of components, including their subclasses
        :param out: out is to pass when the middle of a recursive operation, do not use it!
        """

        # init the set by default
        if out is None:
            out = set()

        for cls in cls.__subclasses__():
            out.add(cls)
            cls.subclasses(out)

        return out

    #UPDATE & POST UPDATE recieve the same kw args
    def update(self, system,**kw):
        """override with custom system interaction"""
        pass

    def post_update(self, system,**kw):
        """override with custom system interaction, will execute after all components have been updated"""
        pass    

    def update_internal(self,ignore:set=None,**kw):
        """updates internal components with self"""
        self.debug(f'updating internal {self.__class__.__name__}.{self}')
        for key, config in self.internal_components().items():
            if ignore is not None and config in ignore:
                continue
            self.debug(f'updating internal component {key}')
            config.update(self)
            config.update_internal(ignore)
        if ignore: ignore.add(self)

    def post_update_internal(self,ignore:set=None,**kw):
        """updates internal components with self"""
        self.debug(f'post updating internal {self.__class__.__name__}.{self}')
        for key, config in self.internal_components().items():
            if ignore is not None and config in ignore:
                continue
            self.debug(f'post updating internal component {config.__class__.__name__}.{config}')       
            config.post_update(self)
            config.post_update_internal(ignore)    
        if ignore: ignore.add(self)        


# TODO: move inspection for components to mixin for inspection of slots
#


# @classmethod
# def component_subclasses(cls):
#     # We find any components in ottermatics and will exclude them
#     OTTER_ITEMS = set(
#         list(
#             flatten(
#                 [
#                     inspect.getmembers(mod, inspect.isclass)
#                     for mkey, mod in sys.modules.items()
#                     if "ottermatics" in mkey
#                 ]
#             )
#         )
#     )
#     COMP_CLASS = [
#         modcls
#         for mkey, modcls in OTTER_ITEMS
#         if type(modcls) is type and issubclass(modcls, Component)
#     ]
#     return {
#         scls.__name__: scls
#         for scls in cls.__subclasses__()
#         if scls not in COMP_CLASS
#     }


# MOVE THIS SOMEWHERE
#     def reset_data(self):
#         self.reset_table()
#         self._stored_plots = []
#         self.index = 0
#         for config in self.internal_components().values():
#             config.reset_data()
#


#
# @otterize
# class ComponentIterator(Component):
#     """An object to loop through a list of components as the system is evaluated,
#
#     iterates through each component and pipes data_row and data_label to this objects table
#     """
#
#     _iterated_component_type = (
#         None  # provides interface for tabulation & data reflection
#     )
#     _components = []
#     shuffle_mode = False
#     _shuffled = None
#
#     @property
#     def component_list(self):
#         return self._components
#
#     @component_list.setter
#     def component_list(self, new_components):
#         if all([isinstance(item, Component) for item in new_components]):
#             self._components = new_components
#         else:
#             self.warning("Input Components Were Not All Of Type Component")
#
#     @property
#     def current_component(self) -> Component:
#         out = self[self.index]
#         if out is None:
#             if self.index == 0:
#                 return self[0]
#             return self[-1]
#         return out
#
#     # Wrappers for current component
#     @property
#     def data_dict(self):
#         base = super(ComponentIterator, self).data_dict
#         base.update(self.current_component.data_dict)
#         return base
#
#     @property
#     def data_row(self):
#         return (
#             super(ComponentIterator, self).data_row
#             + self.current_component.data_row
#         )
#
#     @property
#     def data_label(self):
#         return (
#             super(ComponentIterator, self).data_label
#             + self.current_component.data_label
#         )
#
#     @property
#     def plot_variables(self):
#         return (
#             super(ComponentIterator, self).plot_variables
#             + self.current_component.plot_variables
#         )
#
#     @classmethod
#     def cls_all_property_labels(cls):
#         these_properties = [
#             obj.label.lower()
#             for k, obj in cls.__dict__.items()
#             if isinstance(obj, system_property)
#         ]
#         if (
#             cls._iterated_component_type is not None
#             and type(cls._iterated_component_type) is type
#             and issubclass(cls._iterated_component_type, Component)
#         ):
#             iterated_properties = (
#                 cls._iterated_component_type.cls_all_property_labels()
#             )
#             these_properties = list(set(iterated_properties + these_properties))
#         return these_properties
#
#     @classmethod
#     def cls_all_property_keys(cls):
#         these_properties = [
#             k
#             for k, obj in cls.__dict__.items()
#             if isinstance(obj, system_property)
#         ]
#         if (
#             cls._iterated_component_type is not None
#             and type(cls._iterated_component_type) is type
#             and issubclass(cls._iterated_component_type, Component)
#         ):
#             iterated_properties = (
#                 cls._iterated_component_type.cls_all_property_keys()
#             )
#             these_properties = list(set(iterated_properties + these_properties))
#         return these_properties
#
#     @classmethod
#     def cls_all_attrs_fields(cls):
#         these_properties = attr.fields_dict(cls)  # dictonary
#         if cls._iterated_component_type is not None and issubclass(
#             cls._iterated_component_type, Component
#         ):
#             iterated_properties = attr.fields_dict(cls._iterated_component_type)
#             these_properties.update(iterated_properties)
#         return these_properties
#
#     # these are the methods we're overrideing
#     # @classmethod
#     # def cls_all_property_labels(cls):
#     #     return [obj.label for k,obj in cls.__dict__.items() if isinstance(obj,system_property)]
#
#     # @classmethod
#     # def cls_all_attrs_fields(cls):
#     #     return attr.fields_dict(cls)
#
#     # Magicz
#     # @functools.cached_property
#     def _item_generator(self):
#         if self.shuffle_mode:
#             if self._shuffled is None:
#                 self._shuffled = random.sample(
#                     self.component_list, len(self.component_list)
#                 )
#             return self._shuffled
#         return self.component_list
#
#     # def __getitem__(self,index):
#     #     if index >= 0 and index < len(self.component_list):
#     #         return self.component_list[index]
#     #     if index < 0 and index > -len(self.component_list):
#     #         return self.component_list[index]
#
#     # def __iter__(self):
#     #     #TODO: Add shuffle mode!
#     #     for item in self._item_generator():
#     #         self._anything_changed = True
#     #         yield item
#     #         self._anything_changed = True
