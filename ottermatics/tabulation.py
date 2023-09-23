"""Tabulation Module:

Incrementally records attrs input values and system_properties per save_data() call.

save_data() is called after item.evaluate() is called.
"""

from contextlib import contextmanager
import attr

from ottermatics.common import inst_vectorize, chunks
from ottermatics.configuration import Configuration, otterize
from ottermatics.logging import LoggingMixin
from ottermatics.typing import *
from ottermatics.properties import *
from typing import Callable

import numpy
import pandas
import os
import collections
import uuid


class TableLog(LoggingMixin):
    pass


log = TableLog()


# @otterize
class TabulationMixin(Configuration):
    """In which we define a class that can enable tabulation"""

    # Super Special Tabulating Index
    index = 0  # Not an attr on purpose, we want pandas to provide the index

    # override per class:
    skip_parms = None

    # Cached and private
    _table: dict = None
    _anything_changed: bool
    _always_save_data = False

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
            for config in self.internal_components.values():
                if config is None:
                    continue
                if config not in saved:
                    self.debug(f"saving {config.identity}")
                    config.save_data(index, saved=saved, force=subforce)
                else:
                    self.debug(f"skipping saved config {config.identity}")

        # reset value
        self._anything_changed = False

    @instance_cached
    def internal_components(self) -> dict:
        """get all the internal components"""
        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, TabulationMixin)}
        return o

    @instance_cached
    def iterable_components(self) -> dict:
        """Finds ComponentIter internal_components that are not 'wide'"""
        from ottermatics.component_collections import ComponentIter

        return {
            k: v
            for k, v in self.internal_components.items()
            if isinstance(v, ComponentIter) and not v.wide
        }

    @instance_cached
    def internal_references(self) -> dict:
        """get references to all internal attributes and values"""
        out = {}
        out["attributes"] = at = {}
        out["properties"] = pr = {}

        for key in self.classmethod_system_properties():
            pr[key] = Ref(self, key)

        for key in self.input_fields():
            at[key] = Ref(self, key, False)

        return out

    @property
    def anything_changed(self):
        """use the on_setattr method to determine if anything changed,
        also assume that stat_tab could change without input changes"""
        if not hasattr(self, "_anything_changed"):
            self._anything_changed = True

        if self._anything_changed or self.always_save_data:
            self.debug(
                f"change: {self._anything_changed}| always: {self.always_save_data}"
            )
            return True
        return False

    def reset_table(self):
        """Resets the table, and attrs label stores"""
        self.index = 0
        self._table = None

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
        data = [self.TABLE[v] for v in sorted(self.TABLE)]
        return pandas.DataFrame(data=data, copy=True)

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
        sref = self.internal_references
        for k, v in sref["attributes"].items():
            if k in self.attr_raw_keys:
                out[k] = v.value()
        for k, v in sref["properties"].items():
            out[k] = v.value()
        return out

    @instance_cached
    def skip_attr(self) -> list:
        if self.skip_parms is None:
            return list(self.internal_configurations.keys())
        return self.skip_parms + list(self.internal_configurations.keys())

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
            for k, obj in self.class_system_properties.items()
        }

    @solver_cached
    def system_properties(self):
        # We use __get__ to emulate the property, we could call regularly from self but this is more straightforward
        tabulated_properties = [
            obj.__get__(self) for k, obj in self.class_system_properties.items()
        ]
        return tabulated_properties

    @instance_cached
    def system_properties_labels(self) -> list:
        """Returns the labels from table properties"""
        class_dict = self.__class__.__dict__
        tabulated_properties = [
            obj.label.lower() for k, obj in self.class_system_properties.items()
        ]
        return tabulated_properties

    @instance_cached
    def system_properties_types(self) -> list:
        """Returns the types from table properties"""
        class_dict = self.__class__.__dict__
        tabulated_properties = [
            obj.return_type for k, obj in self.class_system_properties.items()
        ]
        return tabulated_properties

    @instance_cached
    def system_properties_keys(self) -> list:
        """Returns the table property keys"""
        tabulated_properties = [
            k for k, obj in self.class_system_properties.items()
        ]
        return tabulated_properties

    @instance_cached
    def system_properties_description(self) -> list:
        """returns system_property descriptions if they exist"""
        class_dict = self.__class__.__dict__
        tabulated_properties = [
            obj.desc for k, obj in self.class_system_properties.items()
        ]
        return tabulated_properties

    @classmethod
    def cls_all_property_labels(cls):
        return [
            obj.label for k, obj in cls.classmethod_system_properties().items()
        ]

    @classmethod
    def cls_all_property_keys(cls):
        return [k for k, obj in cls.classmethod_system_properties().items()]

    @classmethod
    def cls_all_attrs_fields(cls):
        return attr.fields_dict(cls)

    @solver_cached
    def class_system_properties(self):
        """Combine other classes table properties into this one, in the case of subclassed system_properties"""
        return self.__class__.classmethod_system_properties()

    @classmethod
    def classmethod_system_properties(cls):
        """Combine other classes table properties into this one, in the case of subclassed system_properties"""
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
                # and "ottermatics" not in mrv.__module__
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

        return __system_properties

    @classmethod
    def pre_compile(cls):
        cls._anything_changed = True  # set default on class
        if any(
            [
                v.stochastic
                for k, v in cls.classmethod_system_properties().items()
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

        elif key in cls.classmethod_system_properties():
            val = cls.classmethod_system_properties()[key]

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

        elif key in self.classmethod_system_properties():
            # val= cls.classmethod_system_properties()[key]
            return Ref(self, key)

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
    """A way to create portable references to system's and their component's properties, ref can also take a key to a zero argument function which will be evaluated"""

    __slots__ = ["comp", "key", "use_call"]
    comp: "TabulationMixin"
    key: str
    use_call: bool

    def __init__(self, component, key, use_call=True):
        self.comp = component
        self.key = key
        self.use_call = use_call

    def value(self):
        o = getattr(self.comp, self.key)
        if self.use_call and callable(o):
            return o()
        return o

    def set_value(self, val):
        return setattr(self.comp, self.key, val)

    def __str__(self) -> str:
        return f"REF[{self.comp.classname}.{self.key}]"
