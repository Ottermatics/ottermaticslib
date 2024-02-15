from contextlib import contextmanager
import attr, attrs

from engforge.logging import LoggingMixin, log
from engforge.properties import *

import deepdiff
import typing
import datetime


# make a module logger
class ConfigLog(LoggingMixin):
    pass


log = ConfigLog()


# Class Definition Wrapper Methods
def property_changed(instance, variable, value):
    from engforge.tabulation import TabulationMixin

    if not isinstance(instance, (TabulationMixin)):
        return value
    
    if instance._anything_changed:
        # Bypass Check since we've already flagged for an update
        return value    

    if log.log_level <= 10:
        log.msg(f"checking property changed {instance}{variable.name} {value}")

    # Check if shoudl be updated
    cur = getattr(instance, variable.name)
    attrs = attr.fields(instance.__class__)
    if variable in attrs and value != cur:
        if log.log_level <= 10:
            instance.debug(f"changing variables: {variable.name} {value}")
        instance._anything_changed = True

    elif log.log_level <= 10 and variable in attrs:
        instance.warning(
            f"didnt change variables {variable.name}| {value} == {cur}"
        )

    elif log.log_level <= 10:
        instance.critical(f"missing variable {variable.name} not in {attrs}")

    return value


def signals_slots_handler(
    cls, fields, slots=True, signals=True, solvers=True, sys=True, plots=True
):
    """
    creates attributes as per the attrs.define field_transformer use case.

    Customize initalization with slots,signals,solvers and sys flags.
    """
    log.debug(f"transforming signals and slots for {cls.__name__}")

    for t in fields:
        if t.type is None:
            log.warning(f"{cls.__name__}.{t.name} has no type")

    out = []
    field_names = set([o.name for o in fields])
    log.debug(f"fields: {field_names}")

    # Add Important Fields
    in_fields = {f.name: f for f in fields}
    if "name" in in_fields:
        name = in_fields.pop("name")
        out.append(name)

    else:
        log.warning(f"{cls.__name__} does not have a name!")
        name = attrs.Attribute(
            name="name",
            default=attrs.Factory(lambda inst: str(inst.__class__.__name__).lower(),True),
            validator=None,
            repr=True,
            eq=True,
            eq_key=None,
            order=True,
            order_key=None,
            hash=None,
            init=True,
            metadata=None,
            type=str,
            converter=None,
            kw_only=True,
            inherited=True,
            on_setattr=None,
            alias="name",
        )
        out.append(name)

    # Assert there is no time in attributes if not a transient
    assert (
        "time" not in field_names
    ), f"`time` is a reserved attribute for transient operations, it will automatically appear in systems with transient configuration"

    # Index
    # assert 'index' not in field_names, f'`index` is a reserved attribute'
    if sys:
        if "index" not in field_names:
            index = attrs.Attribute(
                name="index",
                default=0,
                validator=None,
                repr=True,
                cmp=None,
                hash=None,
                init=False,
                metadata=None,
                type=int,
                converter=None,
                kw_only=True,
                eq=None,
                order=None,
                on_setattr=None,
                inherited=False,
            )
            out.append(index)

        # Add Time Parm
        #TODO: remove after formulated in testing
        if cls.transients_attributes():
            time = attrs.Attribute(
                name="time",
                default=0,
                validator=None,
                repr=True,
                cmp=None,
                hash=None,
                init=False,
                metadata=None,
                type=float,
                converter=None,
                kw_only=True,
                eq=None,
                order=None,
                on_setattr=None,
                inherited=False,
            )
            out.append(time)

    # Add Slots
    if slots:
        for slot_name, slot in cls.slots_attributes().items():
            at = slot.make_attribute(slot_name,cls)
            out.append(at)

    # Add Signals
    if signals:
        for signal_name, signal in cls.signals_attributes().items():
            at = signal.make_attribute(signal_name,cls)
            out.append(at)
    
    # Add SOLVERS
    if solvers:
        for solver_name, solver in cls.solvers_attributes().items():

            at = solver.make_attribute(solver_name,cls)
            out.append(at)

        # Add TRANSIENT
        for solver_name, solver in cls.transients_attributes().items():
            # add from cls since not accessible from attrs
            at = solver.make_attribute(solver_name,cls)
            out.append(at)

    if plots:
        for pltname, plot in cls.plot_attributes().items():
            at = plot.make_attribute(pltname,cls)
            out.append(at)
            

        for pltname, plot in cls.trace_attributes().items():
            at = plot.make_attribute(pltname,cls)
            out.append(at)

    created_fields = set([o.name for o in out])
    # print options
    if cls.log_level < 10:
        from engforge.plotting import PLOT

        for o in out:
            if isinstance(o.type, PLOT):
                print(o)

    # Merge Fields Checking if we are overriding an attribute with system_property
    #hack since TabulationMixin isn't available yet
    #print(cls.mro())
    if 'TabulationMixin' in str(cls.mro()):   
        cls_properties = cls.classmethod_system_properties(True)
    else:
        cls_properties = {}
    #print(f'tab found!! {cls_properties.keys()}')
    for k, o in in_fields.items():
        if k not in created_fields:
            if k in cls_properties and o.inherited:
                log.warning(
                    f"{cls.__name__} overriding inherited attr: {o.name} as a system property overriding it"
                )
            else:
                log.debug(f'{cls.__name__} adding attr: {o.name}')
                out.append(o)
        else:
            log.warning(
                f"{cls.__name__} skipping inherited attr: {o.name} as a custom type overriding it"
            )

    # Enforce Property Changing
    # FIXME: is this more reliable
    # real_out = []
    # for fld in out:
    #     if fld.type in (int,float,str):
    #         #log.warning(f"setting property changed on {fld}")
    #         fld = fld.evolve(on_setattr = property_changed)
    #         real_out.append(fld)
    #     else:
    #         real_out.append(fld)
    # #return real_out
    return out


# TODO: generalize the "Concept" attribute, and apply as attrs.define(field_transformer=this_concept)

# alternate initalisers
comp_transform = lambda c, f: signals_slots_handler(
    c, f, slots=True, signals=False, solvers=False, sys=False, plots=False
)


# This one should wrap all configuraitons to track changes, and special methods
def forge(cls=None, **kwargs):
    """Wrap all Configurations with this decorator with the following behavior
    1) we use the callback when any property changes
    2) repr is default
    3) hash is by object identity"""

    # Define defaults and handle conflicts
    dflts = dict(repr=False, eq=False, slots=False, kw_only=True,hash=False)
    for k, v in kwargs.items():
        if k in dflts:
            dflts.pop(k)

    if cls is not None:
        # we can't import system here since cls might be system, so look for any system subclasses
        if "System" in [c.__name__ for c in cls.mro()]:
            log.info(f"Configuring System: {cls.__name__}")
            acls = attr.s(
                cls,
                on_setattr=property_changed,
                field_transformer=signals_slots_handler,
                **dflts,
                **kwargs,
            )

            # must be here since can't inspect till after fields corrected
            acls.pre_compile() #custom class compiler
            acls.validate_class()
            if acls.__name__ != 'Configuration': #prevent configuration lookup
                acls.cls_compile() #compile subclasses
            return acls

        # Component/Config Flow
        log.msg(f"Configuring: {cls.__name__}")
        acls = attr.s(
            cls,
            on_setattr=property_changed,
            field_transformer=comp_transform,
            **dflts,
            **kwargs,
        )
        # must be here since can't inspect till after fields corrected
        acls.pre_compile() #custom class compiler
        acls.validate_class()
        if acls.__name__ != 'Configuration': #prevent configuration lookup
            acls.cls_compile() #compile subclasses
        return acls

    else:

        def f(cls, *args):
            return forge(cls, **kwargs)

        return f


def meta(title, desc=None, **kwargs):
    """a convienience wrapper to add metadata to attr.ib
    :param title: a title that gets formatted for column headers
    :param desc: a description of the property"""
    out = {
        "label": title.replace("_", " ").replace("-", " ").title(),
        "desc": None,
        **kwargs,
    }
    return out


# TODO: Make A MetaClass for Configuration, and provide forge interface there. Problem with replaceing metaclass later, as in the case of a singleton.


@forge
class Configuration(LoggingMixin):
    """Configuration is a pattern for storing attributes that might change frequently, and proivdes the core functionality for a host of different applications.

    Configuration is able to go through itself and its objects and map all included Configurations, just to a specific level.

    Common functionality includes an __on_init__ wrapper for attrs post-init method
    """

    _temp_vars = None

    name: str = attr.ib(
        default = attrs.Factory(lambda inst: str(inst.__class__.__name__).lower(),True),
        validator=attr.validators.instance_of(str),
        kw_only=True,
    )

    log_fmt = "[%(name)-24s]%(message)s"
    log_silo = True

    _created_datetime = None


    # Our Special Init Methodology
    def __on_init__(self):
        """Override this when creating your special init functionality, you must use attrs for input variables, this is called after parents are assigned"""
        pass

    def __pre_init__(self):
        """Override this when creating your special init functionality, you must use attrs for input variables, this is called before parents are assigned"""
        pass

    def __attrs_post_init__(self):
        """This is called after __init__ by attr's functionality, we expose __oninit__ for you to use!"""
        # Store abs path On Creation, in case we change
        
        from engforge.components import Component

        self._log = None
        self._anything_changed = True  # save by default first go!
        self._created_datetime = datetime.datetime.utcnow()
        self.__pre_init__()
                
        #Assign Parents, ensure single componsition
        for compnm,comp in self.internal_configurations(False).items():
            if isinstance(comp,Component):
                #TODO: allow multiple parents
                if (not hasattr(comp,'parent')) and (comp.parent is not None):
                    self.warning(f"Component {compnm} already has a parent {comp.parent} copying, and assigning to {self}")
                    setattr(self,compnm,attrs.evolve(comp,parent=self))
                else:
                    comp.parent = self
            
        self.debug(f"created {self.identity}")
        self.__on_init__()


    @classmethod
    def validate_class(cls):
        """A customizeable validator at the end of class creation in forge"""
        return

    @classmethod
    def pre_compile(cls):
        """an overrideable classmethod that executes when compiled, however will not execute as a subclass"""
        pass

    @classmethod
    def cls_compile(cls):
        """compiles all subclass functionality"""
        
        for subcls in cls.parent_configurations_cls():
            if subcls.subcls_compile is not Configuration:
                log.debug(f'{cls.__name__} compiling {subcls.__name__}')
            subcls.subcls_compile()

    @classmethod
    def subcls_compile(cls):
        """reliably compiles this method even for subclasses, override this to compile functionality for subclass interfaces & mixins"""
        pass

    @classmethod
    def parent_configurations_cls(cls)->list:
        """returns all subclasses that are a Configuration"""
        return [c for c in cls.mro() if issubclass(c,Configuration)]

    #Identity & location Methods
    @property
    def filename(self):
        """A nice to have, good to override"""
        fil = (
            self.identity.replace(" ", "_")
            .replace("-", "_")
            .replace(":", "")
            .replace("|", "_")
            .title()
        )
        filename = "".join(
            [
                c
                for c in fil
                if c.isalpha() or c.isdigit() or c == "_" or c == "-"
            ]
        ).rstrip()
        return filename

    @property
    def displayname(self):
        dn = (
            self.identity.replace("_", " ")
            .replace("|", " ")
            .replace("-", " ")
            .replace("  ", " ")
            .title()
        )
        # dispname = "".join([c for c in dn if c.isalpha() or c.isdigit() or c=='_' or c=='-']).rstrip()
        return dn

    @property
    def identity(self):
        """A customizeable property that will be in the log by default"""
        if not self.name or self.name == "default":
            return self.classname.lower()
        return f"{self.classname}-{self.name}".lower()

    @property
    def classname(self):
        """Shorthand for the classname"""
        return str(type(self).__name__).lower()

    # Configuration Information
    def internal_configurations(self,check_config=True)->dict:
        """go through all attributes determining which are configuration objects
        additionally this skip any configuration that start with an underscore (private variable)
        """

        if check_config:
            chk = lambda k,v: isinstance(v, Configuration)
        else:
            chk = lambda k,v: k in self.slots_attributes()

        return {
            k: v
            for k, v in self.__dict__.items()
            if chk(k,v) and not k.startswith("_")
        }

    def go_through_configurations(
        self, level=0, levels_to_descend=-1, parent_level=0,**kw
    ):
        """A generator that will go through all internal configurations up to a certain level
        if levels_to_descend is less than 0 ie(-1) it will go down, if it 0, None, or False it will
        only go through this configuration

        :return: level,config"""

        should_yield_level = lambda level: all(
            [
                level >= parent_level,
                any([levels_to_descend < 0, level <= levels_to_descend]),
            ]
        )

        if should_yield_level(level):
            yield "", level, self

        level += 1
        if 'check_config' not in kw:
            kw['check_config'] = False
        for key, config in self.internal_configurations(**kw).items():

            if isinstance(config,Configuration):
                for skey, level, iconf in config.go_through_configurations(
                    level, levels_to_descend, parent_level
                ):
                    yield f"{key}.{skey}" if skey else key, level, iconf
            else:
                yield key,level,config

    @property
    def attrs_fields(self) -> set:
        return set(attr.fields(self.__class__))

    @classmethod
    def _get_init_attrs_data(cls, subclass_of: type, exclude=False):
        choose = issubclass
        if exclude:
            choose = lambda ty, type_set: not issubclass(ty, type_set)

        attrval = {}
        if "__attrs_attrs__" in cls.__dict__:  # Handle Attrs Class
            for k, v in attrs.fields_dict(cls).items():
                if isinstance(v.type, type) and choose(v.type, subclass_of):
                    attrval[k] = v

        # else:  # Handle Pre-Attrs Class
        for k, v in cls.__dict__.items():
            if isinstance(v, type) and choose(v, subclass_of):
                attrval[k] = v

        return attrval

    @classmethod
    def _extract_type(cls,typ):
        """gathers valid types for an attribute.type"""
        from engforge.slots import SLOT
            
        if not isinstance(typ,type) or typ is None:
            return list()

        if issubclass(typ,SLOT):
            accept = typ.accepted
            if isinstance(accept,(tuple,list)):
                return list(accept)
            return [accept]
            
        elif issubclass(typ,Configuration):
            return [typ]
        
        elif issubclass(typ,TABLE_TYPES):
            return [typ]

    @classmethod
    def check_ref_slot_type(cls,sys_key:str)->list:
        """recursively checks class slots for the key, and returns the slot type"""
        slot_refs = cls.slot_refs()
        if sys_key in slot_refs:
            return slot_refs[sys_key]
        
        slts = cls.input_attrs()
        key_segs = sys_key.split('.')
        out = []
       # print(slts.keys(),sys_key)
        if '.' not in sys_key and sys_key not in slts:
            pass

        elif sys_key in slts:
            #print(f'slt find {sys_key}')
            return cls._extract_type(slts[sys_key].type)
        else:
            fst = key_segs[0]
            rem = key_segs[1:]
            if fst in slts:
                sub_clss = cls._extract_type(slts[fst].type)
                out = []
                for acpt in sub_clss:
                    if isinstance(acpt,type) and issubclass(acpt,Configuration):
                        
                        vals = acpt.check_ref_slot_type('.'.join(rem))
                        #print(f'recursive find {acpt}.{rem} = {vals}')
                        if vals:
                            out.extend(vals)
                        
                    elif isinstance(acpt,type):
                        out.append(acpt)

        slot_refs[sys_key] = out

        return out
    
    @classmethod
    def slot_refs(cls,recache=False):
        """returns all slot references in this configuration"""
        key = f'{cls.__name__}_prv_slot_sys_refs'
        if recache == False and hasattr(cls,key):
            return getattr(cls,key)
        o = {}
        setattr(cls,key,o)
        return o
        
    @classmethod
    def slots_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all slots attributes for class"""
        from engforge.slots import SLOT

        return cls._get_init_attrs_data(SLOT)

    @classmethod
    def signals_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all signals attributes for class"""
        from engforge.signals import SIGNAL

        return cls._get_init_attrs_data(SIGNAL)

    @classmethod
    def solvers_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all signals attributes for class"""
        from engforge.solver import SOLVER

        return cls._get_init_attrs_data(SOLVER)

    @classmethod
    def transients_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all signals attributes for class"""
        from engforge.dynamics import TRANSIENT

        return cls._get_init_attrs_data(TRANSIENT)

    @classmethod
    def trace_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all trace attributes for class"""
        from engforge.plotting import TRACE

        return cls._get_init_attrs_data(TRACE)

    @classmethod
    def plot_attributes(cls) -> typing.Dict[str, "Attribute"]:
        """Lists all plot attributes for class"""
        from engforge.plotting import PLOT

        return cls._get_init_attrs_data(PLOT)

    @classmethod
    def input_attrs(cls):
        return attr.fields_dict(cls)

    @classmethod
    def input_fields(cls):
        from engforge.dynamics import TRANSIENT
        from engforge.solver import SOLVER
        from engforge.signals import SIGNAL
        from engforge.slots import SLOT
        from engforge.plotting import PLOT, TRACE

        ignore_types = (
            SLOT,
            SIGNAL,
            SOLVER,
            TRANSIENT,
            tuple,
            list,
            PLOT,
            TRACE,
        )
        return cls._get_init_attrs_data(ignore_types, exclude=True)

    @classmethod
    def numeric_fields(cls):
        from engforge.dynamics import TRANSIENT
        from engforge.solver import SOLVER
        from engforge.signals import SIGNAL
        from engforge.slots import SLOT
        from engforge.plotting import PLOT, TRACE

        ignore_types = (
            SLOT,
            SIGNAL,
            SOLVER,
            TRANSIENT,
            str,
            tuple,
            list,
            PLOT,
            TRACE,
        )
        typ = cls._get_init_attrs_data(ignore_types, exclude=True)
        return {k: v for k, v in typ.items() if v.type in (int, float)}

    @classmethod
    def table_fields(cls):

        keeps = (str, float, int)  # TODO: add numpy fields
        typ = cls._get_init_attrs_data(keeps)
        return {k: v for k, v in typ.items()}

    # Dictonaries
    @property
    def as_dict(self):
        o = {k: getattr(self, k, None) for k, v in self.input_attrs().items()}
        o = {
            k: v if not isinstance(v, Configuration) else v.as_dict
            for k, v in o.items()
        }
        return o

    @property
    def input_as_dict(self):
        o = {k: getattr(self, k, None) for k in self.input_fields()}
        o = {
            k: v if not isinstance(v, Configuration) else v.input_as_dict
            for k, v in o.items()
        }
        return o

    @property
    def numeric_as_dict(self):
        o = {k: getattr(self, k, None) for k in self.numeric_fields}
        o = {
            k: v if not isinstance(v, Configuration) else v.numeric_as_dict
            for k, v in o.items()
        }
        return o

    # Hashes
    @property
    def unique_hash(self):
        d = self.as_dict
        return deepdiff.DeepHash(d)[d]

    @property
    def numeric_hash(self):
        d = self.input_as_dict
        return deepdiff.DeepHash(d)[d]

    @property
    def numeric_hash(self):
        d = self.numeric_as_dict
        return deepdiff.DeepHash(d)[d]

    @contextmanager
    def difference(self, **kwargs):
        """a context manager that will allow you to dynamically change any information, then will change it back in a fail safe way.

        with self.difference(name='new_name', value = new_value) as new_config:
            #do stuff with config, ok to fail

        you may not access any "private" variable that starts with an `_` as in _whatever

        difference is useful for saving slight differences in configuration in conjunction with solve
        you might create wrappers for evaluate, or implement a strategy pattern.

        only attributes may be changed.

        #TODO: allow recursive operation with sub comps or systems.
        #TODO: make a full system copy so the system can be reverted later
        """
        _temp_vars = {}

        _temp_vars.update(
            {
                arg: getattr(self, arg)
                for arg in kwargs.keys()
                if hasattr(self, arg)
                if not arg.startswith("_")
            }
        )

        bad_vars = set.difference(set(kwargs.keys()), set(_temp_vars.keys()))
        if bad_vars:
            self.warning("Could Not Change {}".format(",".join(list(bad_vars))))

        try:  # Change Variables To Input
            self.setattrs(kwargs)
            yield self
        finally:
            rstdict = {k:_temp_vars[k] for k,v in kwargs.items()}
            self.setattrs(rstdict)

    def setattrs(self,dict):
        """sets attributes from a dictionary"""
        msg = f"invalid keys {set(dict.keys()) - set(self.input_attrs())}"
        assert set(dict.keys()).issubset(set(self.input_attrs())), msg
        for k,v in dict.items():
            setattr(self,k,v)
