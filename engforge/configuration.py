import attr, attrs

from engforge.engforge_attributes import AttributedBaseMixin
from engforge.logging import LoggingMixin, log
from engforge.properties import *
import datetime


# make a module logger
class ConfigLog(LoggingMixin):
    pass


log = ConfigLog()


#Wraps Configuration with a decorator, similar to @attrs.define(**options)
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


# Class Definition Wrapper Methods
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
            cmp=None,
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
        from engforge.attr_plotting import PLOT

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


# alternate initalisers
comp_transform = lambda c, f: signals_slots_handler(
    c, f, slots=True, signals=False, solvers=False, sys=False, plots=False
)

# TODO: Make A MetaClass for Configuration, and provide forge interface there. Problem with replaceing metaclass later, as in the case of a singleton.
@forge
class Configuration(AttributedBaseMixin):
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


