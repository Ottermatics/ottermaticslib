"""Defines a customizeable attrs attribute that is handled in configuration,

on init an instance of `Instance` type for any ATTR_BASE subclass is created """

import attrs, attr, uuid
from engforge.logging import LoggingMixin, log


class ATTRLog(LoggingMixin):
    pass
log = ATTRLog()


DEFAULT_COMBO = 'default'

class AttributeInstance:
    class_attr: "ATTR_BASE"
    system: "System"
    classname: str = "attribute" #for ref compatability
    backref: "ATTR_BASE"

    # TODO: universal slots method
    # __slots__ = ["system", "class_attr"]

    def __init__(
        self, class_attr: "CLASS_ATTR", system: "System", **kwargs
    ) -> None:
        self.class_attr = class_attr
        self.system = system
        self.compile(**kwargs)

    def compile(self, **kwargs):
        # raise NotImplementedError("Override Me!")
        pass

    def as_ref_dict(self)->dict:
        log.info(f'pass - as ref {self}')
        return None
    
    def get_alias(self,path):
        return path.split('.')[-1]
    
    def is_active(self,value=False) -> bool:
        mthd = 'dflt'
        if hasattr(self,'_active'):
            mthd = 'instance'
            value = self._active

        elif hasattr(self.class_attr,'active'):
            mthd = 'class'
            value = self.class_attr.active
        
        #print(f'{self}| {mthd} is_active: {value}')
        #the default
        return value       
    
    @property
    def combos(self):
        return self.class_attr.combos
    
    @property
    def active(self):
        return self.class_attr.active    


class ATTR_BASE(attrs.Attribute):
    """A base class that handles initalization in the attrs meta class scheme by ultimately createing an Instance"""

    name: str
    config_cls: "System"
    attr_prefix = "ATTR"
    instance_class: AttributeInstance = AttributeInstance  # Define me
    default_options: dict
    template_class = True

    #TODO: add generic selection & activation of attributes
    active: bool
    combos: list

    none_ok = False

    #Activation & Combo Selection Functionality
    @classmethod
    def process_combos(cls, combos):
        if isinstance(combos, str):
            if '*' in combos:
                raise KeyError("wildcard (*) not allowed in combos!")
            return combos.split(",")
        elif isinstance(combos, list):
            if any(['*' in c for c in combos]):
                raise KeyError("wildcard (*) not allowed in combos!")            
            return combos     

    #Initialization
    @classmethod
    def configure_for_system(cls, name, config_class, cb=None, **kwargs):
        """add the config class, and perform checks with `class_validate)
        :returns: [optional] a dictionary of options to be used in the make_attribute method
        """
        log.debug(f"{cls.__name__} is being configured for {cls.attr_prefix}")
        cls.name = name
        cls.config_cls = config_class

        if isinstance(cls.instance_class,ATTR_BASE) and not hasattr(cls.instance_class, "backref"):
            #get parent class
            refs = [cls,ATTR_BASE]
            mro_group = cls.mro()
            mro_group = mro_group[:mro_group.index(ATTR_BASE)]
            cans = [v for v in mro_group if v not in refs]
            last = cans[-1]
            # if not hasattr(cls,'instance_class') or not cls.instance_class:
            #     log.info(f'no instance class! {cls}')
            cls.instance_class.backref = last
        else:
            pass
            #print(f'backref exists {cls.instance_class.backref} {cls}')

        if cb is not None:
            cb(cls, config_class, **kwargs)

        return {}  # OVERWRITE ME "custom_options":False

    @classmethod
    def create_instance(cls, instance: "Configuration") -> AttributeInstance:
        """Create an instance of the instance_class"""
        if cls.instance_class is None:
            raise Exception(
                f"Instance Class Hasnt Been Defined For `{cls}.instance_class`"
            )
        if not hasattr(cls, "config_cls"):
            raise Exception(f"`config_cls` hasnt been defined for `{cls}`")

        cls.class_validate(instance=instance)
        return cls.instance_class(cls, instance)

    #Override Me:
    @classmethod
    def configure_instance(cls, instance, attribute, value):
        """validates the instance given attr's init routine"""
        pass

    @classmethod
    def class_validate(cls, instance, **kwargs):
        """validates onetime A method to validate the kwargs passed to the define method"""
        pass

    @classmethod
    def define_validate(cls, **kwargs):
        """A method to validate the kwargs passed to the define method"""
        pass
    
    #Interafce & Utility
    @classmethod
    def define(cls, **kwargs):
        """taking a component or system class as possible input valid input is later validated as an instance of that class or subclass"""
        assert not cls.template_class, f"{cls} is not template class and cannot defined anything, ie anything that has been created as a function of `define` cannot be defined"

        cls.define_validate(**kwargs)

        # Create A New Signals Class
        kw_pairs = [f"{k}_{v}" for k, v in kwargs.items()]
        new_name = f"{cls.attr_prefix}_{('_'.join(kw_pairs))}".replace(".", "_")
        # define the class dictionary
        new_dict = dict(name=new_name)
        new_slot = cls._setup_cls(new_name, new_dict, **kwargs)
        return new_slot

    @classmethod
    def _setup_cls(cls, name, new_dict, **kwargs):
       
        #randomize name for specifics reasons
        uid = str(uuid.uuid4())
        name = name + "_" + uid.replace("-", "")[0:16]
        new_dict['uuid'] = uid
        new_dict['default_options'] = cls.default_options.copy()
        new_dict['template_class'] = False
        new_dict['name'] = name
        new_slot = type(name, (cls,), new_dict)
        new_slot.default_options["default"] = new_slot.make_factory()
        new_slot.default_options["validator"] = new_slot.configure_instance
        log.debug(
            f"defined {name}|{new_slot} with {kwargs}| {new_slot.default_options}"
        )         
        return new_slot   

    @classmethod
    def make_factory(cls, **kwargs):
        #print(f"{cls} making factory with: {kwargs}")
        return attrs.Factory(cls.create_instance, takes_self=True)

    @classmethod
    def make_attribute(cls, name, comp_class, **kwargs):
        """makes an attrs.Attribute for the class"""
        cust_options = cls.configure_for_system(name, comp_class, **kwargs)
        # make copy for new instance
        opts = cls.default_options.copy()
        # update with custom kwargs
        if isinstance(cust_options, dict):
            opts.update(cust_options)
        # input has the final override
        opts.update(kwargs)
        # The core functionality
        opts.update(name=name, type=cls)

        # Handle the default and validator, if not overridden
        if "default" not in kwargs:
            opts["default"] = cls.make_factory()
        if "validator" not in kwargs:
            opts["validator"] = cls.configure_instance

        return attrs.Attribute(**opts)

    # Structural Orchestration Through Subclassing
    @classmethod
    def collect_cls(cls,system)->dict:
        """collects all the attributes for a system"""
        if not isinstance(system,type):
            system = system.__class__
        return {k:at.type for k,at in system._get_init_attrs_data(cls).items()}
    
    @classmethod
    def collect_attr_inst(cls,system,handle_inst=True)->dict:
        """collects all the attribute instances for a system"""
        cattr = cls.collect_cls(system)
        out = {}
        for k,v in cattr.items():
            inst = getattr(system,k)
            if inst is None and getattr(cls.instance_class,'none_ok',False) or (cls.instance_class is not None and not isinstance(inst,cls.instance_class)):
                log.warning(f"Attribute {k}|{inst} is not an instance of {cls.instance_class} in {system}")
                continue

            if handle_inst:
                inst = cls.handle_instance(inst)

            out[k] = inst
        return out

    @staticmethod
    def unpack_atrs(d,pre='',conv=None):

        if not conv:
            conv = lambda v: v
        if isinstance(d,dict):
            for k,v in d.items():
                if not isinstance(v,dict):
                    yield k,pre,conv(v)
                elif v:
                    if pre and not pre.endswith('.'): ppre = pre+'.'
                    ck = f'{ppre}{k}'
                    for kii,ki,vi in ATTR_BASE.unpack_atrs(v,ck):
                        yield kii,ki,conv(vi)
                else:
                    if pre and not pre.endswith('.'): ppre = pre+'.'
                    ck = f'{ppre}{k}'           
                    yield '',ck,v
        return d
    
    @classmethod
    def handle_instance(cls,canidate):
        """handles the instance, override as you wish"""
        #print('handle_instance',cls,canidate,cls.instance_class)

        if isinstance(canidate,cls.instance_class):
            o =  canidate.as_ref_dict()
            return o
        return canidate #live and let live
    
    @classmethod
    def subclasses(cls, out=None):
        """return all subclasses of components, including their subclasses
        :param out: out is to pass when the middle of a recursive operation, do not use it!
        """

        # init the set by default
        if out is None:
            out = set()

        for cls in cls.__subclasses__():
            if not cls.template_class:
                continue
            out.add(cls)
            cls.subclasses(out)

        return out    

ATTR_BASE.default_options = dict(
    # validator=ATTR_BASE.configure_instance,
    repr=False,
    cmp=None,
    hash=None,
    init=False,  # change this to allow for init input
    metadata=None,
    converter=None,
    kw_only=True,
    eq=None,
    order=None,
    on_setattr=None,
    inherited=False,
)
