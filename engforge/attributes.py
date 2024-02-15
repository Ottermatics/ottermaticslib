"""Defines a customizeable attrs attribute that is handled in configuration,

on init an instance of `Instance` type for any ATTR_BASE subclass is created """

import attrs
from engforge.logging import LoggingMixin, log

class ATTRLog(LoggingMixin):
    pass


log = ATTRLog()


class AttributeInstance:
    class_attr: "ATTR_BASE"
    system: "System"

    #TODO: universal slots method
    #__slots__ = ["system", "class_attr"]

    def __init__(self, class_attr: "CLASS_ATTR", system: "System",**kwargs) -> None:
        self.class_attr = class_attr
        self.system = system
        self.compile(**kwargs)

    def compile(self,**kwargs):
        #raise NotImplementedError("Override Me!")
        pass
    


class ATTR_BASE(attrs.Attribute):
    """A base class that handles initalization in the attrs meta class scheme by ultimately createing an Instance"""
    name: str
    config_cls: "System"
    attr_prefix = 'ATTR'
    instance_class:AttributeInstance = None #Define me
    default_options: dict
    
    @classmethod
    def configure_for_system(cls, name, config_class,**kwargs):
        """add the config class, and perform checks with `class_validate)
        :returns: [optional] a dictionary of options to be used in the make_attribute method
        """
        log.info(f'{cls.__name__} is being configured for {cls.attr_prefix}')
        cls.name = name
        cls.config_cls = config_class

        return {} #OVERWRITE ME "custom_options":False

    @classmethod
    def create_instance(cls, instance: "Configuration") -> AttributeInstance:
        """Create an instance of the instance_class"""
        if cls.instance_class is None:
            raise Exception(f'Instance Class Hasnt Been Defined For `{cls}.instance_class`')
        if not hasattr(cls, 'config_cls'):
            raise Exception(f'`config_cls` hasnt been defined for `{cls}`')
        
        cls.class_validate(instance=instance)
        return cls.instance_class(cls,instance)
    
    @classmethod
    def class_validate(cls,instance,**kwargs):
        """validates onetime A method to validate the kwargs passed to the define method"""
        pass
        

    @classmethod
    def define_validate(cls, **kwargs):
        """A method to validate the kwargs passed to the define method"""
        pass

    @classmethod
    def define(cls,**kwargs):
        """taking a component or system class as possible input valid input is later validated as an instance of that class or subclass"""
        cls.define_validate(**kwargs)

        # Create A New Signals Class
        kw_pairs = [f"{k}_{v}" for k,v in kwargs.items()]
        new_name = f"{cls.attr_prefix}_{('_'.join(kw_pairs))}".replace(".", "_")
        new_dict = dict(
            name=new_name,
            default_options=cls.default_options.copy()
            **kwargs
        )
        
        new_slot = type(new_name, (cls,), new_dict)
        #Prep options in case of copying, do this in subclasses
        new_slot.default_options['default'] = new_slot.make_factory()
        new_slot.default_options['validator'] = new_slot.validate_instance

        log.info(f'defined {new_slot} with {kwargs}| {new_slot.default_options}')

        return new_slot
    
    @classmethod
    def make_factory(cls,**kwargs):
        print(f'{cls} making factory with: {kwargs}')
        return attrs.Factory(cls.create_instance, takes_self=True)    
    
    @classmethod
    def make_attribute(cls,name,comp_class,**kwargs):
        """makes an attrs.Attribute for the class"""
        cust_options = cls.configure_for_system(name, comp_class,**kwargs)
        #make copy for new instance
        opts = cls.default_options.copy()
        #update with custom kwargs
        if isinstance(cust_options,dict): 
            opts.update(cust_options)
        #input has the final override
        opts.update(kwargs)
        #The core functionality
        opts.update(name = name,
                    type = cls)
        
        #Handle the default and validator, if not overridden
        if 'default' not in kwargs:
            opts['default'] = cls.make_factory()
        if 'validator' not in kwargs:
            opts['validator'] = cls.validate_instance

        return attrs.Attribute(**opts)
    
    @classmethod
    def validate_instance(cls,instance,attribute,value):
        """validates the instance given attr's init routine"""
        pass
    
ATTR_BASE.default_options = dict(
        #validator=ATTR_BASE.validate_instance,
        repr=False,
        cmp=None,
        hash=None,
        init=False, #change this to allow for init input
        metadata=None,
        converter=None,
        kw_only=True,
        eq=None,
        order=None,
        on_setattr=None,
        inherited=False,
)        