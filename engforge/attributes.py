"""Defines a customizeable attrs attribute that is handled in configuration,

on init an instance of `Instance` type for any ATTR_BASE subclass is created """

import attrs

class AttributeInstance:
    class_attr: "ATTR_BASE"
    system: "System"

    __slots__ = ["system", "class_attr"]

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
    config_obj: "System"
    attr_prefix = 'ATTR'
    instance_class:AttributeInstance = None #Define me
    default_options = dict(
            validator=None,
            repr=False,
            cmp=None,
            hash=None,
            init=False,
            metadata=None,
            converter=None,
            kw_only=True,
            eq=None,
            order=None,
            on_setattr=None,
            inherited=False,
    )    
    
    @classmethod
    def configure_for_system(cls, name, system):
        """add the system class, and check the dependent and independent values
        :returns: [optional] a dictionary of options to be used in the make_attribute method
        """
        cls.name = name
        cls.config_obj = system

        return {"custom_options":False} #OVERWRITE ME

    @classmethod
    def create_instance(cls, instance: "Configuration") -> AttributeInstance:
        """Create an instance of the instance_class"""
        if cls.instance_class is None:
            raise Exception(f'Instance Class Hasnt Been Defined For `{cls}.instance_class`')
        if not hasattr(cls, 'config_obj'):
            raise Exception(f'`config_obj` hasnt been defined for `{cls}`')
        
        cls.instance_validate(instance=instance)
        return cls.instance_class(instance, cls)
    
    @classmethod
    def instance_validate(cls,instance,**kwargs):
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
        new_name = f"{cls.attr_prefix}_{('_'.join(kwargs.values()))}".replace(".", "_")
        new_dict = dict(
            name=new_name,
            **kwargs
        )
        new_slot = type(new_name, (cls,), new_dict)
        return new_slot
    
    @classmethod
    def make_factory(cls,**kwargs):
        return attrs.Factory(cls.create_instance, takes_self=True)    
    
    @classmethod
    def make_attribute(cls,name,**kwargs):
        """makes an attrs.Attribute for the class"""
        cust_options = cls.configure_for_system(name, cls)
        #make copy for new instance
        opts = cls.default_options.copy()
        #update with custom kwargs
        if isinstance(cust_options,dict): 
            opts.update(cust_options)
        #input has the final override
        opts.update(kwargs)
        return attrs.Attribute(
                        default=cls.make_factory(),
                        name = name,
                        type=cls,
                        **opts)