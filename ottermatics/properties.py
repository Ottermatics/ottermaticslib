"""
Like typical python properties, normal functions are embelished with additional functionality.

`system_properties` is a core function that adds meta information to a normal python property to include its output in the results. It is the "y" in y=f(x).

`class_cache` a subclassing safe property that stores the result at runtime

`solver_cache` a property that is recalculated any time there is an update to any attrs variable.
"""

from ottermatics.logging import LoggingMixin
from ottermatics.typing import TABLE_TYPES


class PropertyLog(LoggingMixin):
    pass


log = PropertyLog()


class otter_prop:
    """an interface for extension and identification"""

    pass


class system_property(otter_prop):
    """
    this property notifies the system this is a property to be tabulated in the dataframe output.

    @system_property
    def function(...): < this uses __init__ to assign function

    @system_property(desc='really nice',label='funky function')
    def function(...):    < this uses __call__ to assign function

    When the underlying data has some random element, set stochastic=True and this will flag the component to always save data.

    Functions wrapped with table type must have an return annotion of (int,float,str)... ex def func() -> int:
    """

    desc = ""
    label = None
    stochastic = False
    get_func_return: type = None
    return_type = None

    def __init__(
        self,
        fget=None,
        fset=None,
        fdel=None,
        doc=None,
        desc=None,
        label=None,
        stochastic=False,
    ):
        """You can initalize just the functions, or precreate the object but with meta
        @system_property
        def function(...): < this uses __init__ to assign function

        @system_property(desc='really nice',label='funky function')
        def function(...):    < this uses __call__ to assign function
        """

        self.fget = fget
        if fget:
            self.get_func_return(fget)

        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        elif doc is not None:
            self.__doc__ = doc

        if desc is not None:
            self.desc = desc

        if label is not None:
            self.label = label
        elif fget is not None:
            self.label = fget.__name__.lower()

        # Set random flag
        self.stochastic = stochastic

    def __call__(self, fget=None, fset=None, fdel=None, doc=None):
        """this will be called when input is provided before property is set"""
        if fget and self.fget is None:
            self.get_func_return(fget)
            self.fget = fget
        if self.fset is None:
            self.fset = fset
        if self.fdel is None:
            self.fdel = fdel

        if doc is None and fget is not None:
            doc = fget.__doc__
        elif doc is not None:
            self.__doc__ = doc

        if self.label is None and fget is not None:
            self.label = fget.__name__.lower()

        return self

    def get_func_return(self, func):
        """ensures that the function has a return annotation, and that return annotation is in valid sort types"""
        anno = func.__annotations__
        typ = anno.get("return", None)
        if not typ in (int, str, float):
            raise Exception(
                f"system_property input: function {func.__name__} must have valid return annotation of type: {(int,str,float)}"
            )
        else:
            self.return_type = typ

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)


class cached_system_property(system_property):
    """A system property that caches the result when nothing changes. Use for expensive functions since the checking adds some overhead"""

    gname: str

    def get_func_return(self, func):
        anno = func.__annotations__
        self.gname = func.__name__
        typ = anno.get("return", None)
        if not typ in (int, str, float):
            raise Exception(
                f"system_property input: function {func.__name__} must have valid return annotation of type: {(int,str,float)}"
            )
        else:
            self.return_type = typ

    @property
    def private_var(self):
        return f"_{self.gname}"

    def __get__(self, instance: "TabulationMixin", objtype=None):
        if not hasattr(instance, self.private_var):
            from ottermatics.tabulation import TabulationMixin

            assert issubclass(
                instance.__class__, TabulationMixin
            ), f"incorrect class: {instance.__class__.__name__}"
            return self.set_cache(instance, reason="set")
        elif instance.anything_changed:
            return self.set_cache(instance)
        return getattr(instance, self.private_var)

    def __set__(self, instance, value):
        raise Exception(f"cannot set {self.gname}")

    def set_cache(self, instance, reason="update"):
        log.debug(
            f"solver cache for {instance.identity}.{self.private_var}| {reason}"
        )
        val = self.fget(instance)
        setattr(instance, self.private_var, val)
        return val


class solver_cached(otter_prop):
    """
    A property that updates a first time and then anytime time the input data changed, as signaled by attrs.on_setattr callback
    """

    def __init__(self, getter):
        # prevent runtime error by checking correct class

        self.gname = getter.__name__
        self.getter = getter
        self.setter = None
        self.deleter = None

    @property
    def private_var(self):
        return f"_{self.gname}"

    def __get__(self, instance: "TabulationMixin", objtype=None):
        if not hasattr(instance, self.private_var):
            from ottermatics.tabulation import TabulationMixin

            assert issubclass(
                instance.__class__, TabulationMixin
            ), f"incorrect class: {instance.__class__.__name__}"
            return self.set_cache(instance, reason="set")
        elif instance.anything_changed:
            return self.set_cache(instance)
        return getattr(instance, self.private_var)

    def __set__(self, instance, value):
        raise Exception(f"cannot set {self.gname}")

    def set_cache(self, instance, reason="update"):
        log.debug(
            f"caching attr for {instance.identity}.{self.private_var}| {reason}"
        )
        val = self.getter(instance)
        setattr(instance, self.private_var, val)
        return val


class instance_cached(otter_prop):
    """
    A property that caches a result to an instance the first call then returns that each successive call
    """

    def __init__(self, getter):
        # prevent runtime error by checking correct class

        self.gname = getter.__name__
        self.getter = getter
        self.setter = None
        self.deleter = None

    @property
    def private_var(self):
        return f"_{self.gname}"

    def __get__(self, instance: "TabulationMixin", objtype=None):
        if not hasattr(instance, self.private_var):
            from ottermatics.tabulation import TabulationMixin

            assert issubclass(
                instance.__class__, TabulationMixin
            ), f"incorrect class: {instance.__class__.__name__}"
            return self.set_cache(instance, reason="set")
        return getattr(instance, self.private_var)

    def __set__(self, instance, value):
        raise Exception(f"cannot set {self.gname}")

    def set_cache(self, instance, reason="update"):
        log.debug(
            f"caching instance for {instance.identity}.{self.private_var}| {reason}"
        )
        val = self.getter(instance)
        setattr(instance, self.private_var, val)
        return val


# TODO: make a system_property_cache = system_property + solver_cache
class class_cache(otter_prop):
    """a property that caches a value on the class at runtime, and then maintains that value for the duration of the program. A flag with the class name ensures the class is correct. Intended for instance methods"""

    def __init__(self, getter):
        self.gname = getter.__name__
        self.getter = getter
        self.setter = None
        self.deleter = None

    @property
    def private_var(self):
        return f"_{self.gname}"

    @property
    def id_var(self):
        return f"__{self.gname}"

    def __get__(self, instance: "Any", objtype=None):
        cls = instance.__class__
        if hasattr(cls, self.private_var):
            # check and gtfo
            # print(f'getting private var {self.private_var}')
            return getattr(cls, self.private_var)
        else:
            if not hasattr(cls, self.id_var):
                # initial cache
                return self.set_cache(instance, cls)
            elif cls.__name__ != getattr(cls, self.id_var):
                # recache the system if name doesn't match
                return self.set_cache(instance, cls)
        # print(f'getting private def {self.private_var}')

        return getattr(cls, self.private_var)

    def set_cache(self, instance, cls):
        log.debug(f"cls caching for {cls.__name__}.{self.private_var}")
        val = self.getter(instance)
        setattr(cls, self.private_var, val)
        setattr(cls, self.id_var, cls.__name__)
        return val

    def __set__(self, instance, value):
        raise Exception(f"cannot set {self.gname}")
