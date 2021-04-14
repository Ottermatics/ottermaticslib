import numpy, functools

class inst_vectorize(numpy.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]   



class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated_cls = decorated

        self.__class__.__name__ = self._decorated_cls.__name__

    def instance(self,*args,**kwargs):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated_cls(*args,**kwargs)
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)



#These are not working  vvvvvv
class SingletonMeta(type):
    """Metaclass for singletons. Any instantiation of a Singleton class yields
    the exact same object, e.g.:

    >>> class MyClass(metaclass=Singleton):
            pass
    >>> a = MyClass()
    >>> b = MyClass()
    >>> a is b
    True
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args,
                                                                **kwargs)
        return cls._instances[cls]

    @classmethod
    def __instancecheck__(mcs, instance):
        if instance.__class__ is mcs:
            return True
        else:
            return isinstance(instance.__class__, mcs)

def singleton_meta_object(cls):
    """Class decorator that transforms (and replaces) a class definition (which
    must have a Singleton metaclass) with the actual singleton object. Ensures
    that the resulting object can still be "instantiated" (i.e., called),
    returning the same object. Also ensures the object can be pickled, is
    hashable, and has the correct string representation (the name of the
    singleton)
    """
    assert isinstance(cls, SingletonMeta), \
        cls.__name__ + " must use Singleton metaclass"

    def instance(self):
        return cls

    cls.__call__ = instance
    cls.__hash__ = lambda self: hash(cls)
    cls.__repr__ = lambda self: cls.__name__
    cls.__reduce__ = lambda self: cls.__name__
    obj = cls()
    obj.__name__ = cls.__name__
    return obj