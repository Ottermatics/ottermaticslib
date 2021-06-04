import numpy, functools

from ottermatics.logging import LoggingMixin, logging

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







# class MetaRegistry(type):
    
#     REGISTRY = {}

#     def __new__(meta, name, bases, class_dict):
#         cls = type.__new__(meta, name, bases, class_dict)
#         if name not in registry:
#             meta.register_class(cls)
#         return cls

#     def register_class(target_class):
#         REGISTRY[target_class.__name__] = target_class        


class InputSingletonMeta(type):
    """Metaclass for singletons. Any instantiation of a Singleton class yields
    the exact same object, for the same given input, e.g.:

    >>> class MyClass(metaclass=Singleton):
            pass
    >>> a = MyClass(input='same')
    >>> b = MyClass(input='diff')
    >>> a is b
    False
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        keyarg = {'class':cls,'args':args,**kwargs}
        keyarg['class'] = cls
        keyarg['args'] = args
        key = frozenset(keyarg.items())
        if key not in cls._instances:
            #print(f'creating new {key}')
            cls._instances[key] = super(InputSingletonMeta, cls).__call__(*args,
                                                                **kwargs)
        return cls._instances[key]

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



from ottermatics.common import *
from ottermatics.logging import *
from ottermatics.locations import *
import numpy
import os
import logging
import arrow, datetime


#TODO: Move to ray-util
# FROM RAY INSPECT_SERIALIZE
"""A utility for debugging serialization issues."""
from typing import Any, Tuple, Set, Optional
import inspect
import ray.cloudpickle as cp
import colorama
from contextlib import contextmanager

def recursive_python_module_line_counter(curpath=None):
    total_lines = 0
    if curpath is None or not isinstance(curpath, str):
        curpath = os.path.realpath(os.curdir)
        
    print(f'Getting Python Lines In {curpath}')
    for dirpath, dirs, fils in os.walk(curpath):
        for fil in fils:
            if fil.endswith('.py'):
                filpath = os.path.join(dirpath,fil)
                with open(filpath,'r') as fp:
                    lines = len(str(fp.read()).split('\n'))
                    total_lines += lines
                    print(f'{filpath}: {lines} / {total_lines}')

    print(f'Total Lines {total_lines}')

def flat2gen(alist):
  for item in alist:
    if isinstance(item, (list,tuple)):
      for subitem in item: yield subitem
    else:
      yield item

def flatten(alist):
    return list(flat2gen(alist))

@contextmanager
def _indent(printer):
    printer.level += 1
    yield
    printer.level -= 1


class _Printer(LoggingMixin):

    log_level = logging.WARNING

    def __init__(self):
        self.level = 0

    def indent(self):
        return _indent(self)

    def print(self, msg, warning=False):
        indent = "    " * self.level
        if warning:
            self.warning(indent+msg)
        else:
            self.debug(indent+msg)


_printer = _Printer()


class FailureTuple:
    """Represents the serialization 'frame'.

    Attributes:
        obj: The object that fails serialization.
        name: The variable name of the object.
        parent: The object that references the `obj`.
    """

    def __init__(self, obj: Any, name: str, parent: Any):
        self.obj = obj
        self.name = name
        self.parent = parent

    def __repr__(self):
        return f"FailTuple({self.name} [obj={self.obj}, parent={self.parent}])"


def _inspect_func_serialization(base_obj, depth, parent, failure_set):
    """Adds the first-found non-serializable element to the failure_set."""
    assert inspect.isfunction(base_obj)
    closure = inspect.getclosurevars(base_obj)
    found = False
    if closure.globals:
        _printer.print(f"Detected {len(closure.globals)} global variables. "
                       "Checking serializability...")

        with _printer.indent():
            for name, obj in closure.globals.items():
                serializable, _ = inspect_serializability(
                    obj,
                    name=name,
                    depth=depth - 1,
                    _parent=parent,
                    _failure_set=failure_set)
                found = found or not serializable
                if found:
                    break

    if closure.nonlocals:
        _printer.print(
            f"Detected {len(closure.nonlocals)} nonlocal variables. "
            "Checking serializability...")
        with _printer.indent():
            for name, obj in closure.nonlocals.items():
                serializable, _ = inspect_serializability(
                    obj,
                    name=name,
                    depth=depth - 1,
                    _parent=parent,
                    _failure_set=failure_set)
                found = found or not serializable
                if found:
                    break
    if not found:
        _printer.print(
            f"WARNING: Did not find non-serializable object in {base_obj}. "
            "This may be an oversight.",warning=True)
    return found


def _inspect_generic_serialization(base_obj, depth, parent, failure_set):
    """Adds the first-found non-serializable element to the failure_set."""
    assert not inspect.isfunction(base_obj)
    functions = inspect.getmembers(base_obj, predicate=inspect.isfunction)
    found = False
    with _printer.indent():
        for name, obj in functions:
            serializable, _ = inspect_serializability(
                obj,
                name=name,
                depth=depth - 1,
                _parent=parent,
                _failure_set=failure_set)
            found = found or not serializable
            if found:
                break

    with _printer.indent():
        members = inspect.getmembers(base_obj)
        for name, obj in members:
            if name.startswith("__") and name.endswith(
                    "__") or inspect.isbuiltin(obj):
                continue
            serializable, _ = inspect_serializability(
                obj,
                name=name,
                depth=depth - 1,
                _parent=parent,
                _failure_set=failure_set)
            found = found or not serializable
            if found:
                break
    if not found:
        _printer.print(
            f"WARNING: Did not find non-serializable object in {base_obj}. "
            "This may be an oversight.",warning=True)
    return found


def inspect_serializability(
        base_obj: Any,
        name: Optional[str] = None,
        depth: int = 3,
        _parent: Optional[Any] = None,
        _failure_set: Optional[set] = None) -> Tuple[bool, Set[FailureTuple]]:
    """Identifies what objects are preventing serialization.

    Args:
        base_obj: Object to be serialized.
        name: Optional name of string.
        depth: Depth of the scope stack to walk through. Defaults to 3.

    Returns:
        bool: True if serializable.
        set[FailureTuple]: Set of unserializable objects.

    .. versionadded:: 1.1.0

    """
    colorama.init()
    top_level = False
    declaration = ""
    found = False
    if _failure_set is None:
        top_level = True
        _failure_set = set()
        declaration = f"Checking Serializability of {base_obj}"
        _printer.print("=" * min(len(declaration), 80))
        _printer.print(declaration)
        _printer.print("=" * min(len(declaration), 80))

        if name is None:
            name = str(base_obj)
    else:
        _printer.print(f"Serializing '{name}' {base_obj}...")
    try:
        cp.dumps(base_obj)
        return True, _failure_set
    except Exception as e:
        _printer.print(f"{colorama.Fore.RED}!!! FAIL{colorama.Fore.RESET} "
                       f"serialization: {e}",warning=True)
        found = True
        try:
            if depth == 0:
                _failure_set.add(FailureTuple(base_obj, name, _parent))
        # Some objects may not be hashable, so we skip adding this to the set.
        except Exception:
            pass

    if depth <= 0:
        return False, _failure_set

    # TODO: we only differentiate between 'function' and 'object'
    # but we should do a better job of diving into something
    # more specific like a Type, Object, etc.
    if inspect.isfunction(base_obj):
        _inspect_func_serialization(
            base_obj, depth=depth, parent=base_obj, failure_set=_failure_set)
    else:
        _inspect_generic_serialization(
            base_obj, depth=depth, parent=base_obj, failure_set=_failure_set)

    if not _failure_set:
        _failure_set.add(FailureTuple(base_obj, name, _parent))

    if top_level:
        print("=" * min(len(declaration), 80))
        if not _failure_set:
            _printer.print("Nothing failed the inspect_serialization test, though "
                  "serialization did not succeed.",warning=True)
        else:
            fail_vars = f"\n\n\t{colorama.Style.BRIGHT}" + "\n".join(
                str(k)
                for k in _failure_set) + f"{colorama.Style.RESET_ALL}\n\n"
            _printer.print(f"Variable: {fail_vars}was found to be non-serializable. "
                  "There may be multiple other undetected variables that were "
                  "non-serializable. ",warning=True)
            _printer.print("Consider either removing the "
                  "instantiation/imports of these variables or moving the "
                  "instantiation into the scope of the function/class. ",warning=True)
        _printer.print("If you have any suggestions on how to improve "
              "this error message, please reach out to the "
              "Ray developers on github.com/ray-project/ray/issues/",warning=True)
        _printer.print("=" * min(len(declaration), 80))
    return not found, _failure_set    


import pickle

def pickle_trick(obj, max_depth=10):
    output = {}

    if max_depth <= 0:
        return output

    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError) as e:
        failing_children = []
        
        if isinstance(obj, (list,tuple)):
            for it in obj:
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)        

        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)

        output = {
            "fail": obj, 
            "err": e, 
            "depth": max_depth, 
            "failing_children": failing_children
        }

    return output
