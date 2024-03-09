"""Module to define the Ref class and utility methods for working with it.

Ref's are designed to create adhoc links between the systems & components of the system, and their properties. This is useful for creating optimization problems, and for creating dynamic links between the state and properties of the system. Care must be used on the component objects changing of state, the recommened procedure is to copy your base system to prevent hysterisis and other issues, then push/pop X values through the optimization methods and change state upon success

Ref.component can be a Component,System or dictionary for referencing via keyword. Ref.key can be a string or a callable, if callable the ref will be evaluated as the result of the callable (allow_set=False), this renders the component unused, and disables the ability to set the value of the reference.
"""
import attrs
import inspect
import numpy as np
import collections
import scipy.optimize as sciopt
from contextlib import contextmanager
from engforge.properties import *


class RefLog(LoggingMixin):
    pass


log = RefLog()


def refset_input(refs, delta_dict):
    for k, v in delta_dict.items():
        refs[k].set_value(v)


def refset_get(refs,*args,**kw):
    return {k: refs[k].value(*args,**kw) for k in refs}


def f_root(ResRef: collections.OrderedDict, norm: dict = None):
    residual = [v.value() / (norm[k] if norm else 1) for k, v in ResRef.items()]
    res = np.array(residual)
    return res


def f_min(ResRef: collections.OrderedDict, norm: dict = None):
    res = [v.value() / (norm[k] if norm else 1) for k, v in ResRef.items()]
    ans = np.linalg.norm(res, 2)

    if ans < 1:
        return ans**0.5
    return ans


def eval_ref(canidate,*args,**kw):
    #print(f'eval_ref {canidate,args,kw}')
    if isinstance(canidate, Ref):
        return canidate.value(*args,**kw)
    elif callable(canidate):  
        o = canidate(*args,**kw)
        return o
    return canidate

def scale_val(val, mult=None, bias=None, power=None)->float:
    if any((mult,bias,power)):
        if power is not None:
            val = val ** power    
        if mult is not None:
            val = mult * val
        if bias is not None:
            val = bias + val
    return val


def maybe_ref(can,astype=None,mult=None,bias=None,power=None,*args,**kw):
    """returns the value of a ref if it is a ref, otherwise returns the value"""
    #print(f'maybe  {can,astype}')
    #print(can,astype,mult,bias,power,args,kw)
    if isinstance(can,Ref):
        val = can.value(*args,**kw)
        return scale_val(val,mult,bias,power)
    elif astype and isinstance(can,astype):
        ref = can.as_ref_dict() #TODO: optimize this
        return scale_val(ref.value(*args,**kw),mult,bias,power)
    return can

def maybe_attr_inst(can,astype=None):
    """returns the ref if is one otherwise convert it, otherwise returns the value"""
    if isinstance(can,Ref):
        return can
    elif astype and isinstance(can,astype):
        return astype.backref.handle_instance(can)
    return can


#Important State Preservation
#TODO: check for hidden X dependents / circular references ect.
#TODO: move to execution context
@contextmanager
def revert_X(system, refs, Xnext=None, pre_exec=True, post_exec=False):
    """
    Stores the _X parameter at present, the reverts to that state when done

    #TODO: add a system,ref.key pair based global storage, this will allow for the use of the same ref in different systems. This will allow set/value caching to speed up write-time from the current method of looking up value before write, which is faster than just writing on average.    
    """

    #print('revert_X',system,refs)

    X_now = Ref.refset_get(refs)

    #Change the current state
    if Xnext:
        Ref.refset_input(refs, Xnext)
        if pre_exec:
            system.pre_execute()
        if post_exec:
            system.post_execute()

    try:  # Change Variables To Input
        yield X_now
        
    finally:
        #revert and call pre/post execute
        Ref.refset_input(refs, X_now)
        if pre_exec:
            system.pre_execute()
        if post_exec:
            system.pre_execute()


#TODO: make global storage for Ref's based on the comp,key pair. This 
class Ref:
    """A way to create portable references to system's and their component's properties, ref can also take a key to a zero argument function which will be evald. This is useful for creating an adhoc optimization problems dynamically between variables. However use must be carefully controlled to avoid circular references, and understanding the safe use of the refset_input and refset_get methods (changing state on actual system).

    A dictionary can be used as a component, and the key will be used to access the value in the dictionary.

    The key can be a callable of (*args,**kw) only in which case Ref.value(*args,**kw) will take input, and the ref will be evaluated as the result of the callable (allow_set=False), this renders the component unused, and disables the ability to set the value of the reference.

    The ability to set this ref with another on key input allows for creating a reference that is identical to another reference except for the component provided if it is not None.
    """

    __slots__ = [
        "comp",
        "key",
        "use_call",
        "use_dict",
        "allow_set",
        "eval_f",
        "key_override",
        '_last'
    ]
    comp: "TabulationMixin"
    key: str
    use_call: bool
    use_dict: bool
    allow_set: bool
    eval_f: callable
    key_override: bool

    def __init__(self, comp, key, use_call=True, allow_set=True, eval_f=None):
        self.set(comp, key, use_call, allow_set, eval_f)

    def set(self, comp, key, use_call=True, allow_set=True, eval_f=None):
        
        #key can be a ref, in which case this ref will be identical to the other ref except for the component provided if it is not None
        if isinstance(key,Ref):
            self.__dict__.update(key.__dict__)
            if comp is not None:
                self.comp = comp
            return #a monkey patch

        self.comp = comp
        if isinstance(self.comp, dict):
            self.use_dict = True
        else:
            self.use_dict = False

        self.key_override = False
        if callable(key):
            self.key_override = True
            self.key = key #this should take have signature f(system,slv_info)
            self.use_call = False
            self.allow_set = False
            self.eval_f = eval_f
        else:
            self.key = key
            self.use_call = use_call
            self.allow_set = allow_set
            self.eval_f = eval_f

    def copy(self, **changes):
        """copy the reference, and make changes to the copy"""
        if "key" not in changes:
            changes["key"] = self.key
        if "comp" not in changes:
            changes["comp"] = self.comp
        copy = copy.copy(self)
        copy.set(**changes)
        return copy

    def value(self,*args,**kw):

        if self.comp and self.comp.log_level < 5:
            self.comp.msg(f"REF[get] {self.comp} {self.key}")

        if self.key_override:
            return self.key(*args,**kw)

        if self.use_dict:
            o = self.comp.get(self.key)
        else:
            o = getattr(self.comp, self.key)
        if self.use_call and callable(o):
            o = o()
        if self.eval_f:
            o = self.eval_f(o)
        self._last = o

        if self.comp and self.comp.log_level < 5:
            self.comp.msg(f"REF[get] {self.comp} {self.key} | got: {o}")

        return o

    def set_value(self, val):
        if self.allow_set:
            if self.value() != val: #this increases perf. by reducing writes
                if self.comp and self.comp.log_level < 10:
                    self.comp.msg(f"REF[set] {self} -> {val}")
                self._last = val
                return setattr(self.comp, self.key, val)
        else:
            raise Exception(f"not allowed to set value on {self.key}")

    def __str__(self) -> str:
        if self.use_dict:
            return f"REF[DICT.{self.key}]"
        if self.key_override:
            return f"REF[{self.comp.classname}.{self.key.__name__}]"
        return f"REF[{self.comp.classname}.{self.key}]"

    def __repr__(self) -> str:
        if self.key_override:
            return f"REF[{self.comp.classname}.{self.key.__name__}]"        
        return f"REF[{self.comp.classname}.{self.key}]"

    # Utilty Methods
    refset_get = refset_get
    refset_input = refset_input



























#### TODO: solve equillibrium case first using root solver for tough problems
# # make anonymous function
# def f_lin_slv(system, Xref: dict, Yref: dict, normalize=None,slv_info=None):
#     parms = list(Xref.keys())  # static x_basis
#     yparm = list(Yref.keys())
#     def f(x):  # anonymous function
#         # set state
#         for p, xi in zip(parms, x):
#             Xref[p].set_value(xi)
#         print(Xref,Yref)
#         grp = (yparm, x, normalize)
#         vals = [eval_ref(Yref[p],system,slv_info) / n for p, x, n in zip(*grp)]
#         return vals  # n sized normal residual vector
# 
#     return f
# 
# def refroot_solve(
#     system,
#     Xref: dict,
#     Yref: dict,
#     Xo=None,
#     normalize: np.array = None,
#     reset=True,
#     doset=True,
#     fail=True,
#     ret_ans=False,
#     ffunc=f_lin_slv,
#     **kw,
# ):
#     """find the input X to ensure the difference between two dictionaries of refernces, x references are changed in place, y will be solved to zero, options ensue for cases where the solution is not ideal
# 
#     :param Xref: dictionary of references to the x values
#     :param Yref: dictionary of references to the y values
#     :param Xo: initial guess for the x values as a list against Xref order, or a dictionary
#     :param normalize: a dictionary of values to normalize the x values by, list also ok as long as same length and order as Xref
#     :param reset: if the solution fails, reset the x values to their original state, if true will reset the x values to their original state on failure overiding doset.
#     :param doset: if the solution is successful, set the x values to the solution by default, otherwise follows reset, if not successful reset is checked first, then doset
#     """
#     parms = list(Xref.keys())  # static x_basis
#     if normalize is None:
#         normalize = np.ones(len(parms))
#     elif isinstance(normalize, (list, tuple, np.ndarray)):
#         assert len(normalize) == len(parms), "bad length normalize"
#     elif isinstance(normalize, (dict)):
#         normalize = np.array([normalize[p] for p in parms])
# 
#     # make objective function
#     f = ffunc(system, Xref, Yref, normalize)
# 
#     # get state
#     if reset:
#         x_pre = refset_get(Xref)  # record before changing
# 
#     if Xo is None:
#         Xo = [Xref[p].value() for p in parms]
# 
#     elif isinstance(Xo, dict):
#         Xo = [Xo[p] for p in parms]
# 
#     # solve
#     ans = sciopt.root(f, Xo, **kw)
#  
#     return process_ans(ans,parms,Xref,x_pre,doset,reset,fail,ret_ans)
# 
