"""Module to define the Ref class and utility methods for working with it."""
import attrs
import numpy as np
import collections
import scipy.optimize as sciopt
from engforge.properties import *



class RefLog(LoggingMixin):
    pass

log = RefLog()

def refset_input(refs,delta_dict):
        for k, v in delta_dict.items():
            refs[k].set_value(v)

def refset_get(refs):
    return {k: refs[k].value() for k in refs}

def f_root(ResRef:collections.OrderedDict,norm:dict=None):
    residual = [v.value()/(norm[k] if norm else 1) for k,v in ResRef.items()]
    res = np.array(residual)
    return res

def f_min(ResRef:collections.OrderedDict,norm:dict=None):
    res = [v.value()/(norm[k] if norm else 1) for k,v in ResRef.items()]
    ans = np.linalg.norm(res, 2)

    if ans < 1:
        return ans**0.5
    return ans

#make anonymous function
def f_lin_slv(system,Xref:dict,Yref:dict,normalize=None):
    parms = list(Xref.keys()) #static x_basis

    def f(x): #anonymous function
        #set state
        for p,xi in zip(parms,x):
            Xref[p].set_value(xi)
        grp = (parms,x,normalize)
        vals = [(Yref[p].value()/n) for p,x,n in zip(*grp)]
        return vals #n sized normal residual vector
    
    return f

def f_lin_min(system,Xref:dict,Yref:dict,normalize=None):
    parms = list(Xref.keys()) #static x_basis

    def f(x): #anonymous function
        #set state
        for p,xi in zip(parms,x):
            Xref[p].set_value(xi)
        grp = (parms,x,normalize)
        vals = [(Yref[p].value()/n)**2 for p,x,n in zip(*grp)]
        rs = np.array(vals)
        out = np.sum(rs)**0.5        
        return out #n sized normal residual vector
    
    return f


def refmin_solve(system,Xref:dict,Yref:dict,Xo=None,normalize:np.array=None,reset=True,doset=True,fail=True,ret_ans=False,ffunc=f_lin_min,**kw):
    """minimize the difference between two dictionaries of refernces, x references are changed in place, y will be solved to zero, options ensue for cases where the solution is not ideal
    
    :param Xref: dictionary of references to the x values
    :param Yref: dictionary of references to the y values
    :param Xo: initial guess for the x values as a list against Xref order, or a dictionary
    :param normalize: a dictionary of values to normalize the x values by, list also ok as long as same length and order as Xref 
    :param reset: if the solution fails, reset the x values to their original state, if true will reset the x values to their original state on failure overiding doset. 
    :param doset: if the solution is successful, set the x values to the solution by default, otherwise follows reset, if not successful reset is checked first, then doset
    """
    parms = list(Xref.keys()) #static x_basis
    if normalize is None:
        normalize = np.ones(len(parms))
    elif isinstance(normalize,(list,tuple,np.ndarray)):
        assert len(normalize)==len(parms), "bad length normalize"
    elif isinstance(normalize,(dict)):
        normalize = np.array([normalize[p] for p in parms])

    #make objective function
    f = ffunc(system,Xref,Yref,normalize)

    #get state
    if reset:
        x_pre = refset_get(Xref) #record before changing

    if Xo is None:
        Xo = [Xref[p].value() for p in parms]

    elif isinstance(Xo,dict):
        Xo = [Xo[p] for p in parms]

    #solve
    #print(Xref,Yref,Xo,normalize)
    #TODO: IO for jacobean and state estimations (complex as function definition, requires learning)
    ans = sciopt.minimize(f,Xo,**kw)
    if ret_ans:
        return ans
    
    if ans.success:
        ans_dct = {p:a for p,a in zip(parms,ans.x)}
        if doset:
            refset_input(Xref,ans_dct)        
        if reset: 
            refset_input(Xref,x_pre)
        return ans_dct
        
    else:
        min_dict = {p:a for p,a in zip(parms,ans.x)}
        if reset: 
            refset_input(Xref,x_pre)
        if doset:
            refset_input(Xref,min_dict)            
            return min_dict
        if fail:
            raise Exception(f"failed to solve {ans.message}")
        return min_dict
    
def refroot_solve(system,Xref:dict,Yref:dict,Xo=None,normalize:np.array=None,reset=True,doset=True,fail=True,ret_ans=False,ffunc=f_lin_slv,**kw):
    """find the input X to ensure the difference between two dictionaries of refernces, x references are changed in place, y will be solved to zero, options ensue for cases where the solution is not ideal
    
    :param Xref: dictionary of references to the x values
    :param Yref: dictionary of references to the y values
    :param Xo: initial guess for the x values as a list against Xref order, or a dictionary
    :param normalize: a dictionary of values to normalize the x values by, list also ok as long as same length and order as Xref 
    :param reset: if the solution fails, reset the x values to their original state, if true will reset the x values to their original state on failure overiding doset. 
    :param doset: if the solution is successful, set the x values to the solution by default, otherwise follows reset, if not successful reset is checked first, then doset
    """
    parms = list(Xref.keys()) #static x_basis
    if normalize is None:
        normalize = np.ones(len(parms))
    elif isinstance(normalize,(list,tuple,np.ndarray)):
        assert len(normalize)==len(parms), "bad length normalize"
    elif isinstance(normalize,(dict)):
        normalize = np.array([normalize[p] for p in parms])

    #make objective function
    f = ffunc(system,Xref,Yref,normalize)
    
    #get state
    if reset:
        x_pre = refset_get(Xref) #record before changing

    if Xo is None:
        Xo = [Xref[p].value() for p in parms]

    elif isinstance(Xo,dict):
        Xo = [Xo[p] for p in parms]

    #solve
    ans = sciopt.root(f,Xo,**kw)
    
    if ret_ans:
        return ans
    
    if ans.success:
        ans_dct = {p:a for p,a in zip(parms,ans.x)}
        if doset:
            refset_input(Xref,ans_dct)        
        if reset: 
            refset_input(Xref,x_pre)
        return ans_dct
        
    else:
        min_dict = {p:a for p,a in zip(parms,ans.x)}
        if reset: 
            refset_input(Xref,x_pre)
        if doset:
            refset_input(Xref,min_dict)            
            return min_dict
        if fail:
            raise Exception(f"failed to solve {ans.message}")
        return min_dict    
    
#Reference Jacobean Calculation
def calcJacobean(sys,Yrefs:dict,Xrefs:dict,X0:dict=None,pct=0.001,diff=0.0001):
    """
    returns the jacobiean by modifying X' <= X*pct + diff and recording the differences. When abs(x) < pct x' = x*1.1 + diff.
    
    jacobean will be ordered by Xrefs/Yrefs, so use ordered dict to keep order
    """
    

    if X0 is None:
        X0 = refset_get(Xrefs)

    assert len(Xrefs) == len(X0)
    assert len(Yrefs) >= 1

    with sys.revert_X(refs=Xrefs):
        # initalize here
        refset_input(Xrefs,X0)

        rows = []
        dxs = []
        Fbase = refset_get(Yrefs)
        for k, v in Xrefs.items():
            x = v.value()
            if not isinstance(x,(float,int)):
                sys.warning(f'parm: {k} is not numeric {x}, skpping')
                continue
                
            if abs(x) > pct:
                new_x = x * (1 + pct) + diff
            else:
                new_x = x * (1.1) + diff
            dx = new_x - x
            print(dx,new_x,x)
            dxs.append(dx)

            v.set_value(new_x) #set delta
            sys.pre_execute()

            F_ = refset_get(Yrefs)
            Fmod = [(F_[k]-fb)/dx for k,fb in Fbase.items()]

            rows.append(Fmod)
            v.set_value(x) #reset value

    return np.column_stack(rows)    

class Ref:
    """A way to create portable references to system's and their component's properties, ref can also take a key to a zero argument function which will be evald,
    
    A dictionary can be used
    """

    __slots__ = ["comp", "key", "use_call",'use_dict','allow_set','eval_f','key_override']
    comp: "TabulationMixin"
    key: str
    use_call: bool
    use_dict: bool
    allow_set: bool
    eval_f: callable
    key_override: bool

    def __init__(self,component, key, use_call=True,allow_set=True,eval_f=None):
        self.comp = component
        if isinstance(self.comp,dict):
            self.use_dict = True
        else:
            self.use_dict = False
        
        self.key_override = False
        if callable(key):
            self.key_override = True
            self.key = key
            self.use_call = False
            self.allow_set = False
            self.eval_f = None          
        else:
            self.key = key
            self.use_call = use_call
            self.allow_set = allow_set
            self.eval_f = eval_f

    def value(self):
        if self.key_override:
            return self.key(self.comp)
        
        if self.use_dict:
            o = self.comp.get(self.key)
        else:
            o = getattr(self.comp, self.key)
        if self.use_call and callable(o):
            o = o()
        if self.eval_f:
            return self.eval_f(o)
        return o

    def set_value(self, val):
        if self.allow_set:
            return setattr(self.comp, self.key, val)
        else:
            raise Exception(f'not allowed to set value on {self.key}')

    def __str__(self) -> str:
        if self.use_dict:
            return f"REF[DICT.{self.key}]"
        return f"REF[{self.comp.classname}.{self.key}]"
    
    def __repr__(self) -> str:
        return f"REF[{self.comp.classname}.{self.key}]"

    #Utilty Methods 
    refset_get = refset_get
    refset_input = refset_input
    refmin_solve = refmin_solve
    



