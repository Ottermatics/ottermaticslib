from engforge.system_reference import *
from engforge.logging import LoggingMixin
#from engforge.execution_context import *

import fnmatch

class SolverUtilLog(LoggingMixin): pass
log = SolverUtilLog() 

# Objective functions & Utilities
def f_lin_min(system, Xref, Yref,weights=None, *args, **kw):
    """
    Creates an anonymous function with stored references to system, Yref, weights, that returns a scipy optimize friendly function of (x, Xref, *a, **kw) x which corresponds to the order of Xref dicts, and the other inputs are up to application.

    :param system: the system object
    :param Xref: a dictionary of reference values for X
    :param Yref: a dictionary of reference values for Y
    :param weights: optional weights for Yref
    :param args: additional positional arguments
    :param kw: additional keyword arguments

    :return: the anonymous function
    """
    from engforge.problem_context import ProblemExec

    #TODO: move these to problem context!!!
    mult_pos = kw.pop("mult_pos", 1)
    exp_pos = kw.pop("exp_pos", 1)
    mult_neg = kw.pop("mult_neg", 1)
    exp_neg = kw.pop("exp_neg", 1)
    gam = norm_base = kw.pop("norm_base", 1)
    inputs = [mult_neg,exp_neg,mult_pos,exp_pos,gam,norm_base]
    is_lin = all(v==1 for v in inputs)

    solver_ref =  system.collect_solver_refs(conv=False)
    solver_types = solver_ref.get('type',{}).get('solver',{})
    base_dict = {'system':system,'Yref':Yref,'weights':weights,'args':args,'kw':kw}
    xkey = "_".join(Xref.keys())
    ykey = "_".join(Yref.keys())
    alias_name = kw.pop("alias_name", f'min_X_{xkey}_Y_[{ykey}]')

    parms = list(Xref.keys())  # static x_basis
    yparm = list(Yref.keys())
    weights = weights if weights is not None else np.ones(len(Yref))

    def f(x, *rt_a,**rt_kw):
        # anonymous function
        
        Xnext = {p:xi for p, xi in zip(parms, x)}

        if rt_a or rt_kw:
            slv_info = base_dict.copy()
            slv_info.update({'rt_a':rt_a,'kw':rt_kw})
        else:
            slv_info = base_dict

        #with revert_X(system,Xref,Xnext=Xnext):
        #Enter existing problem context
        with ProblemExec(system,{},Xnew=Xnext,ctx_fail_new=True) as exc:
            grp = (yparm, weights)
            vals,pos,neg = [],[],[]
            for p,n in zip(*grp):
                ty = solver_types.get(p,None)
                if ty is None or ty.solver.kind == 'min':
                    arry = pos
                elif ty.solver.kind == 'max':
                    arry = neg
                else:
                    system.warning(f'non minmax obj: {p} {ty.solver.kind}')

                ref = Yref[p]
                val = eval_ref(ref,system,slv_info) * n
                arry.append( val )
                vals.append( val )

            if not is_lin:
                ps = mult_pos*np.array( pos )**exp_pos
                #Min-Max Logic
                if neg:
                    ns = mult_neg*np.array( neg )**exp_neg
                    ns = np.sum(ns)
                    out= np.sum(ps)**gam - np.sum(ns)**gam
                else:
                    out = mult_pos*np.sum(ps)**gam
                
                if system.log_level < 5:
                    system.debug(f'obj {alias_name}: {x} -> {vals}')

                return out  # n sized normal residual vector
            else:
                #the faster linear case
                if neg:
                    return np.sum(np.array( pos )) - np.sum(np.array( neg ))
                else:
                    return np.sum(np.array( pos ))
                
    f.__name__ = f'min_Y_{"_".join(Yref.keys())}_X_{"_".join(Xref.keys())}]'

    return f

# signature in solve: refmin_solve(system,system,Xref)
def objectify(function,system,Xrefs,solver_info=None,*args,**kwargs):
    """converts a function f(system,slv_info:dict) into a function that safely changes states to the desired values and then runs the function. A function is returend as f(x,*args,**kw)"""
    from engforge.problem_context import ProblemExec
    base_dict = dict(system=system,Xrefs=Xrefs,args=args,**kwargs)
    alias_name = kwargs.pop("alias_name", f'{function.__name__}_obj')

    def f_obj(x,*rt_args,**rt_kwargs):
        new_state = {p: x[i] for i, p in enumerate(Xrefs)}
        #with revert_X(system, Xrefs, Xnext=new_state ) as x_prev:
        
        #Enter existing problem context
        with ProblemExec(system,{},Xnew=new_state,ctx_fail_new=True) as exc:
            #print(locals()['solver_info'])
            updtinfo = base_dict.copy()
            updtinfo.update(x=x,rt_args=rt_args, **rt_kwargs)
            solver_info = locals().get('solver_info',updtinfo)

            out= function(system,solver_info)
            if system.log_level < 5:
                system.msg(f'obj {alias_name}: {x} -> {out}')
            return out        
    
    if system.log_level < 3:
        system.msg(f'obj setup {function} - > {f_obj}')
        system.msg(inspect.getsource(function))

    fo = lambda x,*a,**kw: f_obj(x,solver_info,*a,**kw)
    fo.__name__ =f'OBJ_{function.__name__}'
    return fo

def secondary_obj(
    obj_f, system,Xrefs,normalize=None,base_func=f_lin_min,*args,**kwargs
):  
    """modifies an objective function with a secondary function that is only considered when the primary function is minimized."""
    from engforge.problem_context import ProblemExec
    parms = list(Xrefs.keys())  # static x_basis
    base_dict = dict(system=system,args=args,Xrefs=Xrefs,**kwargs)
    alias_name = kwargs.pop("alias_name", f'{obj_f.__name__}_scndry')

    def f(x,*rt_args,**rt_kwargs): 

        new_state = {p: x[i] for i, p in enumerate(Xrefs)}
        base_call = base_func(system, Xrefs, normalize)        
        #with revert_X(system, Xrefs,Xnext=new_state) as x_prev:

        #Enter existing problem context
        with ProblemExec(system,{},Xnew=new_state,ctx_fail_new=True) as exc:
            A = base_call(x)
            solver_info = base_dict.copy()
            solver_info.update(x=x,Xrefs=Xrefs,normalize=normalize,rt_args=rt_args, **rt_kwargs)

            out =  A * (1 + obj_f(system, solver_info))
            if system.log_level < 5:
                system.msg(f'obj {alias_name}: {x} -> {out}')
            return out
        
    if system.log_level < 18:
        system.debug(f'secondary setup {obj_f} - > {f}')
        system.debug(inspect.getsource(function))        

    return f

def ref_to_val_constraint(system,Xrefs,parm_ref,kind,val,*args,**kwargs):
    """takes a parameter reference and a value and returns a function that can be used as a constraint for min/max cases. The function will be a function of the system and the info dictionary. The function will return the difference between the parameter value and the value.
    """
    
    info = {'system':system,'Xrefs':Xrefs,'parm_ref':parm_ref,'kind':kind,'val':val,'args':args,'kwargs':kwargs}
    p = parm_ref
    if isinstance(val,Ref):
        if kind == 'min':
            fun = lambda system,info: p.value(system,info) - val.value(system,info)
        else:
            fun = lambda system,info: val.value(system,info) - p.value(system,info)
        fun.__name__ = f'REF{val.comp}.{kind}.{p.key}'
        ref = Ref(val.comp,fun)
    #Function Case
    elif callable(val):
        #print('ref to val con', val,kind,p)
        if kind == 'min':
            fun = lambda system,info: p.value(system,info) - val(system,info)
        else:
            fun = lambda system,info: val(system,info) - p.value(system,info)
        fun.__name__ = f'REF.{kind}.{val.__name__}'
        ref = Ref(p.comp,fun) #comp shouldn't matter
    elif isinstance(val,(int,float)):
        if kind == 'min':
            fun = lambda system,info: p.value(system,info) - val
        else:
            fun = lambda system,info: val - p.value(system,info)
        fun.__name__ = f'REF.{kind}.{val}'
        ref = Ref(p.comp,fun) #comp shouldn't matter
    else:
        raise ValueError(f"bad constraint value: {val}")
    
    #Make Objective
    return create_constraint(system,Xrefs,'ineq',ref,*args,**kwargs)

def create_constraint(
    system,Xref, contype: str, ref, con_args=None,*args, **kwargs
):
    """creates a constraint with bounded solver input from a constraint definition in dictionary with type and value. If value is a function it will be evaluated with the extra arguments provided. If parm is None, then the constraint is assumed to be not in reference to the system x parameters, otherwise lookups are made to that parameter.
    
    Creates F(x_solver:array) such that the current parameters of system are reverted to after the function has returned, which is used directly by SciPy's optimize.minimize
    
    """
    assert contype in (
        "eq",
        "ineq",
    ), f"bad constraint type: {contype}"

    # its a function
    _fun = lambda *args,**kw: ref.value(*args,**kw)
    _fun.__name__ = f'const_{contype}_{ref.comp.classname}_{ref.key}'
    fun = objectify(_fun,system,Xref,*args,**kwargs)
    cons = {"type": contype, "fun": fun}
    if con_args:
        cons['args'] = con_args
    return cons

def misc_to_ref(system,val,*args,**kwargs):
    """takes a parameter reference and a value and returns a function that can be used as a constraint for min/max cases. The function will be a function of the system and the info dictionary. The function will return the difference between the parameter value and the value.
    """
    if isinstance(val,Ref):
        #fun = lambda *a,**kw:val.value()
        ref = Ref(val.comp,val)
    #Function Case
    elif callable(val):
        fun = lambda system,info: val(system,info)
        fun.__name__ = val.__name__
        ref = Ref(None,fun) #comp shouldn't matter
    elif isinstance(val,(int,float)):
        fun = lambda system,info: val
        fun.__name__ = f'const_{str(val)}'
        ref = Ref(None,fun) #comp shouldn't matter
    else:
        raise ValueError(f"bad constraint value: {val}")
    
    return ref


# Reference Jacobean Calculation
#TODO: hessian ect...
def calcJacobean(
    sys, Yrefs: dict, Xrefs: dict, X0: dict = None, pct=0.001, diff=0.0001
):
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
        refset_input(Xrefs, X0)

        rows = []
        dxs = []
        Fbase = refset_get(Yrefs)
        for k, v in Xrefs.items():
            x = v.value()
            if not isinstance(x, (float, int)):
                sys.warning(f"parm: {k} is not numeric {x}, skpping")
                continue

            if abs(x) > pct:
                new_x = x * (1 + pct) + diff
            else:
                new_x = x * (1.1) + diff
            dx = new_x - x
            #print(dx, new_x, x)
            dxs.append(dx)

            v.set_value(new_x)  # set delta
            sys.pre_execute()

            F_ = refset_get(Yrefs)
            Fmod = [(F_[k] - fb) / dx for k, fb in Fbase.items()]

            rows.append(Fmod)
            v.set_value(x)  # reset value

    return np.column_stack(rows)


#TODO: integrate / merge with ProblemExec (all below)
def refmin_solve(
    system,
    Xref: dict,
    Yref: dict,
    Xo=None,
    weights: np.array = None,
    reset=True,
    doset=True,
    fail=True,
    ret_ans=False,
    ffunc=f_lin_min,
    **kw,
):
    """minimize the difference between two dictionaries of refernces, x references are changed in place, y will be solved to zero, options ensue for cases where the solution is not ideal

    :param Xref: dictionary of references to the x values, or independents
    :param Yref: dictionary of references to the value of objectives to be minimized
    :param Xo: initial guess for the x values as a list against Xref order, or a dictionary
    :param weights: a dictionary of values to weights the x values by, list also ok as long as same length and order as Xref
    :param reset: if the solution fails, reset the x values to their original state, if true will reset the x values to their original state on failure overiding doset.
    :param doset: if the solution is successful, set the x values to the solution by default, otherwise follows reset, if not successful reset is checked first, then doset
    #TODO: add 
    """
    parms = list(Xref.keys())  # static x_basis
    
    norm_x,weights = handle_normalize(weights,Xref,Yref)

    # make objective function
    Fc = ffunc(system,Xref,Yref,weights)
    Fc.__name__ = ffunc.__name__

    # get state
    if reset:
        x_pre = refset_get(Xref)  # record before changing

    if Xo is None:
        Xo = [Xref[p].value() for p in parms]

    elif isinstance(Xo, dict):
        Xo = [Xo[p] for p in parms]

    # TODO: IO for jacobean and state estimations (complex as function definition, requires learning)
    system.debug(f'minimize! {Fc.__name__,Xo,parms,kw}')
    kw.pop('info',None)
    ans = sciopt.minimize(Fc, Xo, **kw)
    return process_ans(ans,parms,Xref,x_pre,doset,reset,fail,ret_ans)


def handle_normalize(norm,Xref,Yref):
    parms = list(Xref.keys())  # static x_basis
    if norm is None:
        normX = np.ones(len(parms))
        normY = np.ones(len(Yref))
    elif isinstance(norm, (list, tuple, np.ndarray)):
        assert len(norm) == len(parms), "bad length-X norm input"
        normX = np.array(norm)
        normY = np.ones(len(Yref)) #default to 1
    elif isinstance(norm, (dict)):
        normX = np.array([norm[p] if p in norm else 1 for p in parms])
        normY = np.array([norm[p] if p in norm else 1 for p in Yref])

    return normX,normY  


def process_ans(ans,parms,Xref,x_pre,doset,reset,fail,ret_ans):
    if ret_ans:
        return ans

    if ans.success:
        ans_dct = {p: a for p, a in zip(parms, ans.x)}
        if doset:
            Ref.refset_input(Xref, ans_dct)
        elif reset:
            Ref.refset_input(Xref, x_pre)
        return ans_dct

    else:
        min_dict = {p: a for p, a in zip(parms, ans.x)}
        if reset:
            Ref.refset_input(Xref, x_pre)
        elif doset:
            Ref.refset_input(Xref, min_dict)
            return min_dict
        if fail:
            raise Exception(f"failed to solve {ans.message}")
        return min_dict








#Parsing Utilities
def arg_parm_compare(parm,seq):
    if '*' in seq:
        return fnmatch.fnmatch(parm,seq)
    else:
        return parm == seq
    
def str_list_f(out:list,sep=','):
    if isinstance(out,str):
        out = out.split(sep)
    return out

def ext_str_list(extra_kw,key,default=None):
    if key in extra_kw:
        out = extra_kw[key]
        return str_list_f(out)
    else:
        out = default
    return out

def filt_combo_vars(parm,inst,extra_kw=None,combos_in=None):
    from engforge.attr_solver import SolverInstance
    from engforge.attr_dynamics import IntegratorInstance
    from engforge.attr_signals import SignalInstance
    #not considered
    
    if not isinstance(inst,(SolverInstance,IntegratorInstance,SignalInstance)):
        return True
    
    groups = ext_str_list(extra_kw,'combos',None)
    if groups is None:
        #print('no combo args',inst)
        return None #"no groups"
    igngrp = ext_str_list(extra_kw,'ign_combos',None)
    onlygrp = ext_str_list(extra_kw,'only_combos',None)

    if not combos_in:
        combos = str_list_f(getattr(inst,'combos', ''))
    else:
        combos = str_list_f(combos_in)
    
    if not combos:
        log.info(f'no combos for {parm} {combos} {groups}')
        return None #no combos to filter to match, its permanent
    
    for parm in combos:
        initial_match = [grp for grp in groups if arg_parm_compare(parm,grp)]
        if not any(initial_match):
            if log.log_level < 3:
                log.msg(f'skip {parm}: nothing ')
            continue #not even 
        if onlygrp and not any(arg_parm_compare(parm,grp) for grp in onlygrp):
            if log.log_level < 3:
                log.msg(f'skip {parm} not in {onlygrp}')
            continue
        if igngrp and any(arg_parm_compare(parm,grp) for grp in igngrp):
            if log.log_level < 3:
                log.msg(f'skip {parm} in {igngrp}')
            continue
        return True
    
    return False    
    
def filt_parm_vars(parm,inst,extra_kw=None):
    from engforge.attr_solver import SolverInstance
    from engforge.attr_dynamics import IntegratorInstance
    from engforge.attr_signals import SignalInstance
    
    #not considered
    if not isinstance(inst,(SolverInstance,IntegratorInstance,SignalInstance)):
        return True
    
    groups = ext_str_list(extra_kw,'slv_vars',None)
    if groups is None:
        return True #"no groups, assume yes"
    igngrp = ext_str_list(extra_kw,'ign_vars',None)
    onlygrp = ext_str_list(extra_kw,'only_vars',None)

    initial_match = [grp for grp in groups if arg_parm_compare(parm,grp)]
    if not any(initial_match):
        if log.log_level < 3:
            log.msg(f'skip {parm}: nothing')
        return False
    if onlygrp and not any(arg_parm_compare(parm,grp) for grp in onlygrp):
        if log.log_level < 3:
            log.msg(f'skip {parm} not in {onlygrp}')
        return False    
    if igngrp and any(arg_parm_compare(parm,grp) for grp in igngrp):
        if log.log_level < 3:
            log.msg(f'skip {parm} in {igngrp}')
        return False
    return True

def filt_active(parm,inst,extra_kw=None,dflt=False):
    from engforge.attr_solver import SolverInstance
    from engforge.attr_dynamics import IntegratorInstance
    from engforge.attr_signals import SignalInstance
    
    #not considered
    if not isinstance(inst,(SolverInstance,IntegratorInstance,SignalInstance)):
        return True

    activate = ext_str_list(extra_kw,'activate',[])
    deactivate = ext_str_list(extra_kw,'deactivate',[])  

    act = inst.is_active(dflt) #default is inclusion boolean 
    
    #check for possibilities of activation
    if not act and not activate:
        if log.log_level < 3:
            log.msg(f'skip {parm}: not active')
        return False #shortcut
    if activate and any(arg_parm_compare(parm,grp) for grp in activate):
        if log.log_level < 3:
            log.msg(f'{parm} activated!')
        return True
    if deactivate and any(arg_parm_compare(parm,grp) for grp in deactivate):
        if log.log_level < 3:
            log.msg(f'{parm} deactivated!')
        return False
    log.msg(f'{parm} is {act}!')
    return act










# def solve_root(
#     self, Xref, Yref, Xreset, parms, output=None, fail=True, **kw
# ):
#     """
#     Solve the root problem using the given parameters.
# 
#     :param Xref: The reference input values.
#     :param Yref: The reference output values.
#     :param Xreset: The reset input values.
#     :param parms: The list of parameter names.
#     :param output: The output dictionary to store the results. (default: None)
#     :param fail: Flag indicating whether to raise an exception if the solver doesn't converge. (default: True)
#     :param kw: Additional keyword arguments.
#     :return: The output dictionary containing the results.
#     :rtype: dict
#     """
#     if output is None:
#         output = {
#             "Xstart": Xreset,
#             "parms": parms,
#             "Xans": None,
#             "success": None,
#         }
# 
#     assert len(Xref) == len(Yref), "Xref and Xreset must have the same length"
# 
#     self._ans = refroot_solve(self, Xref, Yref, ret_ans=True, **kw)
#     output["ans"] = self._ans
#     if self._ans.success:
#         # Set Values
#         Xa = {p: self._ans.x[i] for i, p in enumerate(parms)}
#         output["Xans"] = Xa
#         Ref.refset_input(Xref, Xa)
#         self.pre_execute()
#         self._converged = True
#         output["success"] = True
#     else:
#         Ref.refset_input(Xref, Xreset)
#         self.pre_execute()
#         self._converged = False
#         if fail:
#             raise Exception(f"solver didnt converge: {self._ans}")
#         output["success"] = False
# 
#     return output
    








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
