from engforge.system_reference import *

import fnmatch

def _parm_compare(parm,seq):
    print(parm,seq)
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

    if not combos_in and hasattr(inst,'combos'):
        combos = inst.combos
    else:
        combos = str_list_f(combos_in)
    
    if not combos:
        return None #no combos to filter to match, its permanent
    
    for parm in combos:
        initial_match = [grp for grp in groups if _parm_compare(parm,grp)]
        if not any(initial_match):
            continue
        elif onlygrp and not any(_parm_compare(parm,grp) for grp in onlygrp):
            continue
        elif igngrp and any(_parm_compare(parm,grp) for grp in igngrp):
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

    initial_match = [grp for grp in groups if _parm_compare(parm,grp)]
    if not any(initial_match):
        return False
    elif onlygrp and not any(_parm_compare(parm,grp) for grp in onlygrp):
        return False    
    elif igngrp and any(_parm_compare(parm,grp) for grp in igngrp):
        return False
    
    return True

def filt_active(parm,inst,extra_kw=None):
    from engforge.attr_solver import SolverInstance
    from engforge.attr_dynamics import IntegratorInstance
    from engforge.attr_signals import SignalInstance
    
    #not considered
    if not isinstance(inst,(SolverInstance,IntegratorInstance,SignalInstance)):
        return True

    activate = ext_str_list(extra_kw,'activate',[])
    deactivate = ext_str_list(extra_kw,'deactivate',[])  

    act = inst.is_active()
    if not act and not activate:
        return False #shortcut
    
    elif activate and any(_parm_compare(parm,grp) for grp in activate):
        return True
      
    elif deactivate and any(_parm_compare(parm,grp) for grp in deactivate):
        return False
    
    return True


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



#Function assembly
def sys_solver_variables(system,sys_refs,extra_kw=None,**kw):
    """gathers variables from solver vars, and attempts to locate any input_vars to add as well. use exclude_vars to eliminate a variable from  the solver
    
    #TODO: add combo parsing
    """
    extra_kw = extra_kw or {}

    slv_inst = sys_refs.get('type',{}).get('solver',{})
    timz_inst = sys_refs.get('type',{}).get('time',{})
    
    sys_refs = sys_refs.get('attrs',{}) if 'attrs' in sys_refs else sys_refs
    
    dyns = sys_refs.get('dynamics.state',{})
    timz = {timz_inst[k].solver.parameter: v for k,v in sys_refs.get('time.parm',{}).items()}
    # vars = {slv_inst[k].var.key: v for k,v in sys_refs.get('solver.var',{}).items()}
    vars = {k: v for k,v in sys_refs.get('solver.var',{}).items()}


    
    out = dict(dynamics=dyns,integration=timz,variables=vars)
    
    if 'as_set' in kw and kw['as_set']==True:
        vars = set.union(*(set(v) for k,v in out.items()))
        return vars
    
    if 'as_flat' in kw and kw['as_flat']==True:
        flt = {}
        for k,v in out.items():
            #print(k,v)
            flt.update(v)
        return flt
    return out 





def sys_solver_objectives(system,sys_refs,Xrefs,extra_kw=None,add_obj=None,rmv_obj=None,**kw):
    """gathers variables from solver vars, and attempts to locate any input_vars to add as well. use exclude_vars to eliminate a variable from  the solver
    
    #TODO: add combo parsing
    """
    
    #Convert result per kind of objective (min/max ect)
    objs = sys_refs.get('attrs',{}).get('solver.obj',{})
    return {k:v for k,v in objs.items()}
    #return {k:v for k,v in objs.items() if not extra_kw or filt_combo_vars(k,v,extra_kw)}


def sys_solver_constraints( system, sys_refs, Xrefs, add_con=None,combo_filter=True, *args, **kw):
    """formatted as arguments for the solver
    #TODO: add combo parsing
    """
    extra_kw = kw.get('extra_kw')
    deactivated = ext_str_list(extra_kw,'deactivate',[]) if 'deactivate' in extra_kw and extra_kw['deactivate'] else []
    activated = ext_str_list(extra_kw,'activate',[]) if 'activate' in extra_kw and  extra_kw['activate'] else []

    slv_inst = sys_refs.get('type',{}).get('solver',{})
    sys_refs = sys_refs.get('attrs',{})

    if add_con is None:
        add_con = {}
    

    #The official definition of X parameter order
    #
    Xparms = list(Xrefs)

    # constraints lookup
    bnd_list = [[None, None]] * len(Xparms)
    con_list = []
    con_info = [] #names of constraints 
    constraints = {"constraints": con_list, "bounds": bnd_list,"info":con_info}
    

    if isinstance(add_con, dict):
        # Remove None Values
        nones = {k for k, v in add_con.items() if v is None}
        for ki in nones:
            constraints.pop(ki, None)
        assert all(
            [callable(v) for k, v in add_con.items()]
        ), f"all custom input for constraints must be callable with X as argument"
        constraints["constraints"].extend(
            [v for k, v in add_con.items() if v is not None]
        )

    if add_con is False:
        constraints = {} #youre free!
        return constraints
    
    

    # Add Constraints
    ex_arg = {"con_args": (),**kw}
    #Variable limit (function -> ineq, numeric -> bounds)
    for slvr, ref in sys_refs.get('solver.var',{}).items():
        slv = slv_inst[slvr]
        slv_constraints = slv.constraints
        system.debug(f'constraints {slvr} {slv_constraints}')
        for ctype in slv_constraints:
            cval = ctype['value']
            kind = ctype['type']       
            parm = ctype['parm']
            system.debug(f'const: {slvr} {ctype}')
            if cval is not None and slvr in Xparms:
                
                combos = None
                if 'combos' in ctype:
                    combos = ctype['combos']
                    combo_parm = ctype['combo_parm']
                    active = ctype.get('active',True)
                    in_activate = any([_parm_compare(combo_parm,v) for v in activated]) if activated else False
                    in_deactivate = any([_parm_compare(combo_parm,v) for v in deactivated]) if deactivated else False

                    #Check active or activated
                    if not active and not activated:
                        system.msg(f'skip con: inactive {parm} {slvr} {ctype}')
                        continue
                    elif not active and not in_activate:
                        system.msg(f'skip con: inactive {parm} {slvr} {ctype}')
                        continue

                    elif active and in_deactivate:
                        system.msg(f'skip con: deactivated {parm} {slvr} ')
                        continue

                    

                if combos and combo_filter:
                    filt = filt_combo_vars(combo_parm,slv, extra_kw,combos)
                    if not filt: 
                        system.info(f'filtering constraint={filt} {parm} {slv} {ctype} | {combos} {ext_str_list(extra_kw,"combos",None)}')                        
                        continue

                system.debug('adding var constraint',parm,slv,ctype,combos) 

                x_inx = Xparms.index(slvr)            
                #print(cval,kind,parm)
                if (
                    kind in ("min", "max")
                    and slvr in Xparms
                    and isinstance(cval, (int, float))
                ):
                    minv, maxv = bnd_list[x_inx]
                    bnd_list[x_inx] = [
                        cval if kind == "min" else minv,
                        cval if kind == "max" else maxv,
                        ]
                #add the bias of cval to the objective function
                elif kind in ('min','max') and slvr in Xparms:
                    parmref = Xrefs[slvr]
                    #Ref Case
                    cval = ref_to_val_constraint(system,Xrefs,parmref,kind,cval,*args,**kw)
                    con_info.append(f'val_{ref.comp.classname}_{kind}_{slvr}')
                    con_list.append(cval)
                else:
                    log.warning(f"bad constraint: {cval} {kind} {slvr}")

    # Add Constraints
    for slvr, ref in sys_refs.get('solver.ineq',{}).items():
        slv = slv_inst[slvr]
        slv_constraints = slv.constraints
        for ctype in slv_constraints:
            cval = ctype['value']
            kind = ctype['type']                    
            if cval is not None:
                con_info.append(f'eq_{ref.comp.classname}.{slvr}_{kind}_{cval}')
                con_list.append(
                    create_constraint(
                        system,Xrefs, 'ineq', cval, *args, **kw
                    )
                )

    for slvr, ref in sys_refs.get('solver.eq',{}).items():
        slv = slv_inst[slvr]
        slv_constraints = slv.constraints
        for ctype in slv_constraints:
            cval = ctype['value']
            kind = ctype['type']                    
            if cval is not None:
                con_info.append(f'eq_{ref.comp.classname}.{slvr}_{kind}_{cval}')
                con_list.append(
                    create_constraint(
                        system,Xrefs, 'eq', cval, *args, **kw
                    )
                )


    return constraints


# Objective functions & Utilities
def f_lin_min(system, Xref, Yref,weights=None, setcb=None, *args, **kw):
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

        with revert_X(system,Xref,Xnext=Xnext):
            
            if setcb:
                setcb(*args,**kw)

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
                
                if system.log_level < 10:
                    system.debug(f'obj {alias_name}: {x} -> {vals}')
                    #print(f'obj {f.__name__}: {x} -> {out}')

                return out  # n sized normal residual vector
            else:
                #the faster linear case
                if neg:
                    return np.sum(np.array( pos )) - np.sum(np.array( neg ))
                else:
                    return np.sum(np.array( pos ))

    return f

# signature in solve: refmin_solve(system,system,Xref)
def objectify(function,system,Xrefs,solver_info=None,*args,**kwargs):
    """converts a function f(system,slv_info:dict) into a function that safely changes states to the desired values and then runs the function. A function is returend as f(x,*args,**kw)"""
    base_dict = dict(system=system,Xrefs=Xrefs,args=args,**kwargs)
    alias_name = kwargs.pop("alias_name", f'{function.__name__}_obj')

    def loaf(x,*rt_args,**rt_kwargs):
        new_state = {p: x[i] for i, p in enumerate(Xrefs)}
        with revert_X(system, Xrefs, Xnext=new_state ) as x_prev:
            #print(locals()['solver_info'])
            updtinfo = base_dict.copy()
            updtinfo.update(x=x,rt_args=rt_args, **rt_kwargs)
            solver_info = locals().get('solver_info',updtinfo)

            out= function(system,solver_info)
            if system.log_level < 10:
                system.debug(f'obj {alias_name}: {x} -> {out}')
            return out        
    
    if system.log_level < 18:
        system.debug(f'obj settup {function} - > {loaf}')
        system.debug(inspect.getsource(function))

    fo = lambda x,*a,**kw: loaf(x,solver_info,*a,**kw)
    fo.__name__ ==f'OBJ[{function.__name__}]'
    return fo

def secondary_obj(
    obj_f, system,Xrefs,normalize=None,base_func=f_lin_min,*args,**kwargs
):  
    parms = list(Xrefs.keys())  # static x_basis
    base_dict = dict(system=system,args=args,Xrefs=Xrefs,**kwargs)
    alias_name = kwargs.pop("alias_name", f'{obj_f.__name__}_scndry')

    def f(x,*rt_args,**rt_kwargs): 

        new_state = {p: x[i] for i, p in enumerate(Xrefs)}
        base_call = base_func(system, Xrefs, normalize)        
        with revert_X(system, Xrefs,Xnext=new_state) as x_prev:
            A = base_call(x)
            solver_info = base_dict.copy()
            solver_info.update(x=x,Xrefs=Xrefs,normalize=normalize,rt_args=rt_args, **rt_kwargs)

            out =  A * (1 + obj_f(system, solver_info))
            if system.log_level < 10:
                system.debug(f'obj {alias_name}: {x} -> {out}')
            return out
        
    if system.log_level < 18:
        system.debug(f'secondary setup {obj_f} - > {f}')
        system.debug(inspect.getsource(function))        

    return f

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
    
    #gather unique update methods, to update on each call
    updt_refss = list(system.gather_update_refs().values())
    if updt_refss:
        updt = lambda *args,**kw: [v.value(*args,**kw) for v in updt_refss]
        updt.__name__ = f'{system.name}_glb_updater'
    else:
        updt = None

    # make objective function
    Fc = ffunc(system,Xref,Yref,weights,setcb=updt)
    Fc.__name__ = ffunc.__name__

    # get state
    if reset:
        x_pre = refset_get(Xref)  # record before changing

    if Xo is None:
        Xo = [Xref[p].value() for p in parms]

    elif isinstance(Xo, dict):
        Xo = [Xo[p] for p in parms]

    # TODO: IO for jacobean and state estimations (complex as function definition, requires learning)
    system.info(f'minimize! {Fc,Xo,parms}')
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
            refset_input(Xref, ans_dct)
        elif reset:
            refset_input(Xref, x_pre)
        return ans_dct

    else:
        min_dict = {p: a for p, a in zip(parms, ans.x)}
        if reset:
            refset_input(Xref, x_pre)
        elif doset:
            refset_input(Xref, min_dict)
            return min_dict
        if fail:
            raise Exception(f"failed to solve {ans.message}")
        return min_dict













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