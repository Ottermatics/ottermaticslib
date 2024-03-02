from engforge.system_reference import *

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
def sys_solver_variables(system,sys_refs,add_vars=None,excl_vars=None,**kw):
    """gathers variables from solver vars, and attempts to locate any input_vars to add as well. use exclude_vars to eliminate a variable from  the solver
    
    #TODO: add combo parsing
    """

    slv_inst = sys_refs['type']['solver']
    timz_inst = sys_refs['type']['time']

    sys_refs = sys_refs['attrs'] if 'attrs' in sys_refs else sys_refs

    dyns = sys_refs['dynamics.state']
    timz = {timz_inst[k].solver.parameter: v for k,v in sys_refs['time.parm'].items()}
    vars = {slv_inst[k].solver.var: v for k,v in sys_refs['solver.var'].items()}

    
    out = dict(dynamics=dyns,integration=timz,variables=vars)
    
    if 'as_set' in kw and kw['as_set']==True:
        vars = set.union(*(set(v) for k,v in out.items()))
        return vars
    if 'as_flat' in kw and kw['as_flat']==True:
        flt = {}
        for k,v in out.items():
            flt.update(v)
        return flt
    return out 

def sys_solver_objectives(system,sys_refs,Xrefs,add_obj=None,rmv_obj=None,**kw):
    """gathers variables from solver vars, and attempts to locate any input_vars to add as well. use exclude_vars to eliminate a variable from  the solver
    
    #TODO: add combo parsing
    """
    #Convert result per kind of objective (min/max ect)
    return {k:v for k,v in sys_refs['attrs']['solver.obj'].items()}


def sys_solver_constraints( system, sys_refs, Xrefs, add_con=None, *args, **kw):
    """formatted as arguments for the solver
    #TODO: add combo parsing
    """

    slv_inst = sys_refs['type']['solver']
    sys_refs = sys_refs['attrs']

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
    for slvr, ref in sys_refs['solver.var'].items():
        slv = slv_inst[slvr]
        slv_constraints = slv.constraints
        for ctype in slv_constraints:
            cval = ctype['value']
            kind = ctype['type']       
            parm = ctype['parm']             
            if cval is not None:
                #print(cval,kind,parm)
                if (
                    kind in ("min", "max")
                    and parm in Xparms
                    and isinstance(cval, (int, float))
                ):
                    minv, maxv = bnd_list[Xparms.index(parm)]
                    bnd_list[Xparms.index(parm)] = [
                        cval if kind == "min" else minv,
                        cval if kind == "max" else maxv,
                        ]
                #add the bias of cval to the objective function
                elif kind in ('min','max') and parm in Xparms:
                    parmref = Xrefs[parm]
                    #Ref Case
                    cval = ref_to_val_constraint(system,Xrefs,parmref,kind,cval,*args,**kw)
                    con_info.append(f'val_{ref.comp.classname}_{kind}_{parm}')
                    con_list.append(cval)
                else:
                    log.warning(f"bad constraint: {cval} {kind} {parm}")

    # Add Constraints
    for slvr, ref in sys_refs['solver.ineq'].items():
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

    for slvr, ref in sys_refs['solver.eq'].items():
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
def f_lin_min(system,Xref: dict,Yref: dict,normalizeX=None,normalizeY=None,*args,**kw):
    """Creates an anonymous function with stored references to system,Yref,normalize, that returns a scipy optimize friendly function of (x,Xref,*a,**kw) x which corresponds to the order of Xref dicts, and the other inputs are up to application.
    
    A normal value between -0.1 and 0.1 will be treated as zero or having no effect
    """
    base_dict = {'system':system,'Yref':Yref,'normalizeX':normalizeX,'normalizeY':normalizeY,'args':args,'kw':kw}
    xkey = "_".join(Xref.keys())
    ykey = "_".join(Yref.keys())
    alias_name = kw.pop("alias_name", f'min_X_{xkey}_Y_[{ykey}]')

    parms = list(Xref.keys())  # static x_basis
    yparm = list(Yref.keys())
    normalize = normalizeY if normalizeY is not None else np.ones(len(Yref))

    def f(x, *rt_a,**rt_kw):
        # anonymous function
        
        Xnext = {p:xi for p, xi in zip(parms, x)}

        if rt_a or rt_kw:
            slv_info = base_dict.copy()
            slv_info.update({'rt_a':rt_a,'kw':rt_kw})
        else:
            slv_info = base_dict

        with revert_X(system,Xref,Xnext=Xnext):
            grp = (yparm, normalize)
            vals = []
            for p,n in zip(*grp):
                if abs(n) < 0.1:
                    continue #not in sum zo 0
                ref = Yref[p]
                val = eval_ref(ref,system,slv_info)
                #print(p,n,ref,val,ref.comp,ref.key)
                vals.append( val/ n )

            rs = np.array(vals)
            out = np.sum(rs) ** 0.5
            
            if system.log_level < 10:
                system.debug(f'obj {alias_name}: {x} -> {vals}')
                #print(f'obj {f.__name__}: {x} -> {out}')

            return out  # n sized normal residual vector

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
    normalize: np.array = None,
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
    :param normalize: a dictionary of values to normalize the x values by, list also ok as long as same length and order as Xref
    :param reset: if the solution fails, reset the x values to their original state, if true will reset the x values to their original state on failure overiding doset.
    :param doset: if the solution is successful, set the x values to the solution by default, otherwise follows reset, if not successful reset is checked first, then doset
    #TODO: add 
    """
    parms = list(Xref.keys())  # static x_basis
    
    normalizeX,normalizeY = handle_normalize(normalize,Xref,Yref)
    print(f'norms: {normalizeX,normalizeY}')

    # make objective function
    Fc = ffunc(system,Xref,Yref,normalizeX,normalizeY)
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
        if reset:
            refset_input(Xref, x_pre)
        return ans_dct

    else:
        min_dict = {p: a for p, a in zip(parms, ans.x)}
        if reset:
            refset_input(Xref, x_pre)
        if doset:
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