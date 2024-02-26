"""solver defines a SolverMixin for use by System.

Additionally the SOLVER attribute is defined to add complex behavior to a system as well as add constraints and transient integration.
"""

import attrs
import uuid
import numpy
import scipy.optimize as scopt
from contextlib import contextmanager
import copy
import datetime

#from engforge.dynamics import DynamicsMixin
from engforge.properties import *
from engforge.solveable import SolveableMixin
from engforge.system_reference import *
import pprint   
import itertools,collections

SOLVER_OPTIONS = ["root", "minimize"]


class SolverLog(LoggingMixin):
    pass

log = SolverLog()


#add to any SolvableMixin to allow solver use from its namespace
class SolverMixin(SolveableMixin):
    _run_id: str = None
    _solved = None
    _trans_opts = None

    # TODO: implement custom solver with unified derivative equation.
    # _system_jacobean = None
    # _last_X = None
    # _current_X = None
    # _last_F = None
    # _current_F = None

    _converged = False
    # solver_option = attrs.field(default='root',validator=attrs.validators.in_(SOLVER_OPTIONS))
    solver_option = "root"

    # Configuration Information
    @property
    def solved(self):
        return self._solved
    
    def post_run_callback(self,**kwargs):
        """user callback for when run is complete"""
        pass

    def pre_run_callback(self,**kwargs):
        """user callback for when run is beginning"""
        pass    

    def run(self,*args,**kwargs):
        """the steady state run method for the system. It will run the system with the input parameters and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""

        self._iterate_input_matrix(self._run,*args,**kwargs)

    def _run(self,refs,icur,eval_kw=None,sys_kw=None,*args,**kwargs):
        """the steady state run method for the system. It will run the system with the input parameters and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""
        
        #TODO: what to do with eval / sys kw
        #TODO: option to preserve state
        Ref.refset_input(refs,icur)
        self.info(f"running with {icur}|{kwargs}")
        self.run_method(*args,**kwargs)
        self.debug(f"{icur} run time: {self._run_time}")

    def run_method(self,eval_kw=None,sys_kw=None,cb=None):
        """runs the case as currently defined by the system. This is the method that is called by the solver matrix to run the system with the input parameters. It will run the system with the input parameters and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""
        #set the values
        from engforge.system import System

        # Transeint
        self._run_start = datetime.datetime.now()
        if isinstance(self,System):
            self.system_references(recache=True) #recache is important for iterators, #TODO: only recache when iterators are present
        #steady state
        if self._run_id is None:
            self._run_id = int(uuid.uuid4())
        #Recache system references
        self.eval(cb=cb,eval_kw=eval_kw,sys_kw=sys_kw)

        self._run_end = datetime.datetime.now()
        self._run_time = self._run_end - self._run_start

    def run_internal_systems(self,sys_kw=None):
        """runs internal systems with potentially scoped kwargs"""
        # Pre Execute Which Sets Fields And PRE Signals
        from engforge.system import System
        
        #Record any changed state in components here, important for iterators
        if isinstance(self,System):
            self.system_references(recache=True)                              
        
        #System Solver Loop
        for key, comp in self.internal_systems().items():
            #self.debug(f"checking sys {key}.{comp}")
            if sys_kw and key in sys_kw:
                sys_kw_comp = sys_kw[key]
            else:
                sys_kw_comp = {}   
                         
            #Systems solve cycle
            if isinstance(comp, System): #should always be true
                self.info(f"solving {key} with {sys_kw_comp}")
                comp.eval(**sys_kw_comp)

    # Single Point Flow
    def eval(self, cb=None,eval_kw:dict=None,sys_kw:dict=None, *args,**kw):
        """Evaluates the system with additional inputs for execute()
        :param cb: an optional callback taking the system as an argument of the form (self,eval_kw,sys_kw,*args,**kw)
        """

        if self.log_level < 20:
            self.debug(f"running with X: {self.X} & args:{args} kw:{kw}")

        #0. run pre execute (signals=pre,both)
        self.pre_execute()

        # prep index
        self.index += 1

        #1.Runs The Solver
        try:
            out = self.execute( *args,**kw)
        except Exception as e:
            self.error(e, f"solver failed")
            out = None
            raise e

        #2. Update Internal Elements
        self.update_internal(eval_kw=eval_kw,*args,**kw)
        
        #run components and systems recursively
        self.run_internal_systems(sys_kw=sys_kw)

        # Post Execute (signals=post,both)
        self.post_execute()

        #Post Update Each Internal System
        self.post_update_internal(eval_kw=eval_kw,*args,**kw)
             
        # Save The Data
        self.save_data(index=self.index)

        if cb:
            cb(self,eval_kw,sys_kw,*args,**kw)

        return out

    def pre_execute(self):
        """runs the solver of the system"""
        if self.log_level <= 10:
            self.msg(f"pre execute")

        # TODO: set system fields from input
        for signame, sig in self.signals.items():
            if sig.mode == "pre" or sig.mode == "both":
                sig.apply()

    def post_execute(self):
        """runs the solver of the system"""
        if self.log_level <= 10:
            self.msg(f"post execute")
            
        for signame, sig in self.signals.items():
            if sig.mode == "post" or sig.mode == "both":
                sig.apply()

    def execute(self, *args,**kw):
        """Solves the system's system of constraints and integrates transients if any exist

        Override this function for custom solving functions, and call `solver` to use default solver functionality.

        :returns: the result of this function is returned from eval()
        """
        #steady state
        dflt = dict(obj=None,cons=True,X0=None,dXdt=0)
        dflt.update(kw)
        return self.solver(**dflt)

    #TODO: add global optimization search for objective addin, via a new  `search_optimization` method.

    #TODO: code options for transient integration
    def solver(self,obj=None,cons=True,X0:dict=None,dXdt:dict=None,**kw):
        """runs the system solver using the current system state and modifying it. This is the default solver for the system, and it is recommended to add additional options or methods via the execute method.
        
        #TODO: make it so...
        :param obj: the objective function to minimize, by default will minimize the sum of the squares of the residuals. Objective function should be a function(system,Xs,Xt) where Xs is the system state and Xt is the system transient state. The objective function will be argmin(X)|(1+custom_objective)*residual_RSS when `add_obj` is True in kw otherwise argmin(X)|custom_objective with constraints on the system as balances instead of first objective being included.
        :param cons: the constraints to be used in the solver, by default will use the system's constraints will be enabled when True. If a dictionary is passed the solver will use the dictionary as the constraints in addition to system constraints. These can be individually disabled by key=None in the dictionary.
        :param X0: the initial guess for the solver, by default will use the current system state. If a dictionary is passed the solver will use the dictionary as the initial guess in addition to the system state.
        :param dXdt: can be 0 to indicate steady-state, or None to not run the transient constraints. Otherwise a partial dictionary of parameters for the dynamics rates can be given, those not given will be assumed steady state or 0.
        :param kw: additional options for the solver, such as the solver_option, or the solver method options.
        :param cons_opts: additional options for the constraints, such as the solver_option, or the solver method options. 
        """
        add_obj = kw.pop('add_obj',True)

        sol = self.collect_solver_dynamics()
        info = self.collect_solver_refs(min_set=True)
        pprint.pprint(info)

        #Collect Steady State References, indexed by same x parameter
        Xref = info['ss_states']
        Yref = info['ss_output']        
        
        #get the solver configuration
        solvers = get_solvers(self,sol,info)
        has_constraints = comp_has_constraints(solvers)
        solve_mode = self.solver_option
        if 'solver_option' in kw and kw['solver_option'] in SOLVER_OPTIONS:
            solve_mode = kw['solver_option']
        elif 'solver_option' in kw:
            raise ValueError(f"invalid solver option: {kw['solver_option']}")

        if solve_mode == 'minimize' or has_constraints:
            opts = {"tol": 1e-6,'method':'SLSQP'}
        else:
            opts = {}
        opts.update(kw)
        
        #Get constraints
        constraints = sys_solver_constraints(self,solvers,Xref,cons,**kw.pop('cons_opts',{}))

        #TODO: enhance objective with obj_add for argmin(X)|(1+add_obj)*obj

        #Transient State Target
        #-make distinction between set / call references
        #-use a constraint to achieve desired transient state
        #Time Integration:
        Xtr,Ytr = {},{}
        #rates are directly defined by input for Time integration 
        for k,v in info['tr_sets'].items():
            #check on which variable is callable vs settable.
            if v['dpdt'].allow_set:
                #set the rate since we can
                Ytr[k] = v['dpdt']
                Xtr[k] = v['dpdt']
            else:
                Ytr[k] = v['dpdt']
                Xtr[k] = v['parm']
        
        #Dynamic Components
        for comp,cdict in info['dyn_comp'].items():
            if cdict['state']:
                ckey = comp+'.' if comp else ''
                for p,v in cdict['state'].items():
                    Xtr[p] = v
                    Ytr[p] = cdict['comp'].ref_dXdt(p.replace(ckey,'')) 

        #Dynamics Configuration
        if dXdt is not None and dXdt is not False:
            if dXdt == 0:
                pass
            elif dXdt is True:
                pass
            else:#dXdt is a dictionary
                #TODO: add dynamics integration (steady: dXdt==0, dXdt = C)
                assert isinstance(dXdt,dict),f'not dict for dXdt: {dXdt}'
                raise NotImplementedError(f"dynamic integration not yet implemented for dictionary based input: {dXdt}") #TODO:
        else:
            #nope!
            Ytr = {}
            Xtr = {}
        
        #Time changes:
        #TODO: check parameter conflicts
        #TODO: convert equalities to constraints (global objmin vs constraint?)
        Xref.update(Xtr)
        Yref.update(Ytr)
        opts.update(**constraints)

        #Initial States
        if X0 is not None:
            assert isinstance(X0,dict), f'wrong format for state: {X0}'
            X0 = X0
            #TODO: check if X0 is valid
        else:
            X0 = Xref
            
        parms = list(Xref.keys())
        Xg = Ref.refset_get(Xref)

        #TODO: add basin hopping method in search_optimization
        output = {'Xstart':Xg,'constraints':constraints,'parms':parms,'Xans':None,'dXdt':dXdt,'obj':obj,'add_obj':add_obj}

        #Solve / Minimize
        if solve_mode == "root" and not has_constraints and obj is None:
            return self.solve_root(Xref,Yref,Xg,parms,output,**opts)
        elif solve_mode == "minimize" or has_constraints:
            #handle threahold for success depending on if objective provided
            if 'thresh' not in opts:
                #default threshold
                opts['thresh'] = 0.0001 if not obj else None
            elif obj:
                opts['thresh'] = None

            if obj:
                if add_obj:
                    normi = opts.pop('normalize',None)
                    ffunc = lambda *args,**kw: secondary_obj(obj,*args,**kw)
                    opts['ffunc'] = ffunc
                else:
                    #TODO: define behavior
                    pass
            return self.solve_min(Xref,Yref,Xg,parms,output,**opts)
        else:
            self.warning(f"no solution attempted! for {solve_mode} with {obj} and const: {constraints}")    

    def solve_root(self,Xref,Yref,Xreset,parms,output=None,fail=True,**kw):
        """
        Solve the minimization problem using the given parameters.

        :param Xref: The reference input values.
        :type Xref: dict
        :param Yref: The reference output values.
        :type Yref: dict
        :param Xreset: The reset input values.
        :type Xreset: dict
        :param parms: The list of parameter names.
        :type parms: list
        :param output: The output dictionary to store the results. (default: None)
        :type output: dict, optional
        :param fail: Flag indicating whether to raise an exception if the solver doesn't converge. (default: True)
        :type fail: bool, optional
        :param kw: Additional keyword arguments.
        :type kw: dict
        :return: The output dictionary containing the results.
        :rtype: dict
        """        
        if output is None:
            output = {'Xstart':Xreset,'parms':parms,'Xans':None,'success':None} 

        self._ans = refroot_solve(self,Xref,Yref,ret_ans=True,**kw)
        output['ans'] = self._ans
        if self._ans.success:
            #Set Values
            Xa = {p:self._ans.x[i] for i,p in enumerate(parms)}
            output['Xans'] = Xa
            Ref.refset_input(Xref,Xa)
            self.pre_execute()
            self._converged = True
            output['success'] = True
        else:
            Ref.refset_input(Xref,Xreset)
            self.pre_execute()                
            self._converged = False
            if fail:
                raise Exception(f"solver didnt converge: {self._ans}")
            output['success'] = False

        return output                

    def solve_min(self, Xref, Yref, Xreset, parms, output=None, fail=True,**kw):
        """
        Solve the minimization problem using the given parameters.

        :param Xref: The reference input values.
        :type Xref: dict
        :param Yref: The reference output values.
        :type Yref: dict
        :param Xreset: The reset input values.
        :type Xreset: dict
        :param parms: The list of parameter names.
        :type parms: list
        :param output: The output dictionary to store the results. (default: None)
        :type output: dict, optional
        :param fail: Flag indicating whether to raise an exception if the solver doesn't converge. (default: True)
        :type fail: bool, optional
        :param thresh: float / none passed as a keyword, if evaluates to false any minimum is accepted, otherwise the absolute of the residual must be below threshold for success
        :param kw: Additional keyword arguments.
        :type kw: dict
        :return: The output dictionary containing the results.
        :rtype: dict
        """
        thresh = kw.pop('thresh', 0.0001)

        if output is None:
            output = {'Xstart': Xreset, 'parms': parms, 'Xans': None, 'success': None}

        self._ans = refmin_solve(self, Xref, Yref, ret_ans=True, **kw)
        output['ans'] = self._ans
        de = abs(self._ans.fun)
        if self._ans.success and de < thresh if thresh else True:
            # Set Values
            Xa = {p: self._ans.x[i] for i, p in enumerate(parms)}
            output['Xans'] = Xa
            Ref.refset_input(Xref, Xa)
            self.pre_execute()
            self._converged = True
            output['success'] = True
        elif self._ans.success:
            # out of threshold condition
            Xa = {p: self._ans.x[i] for i, p in enumerate(parms)}
            output['Xans'] = Xa
            self.warning(
                f"solver didnt fully solve equations! {self._ans.x} -> residual: {self._ans.fun}"
            )
            Ref.refset_input(Xref, Xreset)
            self.pre_execute()
            self._converged = False
            output['success'] = False  # only false with threshold
        else:
            Ref.refset_input(Xref, Xreset)
            self.pre_execute()
            self._converged = False
            if fail:
                raise Exception(f"solver didnt converge: {self._ans}")
            output['success'] = False

        return output

    #Replaces Tabulation Method
    @solver_cached
    def data_dict(self):
        """records properties from the entire system, via system references cache"""
        out = collections.OrderedDict()
        sref = self.system_references()
        for k, v in sref["attributes"].items():
            val = v.value()
            if isinstance(val, TABLE_TYPES):
                out[k] = val
            else:
                out[k] = numpy.nan
        for k, v in sref["properties"].items():
            val = v.value()
            if isinstance(val, TABLE_TYPES):
                out[k] = val
            else:
                out[k] = numpy.nan
        return out

#Secondary Objective function
#signature in solve: refmin_solve(system,Xref,Yref,normalize)
#
def secondary_obj(obj_f,system,Xref,Yref,normalize=None,base_func = f_lin_min):
    parms = list(Xref.keys()) #static x_basis
    base_call = base_func(system,Xref,Yref,normalize)
    def f(x): #anonymous function
        for p,xi in zip(parms,x):
            Xref[p].set_value(xi)        
        A = base_call(x)
        return A*(1+obj_f(system,Xref,Yref,normalize))
    return f


#Constraint Functions
def get_solvers(system,sol=None,refs=None):
    if sol is None:
        sol = system.collect_solver_dynamics()
    if refs is None:
        refs = system.collect_solver_refs(min_set=True)
    key = lambda ck,pk: f'{ck+"." if ck else ""}{pk}'
    dct = lambda ck,pk,cd,ci: {'comp':cd['conf'],'inst':getattr(cd['conf'],pk,None),'type':ci,'parm':pk,'sys_key':ck,'ref_indep': refs['solvers'][key(ck,pk)]['ref_indep'],'ref_dep': refs['solvers'][key(ck,pk)]['ref_dep'],'constraints':getattr(cd['conf'],pk,None).solver.constraints,'independent':getattr(cd['conf'],pk,None).solver.independent,'dependent':getattr(cd['conf'],pk,None).dependent }
    return { key(ck,pk):dct(ck,pk,cd,ci) for ck,cd in sol['solvers'].items() for pk,ci in cd['solvers'].items()}

def comp_has_constraints(solvers):
    """checks for any active constrints"""
    for solvname, solv in solvers.items():
        if isinstance(solv,dict): #unpack dictionary expansion
            solv = solv['inst']
        soltype = solv.solver
        if any([s is not None for s in soltype.constraints.values()]):
            return True
    return False


def comp_constraints(solvers):
    """returns a list of constraitns by type"""
    out = []
    for solvname, solv in solvers.items():
        soltype = solv.solver
        for stype, sv in soltype.constraints.items():
            if sv is None:
                continue
            out.append(
                {
                    "name": solvname,
                    "type": stype,
                    "value": sv,
                    "solver": soltype,
                    "independent": soltype.independent,
                    "solv_instance": solv,
                }
            )
    return out

def comp_solver_constraints(solvers,Xrefs,*args,**kw):
    """formatted as arguments for the solver"""
    out = {"bounds": [], "constraints": []}
    
    Xparms = list(Xrefs)

    # boudns must be X length wide
    for parm, solv in solvers.items():
        out["bounds"].append([None, None])

    constraints = comp_constraints(solvers)
    groups = {}
    for const in constraints:
        name = const["name"]
        if name not in groups:
            groups[name] = [const]
        else:
            groups[name].append(const)

    for group, values in groups.items():
        for vcon in values:
            parm = vcon["independent"]  
            vtype = vcon["type"]
            value = vcon["value"]
            con = create_constraint(Xrefs,vtype,value, parm,*args,**kw)
            out["constraints"].append(con)
    return out

def sys_solver_constraints(system,solvers,Xrefs,input_cons,skip=None,*args,**kw):
    """formatted as arguments for the solver"""    
    Xparms = list(Xrefs)

    #constraints lookup
    bnd_list = [[None,None]]*len(Xparms)
    con_list = []
    constraints = {'constraints':con_list,'bounds':bnd_list}
    
    if isinstance(input_cons,dict):
        #Remove None Values
        nones = {k for k,v in input_cons.items() if v is  None}
        for ki in nones:
            constraints.pop(ki,None)
        assert all([callable(v) for k,v in input_cons.items()]), f"all custom input for constraints must be callable with X as argument"
        constraints['constraints'].extend([v for k,v in input_cons.items() if v is not None])
    
    if input_cons is False:
        constraints = {}
    else:
        #Add Constraints
        for slvr,slvdct in solvers.items():
            if skip and slvr in skip:
                continue
            else:
                slv = slvdct['inst']
                slv_constraints = slv.solver.constraints
                for ctype, cval in slv_constraints.items():
                    if cval is not None:
                        parm = slv.solver.independent
                        con_list.append(create_constraint(system,Xrefs,ctype,cval,parm,*args,**kw))
                        if ctype in ('min','max') and parm in Xparms and isinstance(cval,(int,float)):
                            minv,maxv = bnd_list[Xparms.index(parm)]
                            bnd_list[Xparms.index(parm)] = [cval if ctype == 'min' else minv, cval if ctype == 'max' else maxv]

    return constraints

def create_constraint(system,Xrefs, contype:str,value, parm=None,*args,**kwargs):
    """creates a constraint with bounded solver input from a constraint definition in dictionary with type and value. If value is a function it will be evaluated with the extra arguments provided. If parm is None, then the constraint is assumed to be not in reference to the system x parameters, otherwise lookups are made to that parameter."""
    assert contype in ("min", "max",'eq','ineq'), f"bad constraint type: {contype}"
    Xparms = list(Xrefs)
    
    if parm is not None and parm in Xrefs:
        assert contype in ('min','max'), f"bad constraint type for system parameter comparison {contype}"
        x_inx = Xparms.index(parm)
    else:
        assert callable(value), f'no parameter in requires a callable: {value}'
        x_inx = None

    ctype = "ineq" if contype in ('min','max') else contype
    if isinstance(value, (int, float)):
        # its a number
        assert x_inx is not None, f"no parameter for value: {value}"
        val = float(value)
        if contype == "max":
            # make objective that is negative when x > lim
            def fun(x):
                return val - x[x_inx]
        elif contype == "min":
            def fun(x):
                return x[x_inx] + val

        cons = {"type": ctype, "fun": fun}
        return cons
    else:
        val = copy.copy(value)
        # its a function
        if contype == "max":
            # make objective that is negative when x > lim
            def fun(x):
                with revert_X(system,Xrefs) as x_prev:
                    refset_input(Xrefs, {p: x[i] for i, p in enumerate(Xrefs)})
                    return val(x,*args,**kwargs) - (x[x_inx] if x_inx is not None else 0)
        elif contype == "min":
            def fun(x):
                with revert_X(system,Xrefs) as x_prev:
                    refset_input(Xrefs, {p: x[i] for i, p in enumerate(Xrefs)})
                    return (x[x_inx] if x_inx  is not None else 0) - val(x,*args,**kwargs)
        elif contype == "eq":
            def fun(x):
                with revert_X(system,Xrefs) as x_prev:
                    refset_input(Xrefs, {p: x[i] for i, p in enumerate(Xrefs)})
                    return val(x,*args,**kwargs)
                
        elif contype == "ineq":
            def fun(x):
                with revert_X(system,Xrefs) as x_prev:
                    refset_input(Xrefs, {p: x[i] for i, p in enumerate(Xrefs)})
                    return val(x,*args,**kwargs)

        cons = {"type": ctype, "fun": fun}
        return cons
    


if __name__ == '__main__':

    import unittest
    from engforge.system import System
    from engforge.components import Component,forge
    from engforge.attr_slots import Slot
    from engforge.attr_dynamics import Time
    from engforge.attr_signals import Signal
    from engforge.attr_solver import Solver
    from engforge.properties import system_property

    @forge(auto_attribs=True)
    class SolvComp(System):
        x: float = 1.0
        y: float = 1.0
        z: float = 1.0

        cost_x: float = 10
        cost_y: float = 10
        cost_z: float = 5
        cost_per_length: float = 10

        budget: float = 100
        total_length: float = 50
        
        size = Solver.define('edge_margin','x')
        size.add_constraint('min',0.0)
        
        edge = Solver.define('edge_margin','y')
        edge.add_constraint('min',0.0)
        
        height = Solver.define('cost_margin','z')
        height.add_constraint('min',0.0)

        @system_property
        def combine_length(self)->float:
            return (self.x+self.y+self.z)*4
        
        @system_property
        def cost_margin(self)->float:
            dv = self.budget - self.cost
            if dv > 0:
                return np.log(dv)
            return -1*dv
        
        @system_property
        def edge_margin(self)->float:
            dv = self.total_length - self.combine_length
            if dv > 0:
                return np.log(dv)
            return -1*dv
        
        @system_property
        def volume(self)->float:
            return self.x*self.y*self.z
        
        @system_property
        def cost(self)->float:
            return self.x*4*self.cost_x + self.y*4*self.cost_y + self.z*4*self.cost_z
        
        @system_property
        def cost_to_volume(self)->float:
            return self.cost / self.volume
        

        
    # @forge(auto_attribs=True)
    # class SolverSystem(System):
    #     pass


