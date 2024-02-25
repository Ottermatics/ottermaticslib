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
from engforge.system_reference import Ref,refmin_solve,refroot_solve
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

    #TODO: code options for transient integration
    def solver(self,obj=None,cons=True,X0:dict=None,dXdt:dict=None,**kw):
        """runs the system solver using the current system state and modifying it. This is the default solver for the system, and it is recommended to add additional options or methods via the execute method.
        
        #TODO: make it so...
        :param obj: the objective function to minimize, by default will minimize the sum of the squares of the residuals. Objective function should be a function(system,Xs,Xt) where Xs is the system state and Xt is the system transient state. The objective function will be argmin(X)|(1+custom_objective)*residual_RSS
        :param cons: the constraints to be used in the solver, by default will use the system's constraints will be enabled when True. If a dictionary is passed the solver will use the dictionary as the constraints in addition to system constraints. These can be individually disabled by key=None in the dictionary.
        :param X0: the initial guess for the solver, by default will use the current system state. If a dictionary is passed the solver will use the dictionary as the initial guess in addition to the system state.
        :param dXdt: can be 0 to indicate steady-state, or None to not run the transient constraints. Otherwise a partial dictionary of parameters for the dynamics rates can be given, those not given will be assumed steady state or 0.
        :param kw: additional options for the solver, such as the solver_option, or the solver method options.
        """
        
        has_constraints = self.has_constraints
        solve_mode = self.solver_option
        if 'solver_option' in kw and kw['solver_option'] in SOLVER_OPTIONS:
            solve_mode = kw['solver_option']
        elif 'solver_option' in kw:
            raise ValueError(f"invalid solver option: {kw['solver_option']}")

        threshold = 0.001

        if solve_mode == 'minimize' or has_constraints:
            opts = {"rhobeg": 0.01, "catol": 1e-4, "tol": 1e-6}            
            opts.update(kw)
        else:
            opts = {}
        
        #TODO: get the solver configuration
        info = self.collect_solver_refs(min_set=True)
        pprint.pprint(info)

        #Collect Steady State References, indexed by same x parameter
        Xref = info['ss_states']
        Yref = info['ss_output']

        #TODO: enhance objective with obj_add for argmin(X)|(1+add_obj)*obj
        #TODO: add global optimization search for objective addin, via a new  `search_optimization` method.

        #Transient State Target
        #TODO: add dynamics integration (steady: dXdt==0, dXdt = C)
        #-how to handle dXdt cycle references? or in series as in 2nd order systems?
        #-how to handle conflicts with time parameters?
        #Dynamic Systems:

        #Time Integration:
        Xtr,Ytr = {},{}
        #rates are directly defined by input for Time integration 
        for k,v in info['tr_sets'].items():
            #TODO: check on which variable is callable vs settable. 
            Ytr[k] = v['dpdt']
            Xtr[k] = v['dpdt']
        
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
                assert isinstance(dXdt,dict),f'not dict for dXdt: {dXdt}'
                raise NotImplementedError(f"dynamic integration not yet implemented for dictionary based input: {dXdt}")
        else:
            #nope!
            Ytr = {}
            Xtr = {}
        
        #Time changes:
        #TODO: check parameter conflicts
        Xref.update(Xtr)
        Yref.update(Ytr)

        #Initial States
        if X0 is not None:
            assert isinstance(X0,dict), f'wrong format for state: {X0}'
            X0 = X0
            #TODO: check if X0 is valid
        else:
            X0 = Xref
        
        #constraints lookup
        constraints = self.solver_constraints
        if isinstance(cons,dict):
            #Remove None Values
            nones = {k for k,v in cons.items() if v is  None}
            for ki in nones:
                constraints.pop(ki,None)
            constraints.update({k:v for k,v in cons.items() if v is not None})

        elif cons is False or cons is None:
            constraints = {}
            
        parameter_list = list(X0.keys())
        Xg = Ref.refset_get(Xref)

        #TODO: add basin hopping method in search_optimization
        output = {'Xstart':Xg,'ans':None,'dXdt':dXdt,'constraints':constraints,'parms':parameter_list,'Xans':None}

        #Solve / Minimize
        if solve_mode == "root" and not has_constraints and obj is None:
            
            self._ans = refroot_solve(self,Xref,Yref,ret_ans=True,**kw)
            output['ans'] = self._ans
            if self._ans.success:
                #Set Values
                Xa = {p:self._ans.x[i] for i,p in enumerate(parameter_list)}
                output['Xans'] = Xa
                Ref.refset_input(Xref,Xa)
                self.pre_execute()
                self._converged = True
            else:
                Ref.refset_input(Xref,Xg)
                self.pre_execute()                
                self._converged = False
                raise Exception(f"solver didnt converge: {self._ans}")

            return output
        
        elif solve_mode == "minimize" or has_constraints:

            self._ans = refmin_solve(self,Xref,Yref,ret_ans=True,**kw)
            output['ans'] = self._ans
            if self._ans.success and abs(self._ans.fun) < threshold:
                #Set Values
                Xa = {p:self._ans.x[i] for i,p in enumerate(parameter_list)}
                output['Xans'] = Xa
                Ref.refset_input(Xref,Xa)
                self.pre_execute()
                self._converged = True

            elif self._ans.success:
                Xa = {p:self._ans.x[i] for i,p in enumerate(parameter_list)}
                output['Xans'] = Xa
                self.warning(
                    f"solver didnt fully solve equations! {self._ans.x} -> residual: {self._ans.fun}"
                )
                Ref.refset_input(Xref,Xg)
                self.pre_execute()                
                self._converged = False                

            else:
                Ref.refset_input(Xref,Xg)
                self.pre_execute()                      
                self._converged = False
                raise Exception(f"solver didnt converge: {self._ans}")

            return output

        else:
            self.warning(f"no solution attempted!")            



    def _iterate_components(self):
        """sets the current component for each product combination of iterable_components"""

        components = self.iterable_components

        if not components:
            yield  # enter once
        else:

            def _gen(gen, compkey):
                for itemkey, item in gen:
                    yield compkey, itemkey

            iter_vals = {
                cn: _gen(comp._item_gen(), cn)
                for cn, comp in components.items()
            }

            for out in itertools.product(*list(iter_vals.values())):
                for ck, ikey in out:
                    # TODO: progress bar or print location
                    components[ck].current_item = ikey
                yield out

            # finally reset the data!
            for ck, comp in components.items():
                comp.reset()

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

#Constraint Functions
def comp_has_constraints(comp):
    """checks for any active constrints"""
    for solvname, solv in comp.solvers.items():
        soltype = solv.solver
        if any([s is not None for s in soltype.constraints.values()]):
            return True
    return False


def comp_constraints(comp):
    """returns a list of constraitns by type"""
    out = []
    for solvname, solv in comp.solvers.items():
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

def comp_solver_constraints(comp):
    """formatted as arguments for the solver"""
    out = {"bounds": [], "constraints": []}

    # boudns must be X length wide
    for parm, solv in comp.solvers.items():
        out["bounds"].append([None, None])

    groups = {}
    for const in comp.constraints:
        name = const["name"]
        if name not in groups:
            groups[name] = [const]
        else:
            groups[name].append(const)

    for group, values in groups.items():
        for vcon in values:
            con = create_constraint(vcon, out, group)
            out["constraints"].append(con)
    return out

def create_constraint(comp, vcon, out, group):
    contype = vcon["type"]
    value = vcon["value"]
    #x_inx = comp.F_keyword_rev_order[vcon["name"]]
    # print(group,contype,value,x_inx,isinstance(value,(int,float)))

    if isinstance(value, (int, float)):
        # its a number
        val = float(value)
        if contype == "max":
            # make objective that is negative when x > lim
            def fun(x):
                return val - x[x_inx]

        else:

            def fun(x):
                return x[x_inx] + val

        cons = {"type": "ineq", "fun": fun}
        return cons
    else:
        val = copy.copy(value)
        # its a function
        if contype == "max":
            # make objective that is negative when x > lim
            def fun(x):
                with comp.revert_X():
                    comp.setX(x)
                    return val(comp) - x[x_inx]

        else:

            def fun(x):
                with comp.revert_X():
                    comp.setX(x)
                    return x[x_inx] - val(comp)

        cons = {"type": "ineq", "fun": fun}
        return cons
    
