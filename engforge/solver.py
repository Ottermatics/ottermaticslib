"""solver defines a SolverMixin for use by System.

Additionally the Solver attribute is defined to add complex behavior to a system as well as add constraints and transient integration.

### A general Solver Run Will Look Like:
0. run pre execute (signals=pre,both)
1. add execution context with **kwargument for the signals. 
2. parse signals here (through a new Signals.parse_rtkwargs(**kw)) which will non destructively parse the signals and return all the signal candiates which are put into an ProblemExec object that resets the signals after the run depending on the revert behavior
3. the execute method will recieve this ProblemExec object where it can update the solver references / signals so that it can handle them per the signals api
4. with self.execution_context(**kwargs) as ctx_exe: 
    1. pre-update / signals
    <FLEXIBLE_Exec>#self.execute(ctx_exe,**kwargs) 
    2. post-update / signals
    > signals will be reset after the execute per the api
5. run post update
6. exit condition check via problem context input
"""

import attrs
import uuid
import numpy
import scipy.optimize as scopt
from contextlib import contextmanager
import copy
import datetime

# from engforge.dynamics import DynamicsMixin
from engforge.properties import *
from engforge.solveable import SolveableMixin
from engforge.system_reference import *
import pprint
import itertools, collections
import inspect

SOLVER_OPTIONS = ["minimize"]#"root", "global", 
from engforge.solver_utils import *
from engforge.problem_context import *
from engforge.attr_solver import Solver,SolverInstance
import sys
class SolverLog(LoggingMixin):
    pass


log = SolverLog()

SLVR_SCOPE_PARM = ['solver.eq','solver.ineq','solver.var','solver.obj','time.var','time.rate','dynamics.state','dynamics.rate']

def combo_filter(attr_name,var_name, solver_inst, extra_kw,combos=None)->bool:
    #TODO: allow solver_inst to be None for dyn-classes
    #proceed to filter active items if vars / combos inputs is '*' select all, otherwise discard if not active
    if extra_kw is None:
        extra_kw = {}

    outa = True
    if extra_kw.get('only_active',True):

        outa  =  filt_active(var_name,solver_inst,extra_kw=extra_kw,dflt=False)
        if not outa:
            log.msg(f'filt not active: {var_name:>10} {attr_name:>15}| C:{False}\tV:{False}\tA:{False}\tO:{False}')
            return False
    both_match =  extra_kw.get('both_match',True)
    #Otherwise look at the combo filter, its its false return that
    outc = filter_combos(var_name,solver_inst, extra_kw,combos)
    outp = False
    #if the combo filter didn't explicitly fail, check the var filter
    if (outc and attr_name in SLVR_SCOPE_PARM) or not both_match:
        outp = filter_vals(var_name,solver_inst, extra_kw)
        if extra_kw.get('both_match',True):
            outr = all((outp,outc)) #redundant per above
        else:
            outr = any((outp,outc))
    else:
        outr = outc
    

    fin =  bool(outr) and outa

    if not fin:
        log.debug(f'filter: {var_name:>20} {attr_name:>15}| C:{outc}\tV:{outp}\tA:{outa}\tO:{fin}')
    elif fin:
        log.debug(f'filter: {var_name:>20} {attr_name:>15}| C:{outc}\tV:{outp}\tA:{outa}\tO:{fin}| {combos}')

    return fin 

# add to any SolvableMixin to allow solver use from its namespace
class SolverMixin(SolveableMixin):
    """A base class inherited by solveable items providing the ability to solve itself"""

    #TODO: implement constraint equality solver as root
    
    # Configuration Information
    @property
    def solved(self):
        if self.last_context is None: 
            return False        
        elif self.last_context.data:
            return True
        return False
    
    # Replaces Tabulation Method
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
    
    #Official Solver Interface
    def solver_vars(self,check_dynamics=True,addable=None,**kwargs):
        """applies the default combo filter, and your keyword arguments to the collect_solver_refs to test the ref / vars creations
        
        parses `add_vars` in kwargs to append to the collected solver vars
        :param add_vars: can be a str, a list or a dictionary of variables: solver_instance kwargs. If a str or list the variable will be added with positive only constraints. If a dictionary is chosen, it can have keys as parameters, and itself have a subdictionary with keys: min / max, where each respective value is placed in the constraints list, which may be a callable(sys,prob) or numeric. If nothing is specified the default is min=0,max=None, ie positive only.
        """
        from engforge.solver import combo_filter

        out = self.collect_solver_refs(check_atr_f=combo_filter,check_kw=kwargs,check_dynamics=check_dynamics)

        base_const = {'min':0,'max':None} #positive only

        #get add_vars and add to attributes
        if addable and ( addvar:= kwargs.get('add_vars',[])):
            matches = []
            if addvar:
                
                #handle csv, listify string
                if isinstance(addvar,str):
                    addvar = addvar.split(',')
                
                #handle dictionary                    
                added = set(())
                avars = list(addable['attributes'].keys())
                for av in addvar: #list/dict-keys
                    matches = set(fnmatch.filter(avars,av))
                    #Lookup Constraints if input is a dictionary
                    if isinstance(addvar,dict) and isinstance(addvar[av],dict):
                        const = base_const.copy()
                        const.update(addvar[av])
                    elif isinstance(addvar,dict):
                        raise ValueError(f'dictionary must have a subdictionary for {av}')
                    else:
                        const = base_const.copy()

                    #for each match we add the variable
                    for mtch in matches:
                        if mtch in added:
                            continue #add only once
                        else:
                            added.add(mtch)

                        if self.log_level < 5:
                            self.msg(f'adding {mtch} to solver vars')
                        #add constraint values and type/instance
                        ref = addable['attributes'][mtch].copy()
                        mtch_type = Solver.declare_var(mtch) #type
                        #bounds
                        mtch_type.constraints[0]['value'] = const['min']
                        mtch_type.constraints[1]['value'] = const['max'] 
                        #instance
                        mtch_inst = SolverInstance(mtch_type,self)
                        #attach
                        out['attrs']['solver.var'][mtch] = ref
                        out['type']['solver'][mtch] = mtch_inst

        return out
    

    def post_run_callback(self, **kwargs):
        """user callback for when run is complete"""
        pass

    def pre_run_callback(self, **kwargs):
        """user callback for when run is beginning"""
        pass

    def run(self, **kwargs):
        """the steady state run the solver for the system. It will run the system with the input vars and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""

        if 'opt_fail' not in kwargs:
            kwargs['opt_fail'] = False

        with ProblemExec(self,kwargs,level_name='run') as pbx:
            #problem context removes slv/args from kwargs
            return self._iterate_input_matrix(self.eval, return_results=True,**kwargs)


    def run_internal_systems(self, sys_kw=None):
        """runs internal systems with potentially scoped kwargs"""
        # Pre Execute Which Sets Fields And PRE Signals
        from engforge.system import System

        # Record any changed state in components here, important for iterators
        if isinstance(self, System):
            self.system_references(recache=True)

        # System Solver Loop
        for key, comp in self.internal_systems().items():
            if hasattr(comp,'_solver_override') and comp._solver_override:
                if sys_kw and key in sys_kw:
                    sys_kw_comp = sys_kw[key]
                else:
                    sys_kw_comp = {}

                # Systems solve cycle
                if isinstance(comp, System):  # should always be true
                    self.info(f"solving {key} with {sys_kw_comp}")
                    comp.eval(**sys_kw_comp)
 
    # Single Point Flow
    def eval(
        self, Xo=None,eval_kw: dict = None, sys_kw: dict = None,cb=None, **kw
    ):
        """Evaluates the system with pre/post execute methodology
        :param kw: kwargs come from `sys_kw` input in run ect.
        :param cb: an optional callback taking the system as an argument of the form (self,eval_kw,sys_kw,**kw)
        """

        # Transeint
        from engforge.system import System
        if kw.pop('refresh_references',True) and isinstance(self, System):
            #recache is important for iterators #TODO: only with iterable comps
            self.system_references(recache=True)    

        #default behavior of system is to accept non-optimal results but describe the behavior anyways
        if 'opt_fail' not in kw:
            kw['opt_fail'] = False

        
        self.debug(f"running with kw:{kw}")
        
        #execute with problem context and execute signals
        with ProblemExec(self,kw,level_name='eval',eval_kw=eval_kw, sys_kw=sys_kw,post_callback=cb,Xnew=Xo) as pbx:
            out = self.execute(**kw)
            pbx.exit_to_level(level='eval',revert=False)

        if self.log_level >= 20:
            sys.stdout.write('.')

        return out

    def execute(self,**kw):
        """Solves the system's system of constraints and integrates transients if any exist

        Override this function for custom solving functions, and call `solver` to use default solver functionality.

        :returns: the result of this function is returned from solver()
        """
        # steady state
        dflt = dict()#obj=None, cons=True, X0=None, dXdt=0)
        dflt.update(kw)
        return self.solver(**dflt)

    # TODO: add global optimization search for objective addin, via a new  `search_optimization` method.

    # TODO: code options for transient integration
    def solver(
            self, enter_refresh=True,save_on_exit=True,**kw
        ):
            """
            runs the system solver using the current system state and modifying it. This is the default solver for the system, and it is recommended to add additional options or methods via the execute method.

            

            :param obj: the objective function to minimize, by default will minimize the sum of the squares of the residuals. Objective function should be a function(system,Xs,Xt) where Xs is the system state and Xt is the system transient state. The objective function will be argmin(X)|(1+custom_objective)*residual_RSS when `add_obj` is True in kw otherwise argmin(X)|custom_objective with constraints on the system as balances instead of first objective being included.
            :param cons: the constraints to be used in the solver, by default will use the system's constraints will be enabled when True. If a dictionary is passed the solver will use the dictionary as the constraints in addition to system constraints. These can be individually disabled by key=None in the dictionary.

            :param X0: the initial guess for the solver, by default will use the current system state. If a dictionary is passed the solver will use the dictionary as the initial guess in addition to the system state.
            :param dXdt: can be 0 to indicate steady-state, or None to not run the transient constraints. Otherwise a partial dictionary of vars for the dynamics rates can be given, those not given will be assumed steady state or 0.

            :param kw: additional options for the solver, such as the solver_option, or the solver method options. Described below
            :param combos: a csv str or list of combos to include, including wildcards. the default means all combos will be run unless ign_combos or only combos alters behavior. The initial selection of combos is made by matching any case with the full name of the combo, or a parial name with a wildcard(s) in the combo name Ignore and only combos will further filter the selection. Wildcards / queries per fnmatch
            :param ign_combos: a list of combo vars to ignore.
            :param only_combos: a list of combo vars to include exclusively.
            :param add_var: a csv str or variables to include, including wildcards. the default means all combos will be run unless ign_combos or only combos alters behavior. The initial selection of combos is made by matching any case with the full name of the combo, or a parial name with a wildcard(s) in the combo name Ignore and only combos will further filter the selection. Wildcards / queries per fnmatch
            :param ign_var: a list of combo vars to ignore.
            :param only_var: a list of combo vars to include exclusively.            
            :param add_obj: a flag to add the objective to the system constraints, by default will add the objective to the system constraints. If False the objective will be the only constraint.
            :param only_active: default True, will only look at active variables objectives and constraints
            :param activate: default None, a list of solver vars to activate
            :param deactivate: default None, a list of solver vars to deactivate (if not activated above)
            """

            #to your liking sir
            opts = {} #TODO: add defaults
            opts.update(kw) #your custom args!
                   
            #use problem execution context
            self.debug(f'starting solver: {opts}')
            
            with ProblemExec(self,opts,level_name='sys_slvr',enter_refresh=enter_refresh,save_on_exit=save_on_exit) as pbx:
                
                #Use Solver Context to Solve
                out = pbx.solve_min(**opts)
                has_ans = 'ans' in out
                #depending on the solver success, failure or no solution, we can exit the solver
                if has_ans and out['ans'] and out['ans'].success:
                    #this is where you want to be! <<<
                    pbx.set_ref_values(out['Xans'])
                    pbx.exit_to_level('sys_slvr',False)

                elif has_ans and out['ans'] is None:
                    #deterministic input based case for updates / signals
                    self.debug(f'exiting solver with no solution {out}')
                    pbx.exit_with_state() 

                else:
                    #handle failure options
                    if pbx.opt_fail:

                        #if log.log_level < 15:
                        pbx.warning(f'Optimization Failed: {pbx.sys_refs} | {pbx.constraints}')

                        #if log.log_level < 5:
                        pbx.debug(f'{pbx.__dict__}')

                        ve = f"Solver failed to converge: {out['ans']}"
                        raise ValueError(ve)
                    
                    if pbx.fail_revert:
                        pbx.exit_and_revert()
                    else:
                        pbx.exit_with_state()  
                #close context
            
            #return output
            return out













