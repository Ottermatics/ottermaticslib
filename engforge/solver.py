"""solver defines a SolverMixin for use by System.

Additionally the SOLVER attribute is defined to add complex behavior to a system as well as add constraints and transient integration.

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
        #log.info(f'only active: {outa}| {var_name:>10} {attr_name:>15}| C:{False}\tV:{False}\tA:{False}\tO:{False}')
        if not outa:
            log.msg(f'filt not active: {var_name:>10} {attr_name:>15}| C:{False}\tV:{False}\tA:{False}\tO:{False}')
            return False

    #Otherwise look at the combo filter, its its false return that
    outc = filt_combo_vars(var_name,solver_inst, extra_kw,combos)
    outp = False
    #if the combo filter didn't explicitly fail, check the var filter
    if attr_name in SLVR_SCOPE_PARM:
        outp = filt_var_vars(var_name,solver_inst, extra_kw)
        outr = any((outp,outc)) #redundant per above
    else:
        outr = outc
    

    fin =  bool(outr) and outa

    if not fin:
        log.msg(f'filter: {var_name:>10} {attr_name:>15}| C:{outc}\tV:{outp}\tA:{outa}\tO:{fin}')
    elif fin:
        log.msg(f'filter: {var_name:>10} {attr_name:>15}| C:{outc}\tV:{outp}\tA:{outa}\tO:{fin}| {combos} {extra_kw}')
    return fin 

# add to any SolvableMixin to allow solver use from its namespace
class SolverMixin(SolveableMixin):
    """A base class inherited by solveable items providing the ability to solve itself"""
    #TODO: move to problem context
    _run_id: str = None
    _solved = None
    _trans_opts = None
    _converged = False

    #TODO: implement equality solver as root
    solver_option = "minimize" #or root 
    

    # Configuration Information
    @property
    def solved(self):
        return self._solved
    
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
    def solver_vars(self,**kwargs):
        """applies the default combo filter, and your keyword arguments to the collect_solver_refs to test the ref / vars creations"""
        from engforge.solver import combo_filter
        #TODO: group solver vars (group_solver_refs())
        return self.collect_solver_refs(check_atr_f=combo_filter,check_kw=kwargs)
    

    def post_run_callback(self, **kwargs):
        """user callback for when run is complete"""
        pass

    def pre_run_callback(self, **kwargs):
        """user callback for when run is beginning"""
        pass

    def run(self, *args, **kwargs):
        """the steady state run metsolverhod for the system. It will run the system with the input vars and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""

        with ProblemExec(self,kwargs,level_name='run') as pbx:
            return self._iterate_input_matrix(self._run, *args, **kwargs)

    def _run(self, refs, icur, eval_kw=None, sys_kw=None, *args, **kwargs):
        """the steady state run method for the system. It will run the system with the input vars and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""

        # TODO: what to do with eval / sys kw
        # TODO: option to preserve state
        Ref.refset_input(refs, icur)
        self.debug(f"running with {icur}|{kwargs}")
        self.run_method(eval_kw=eval_kw, sys_kw=sys_kw,*args, **kwargs)
        self.debug(f"{icur} run time: {self._run_time}")

    def run_method(self, eval_kw=None, sys_kw=None, cb=None, **method_kw):
        """runs the case as currently defined by the system. This is the method that is called by the solver matrix to run the system with the input vars. It will run the system with the input vars and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""
        # set the values
        from engforge.system import System

        # Transeint
        #TODO: move to problem context!!! add plug-play functionality
        self._run_start = datetime.datetime.now()
        if self._run_id is None:
            self._run_id = int(uuid.uuid4())   

        if isinstance(self, System):
            self.system_references(recache=True)  
            #recache is important for iterators
            #TODO: only recache when iterators are present

        #call solver in scope
        self.eval(cb=cb, eval_kw=eval_kw, sys_kw=sys_kw, **method_kw)

        #TODO: move to problem context
        self._run_end = datetime.datetime.now()
        self._run_time = self._run_end - self._run_start

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
        self, cb=None, eval_kw: dict = None, sys_kw: dict = None, **kw
    ):
        """Evaluates the system with pre/post execute methodology
        :param kw: kwargs come from `sys_kw` input in run ect.
        :param cb: an optional callback taking the system as an argument of the form (self,eval_kw,sys_kw,**kw)
        """

        if self.log_level < 20:
            self.debug(f"running with kw:{kw}")
        
        #execute with problem context and execute signals
        with ProblemExec(self,kw,level_name='eval',eval_kw=eval_kw, sys_kw=sys_kw,post_callback=cb) as pbx:
            #FIXME: change the eval_kw / sys_kw 
            #pbx.pre_execute(**kw)
            self.index += 1
            out = self.execute(**kw)
            #pbx.post_execute( **kw)
            #TODO: move to problem context, and
            self.save_data(index=self.index)

            #TODO: define exit behavior for the problem context
            pbx.exit_to_level(level='eval',revert=False)

        if cb:
            cb(self, eval_kw, sys_kw, **kw)

        return out

    def execute(self,**kw):
        """Solves the system's system of constraints and integrates transients if any exist

        Override this function for custom solving functions, and call `solver` to use default solver functionality.

        :returns: the result of this function is returned from solver()
        """
        #TODO: pass to execution context!!!!
        # steady state
        dflt = dict(obj=None, cons=True, X0=None, dXdt=0)
        dflt.update(kw)
        return self.solver(**dflt)

    # TODO: add global optimization search for objective addin, via a new  `search_optimization` method.

    # TODO: code options for transient integration
    def solver(
            self, obj=None, cons=True, X0: dict = None, dXdt: dict = None, **kw
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

            output = {
                "obj": obj,
                "add_obj": True,
            }

            #Minimizer Args (#TODO: move to problem context!!!)
            opts = {"tol": 1e-6, "method": "SLSQP"}
            #to your liking sir
            opts.update(kw) #your custom args!
                   
            #use problem execution context
            self.debug(f'starting solver: {opts}')
            with ProblemExec(self,opts,level_name='sys_slvr') as pbx:
                #print(pbx.level_name,pbx.level_number)
                Xref = pbx.Xref
                if len(Xref) == 0:
                    self.debug(f'no variables found for solver: {opts}')
                    return
                
                #TODO: swap between vars depending on dxdt=True
                Yref = pbx.Yref

                opts.update(**pbx.constraints)
                output['pbx'] = pbx
                
                #Use Solver Context to Solve
                out = pbx.solve_min(Xref,Yref,output=output, **opts)
                if out['ans'].success:
                    pbx.set_ref_values(out['Xans'])
                    pbx.exit_to_level('sys_slvr',False)
                else:
                    #TODO: handle failure options
                    if pbx.raise_on_opt_failure:
                        raise ValueError(f"Solver failed to converge: {out['ans']}")
                    
                    if pbx.fail_revert:
                        pbx.exit_and_revert()
                    else:
                        pbx.exit_with_state()  

            return out














# # TODO: add basin hopping method in search_optimization
# output = {
#     "Xstart": Xg,
#     "vars": vars,
#     "Xans": None,
#     "dXdt": dXdt,
#     "obj": obj,
#     "add_obj": add_obj,
#     "Yref": Yref,
#     "update_methods": updts,
# }
# output['input_cons'] = constraints
# 
# # Solve / Minimize
# #if solve_mode == "root" and not has_constraints and obj is None:
#     #return self.solve_root(Xref, Yref, Xg, vars, output, **opts)
# if solve_mode == "minimize" or has_constraints:
#     # handle threahold for success depending on if objective provided
#     if "thresh" not in opts:
#         # default threshold
#         opts["thresh"] = self.success_thresh if not obj else None
#     elif obj:
#         opts["thresh"] = None
# 
#     if obj:
#         if add_obj:
#             #normi = opts.pop("normalize", None)
#             ffunc = lambda *args, **kw: secondary_obj(obj, *args, **kw)
#             opts["ffunc"] = ffunc
#         else:
#             # TODO: define behavior
#             pass
#     sol = self.solve_min(Xref, Yref, Xg, vars, output, **opts)
# 
#     x_in = [sol['Xans'][p] for p in vars]
#     cur = refset_get(Xref)
#     refset_input(Xref, sol['Xans'])
#     #print(con_list,constraints['constraints'])
# 
#     Ycon = {}
#     for c,k in zip(constraints['constraints'],constraints['info']):
#         cv = c['fun'](x_in,self,{})
#         #print(c,k,cv)
#         Ycon[k] = cv
#     output['Ycon'] = Ycon
#     output['Yobj'] = {k:v.value(self,output) for k,v in Yref.items()}
#     refset_input(Xref, cur)
# 
#     return sol
# 
# else:
#     self.warning(
#         f"no solution attempted! for {solve_mode} with {obj} and const: {constraints}"
#     )
# 
# 
# 
# 
