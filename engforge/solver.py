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

# from engforge.dynamics import DynamicsMixin
from engforge.properties import *
from engforge.solveable import SolveableMixin
from engforge.system_reference import *
import pprint
import itertools, collections
import inspect

SOLVER_OPTIONS = ["minimize"]#"root", "global", 
from engforge.solver_utils import *

class SolverLog(LoggingMixin):
    pass


log = SolverLog()

SLVR_SCOPE_PARM = ['solver.eq','solver.ineq','solver.var','solver.obj','time.parm','time.rate','dynamics.state','dynamics.rate']

def combo_filter(attr_name,parm_name, solver_inst, extra_kw,combos=None)->bool:
    
    #proceed to filter active items if vars / combos inputs is '*' select all, otherwise discard if not active
    slv_vars = str_list_f(extra_kw.get('slv_vars',[]))
    combos_in = str_list_f(extra_kw.get('combos',[]))
    activate = str_list_f(extra_kw.get('activate',[]))
    deactivate = str_list_f(extra_kw.get('deactivate',[]))


    outa = True
    if extra_kw.get('only_active',True):
        outa  =  filt_active(parm_name,solver_inst,extra_kw=extra_kw)
        if not outa:
            log.debug(f'filt not active: {parm_name:>10} {attr_name:>15}| C:{False}\tV:{False}\tA:{False}\tO:{False} |\t{combos_in} in {solver_inst.combos} | {parm_name} in {slv_vars}')
            return False

    #Otherwise look at the combo filter, its its false return that
    outc = filt_combo_vars(parm_name,solver_inst, extra_kw,combos)
    outp = False
    #if the combo filter didn't explicitly fail, check the parm filter
    if attr_name in SLVR_SCOPE_PARM:
        outp = filt_parm_vars(parm_name,solver_inst, extra_kw)
        outr = any((outp,outc)) #redundant per above
    else:
        outr = outc
    

    fin =  bool(outr) and outa

    #if not fin:
    log.info(f'filter: {parm_name:>10} {attr_name:>15}| C:{outc}\tV:{outp}\tA:{outa}\tO:{fin} |\t{combos_in} in {solver_inst.combos} | {parm_name} in {slv_vars}| {extra_kw}')
    return fin 

#The KW Defaults for Solver
_slv_dflt_kwdict = dict(combos='*',ign_combos=None,only_combos=None,add_obj=True,_fail=True,slv_vars='*',add_vars=None,ign_vars=None,only_vars=None,only_active=True,activate=None,deactivate=None)
#TODO: implement add_vars feature, ie it creates a solver variable, or activates one if it doesn't exist from in system.heirarchy.format

# add to any SolvableMixin to allow solver use from its namespace
class SolverMixin(SolveableMixin):
    _run_id: str = None
    _solved = None
    _trans_opts = None

    _converged = False
    custom_solver = False
    solver_option = "minimize" #or root #TODO: implement equality solver as root

    convergence_threshold = 100 #changer per problem

    # Configuration Information
    @property
    def solved(self):
        return self._solved
    
    def solver_parameters(self,**kwargs):
        """applies the default combo filter, and your keyword arguments to the collect_solver_refs to test the ref / vars creations"""
        from engforge.solver import combo_filter
        return self.collect_solver_refs(check_atr_f=combo_filter,check_kw=kwargs)
    

    def post_run_callback(self, **kwargs):
        """user callback for when run is complete"""
        pass

    def pre_run_callback(self, **kwargs):
        """user callback for when run is beginning"""
        pass

    def run(self, *args, **kwargs):
        """the steady state run method for the system. It will run the system with the input parameters and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""

        self._iterate_input_matrix(self._run, *args, **kwargs)

    def _run(self, refs, icur, eval_kw=None, sys_kw=None, *args, **kwargs):
        """the steady state run method for the system. It will run the system with the input parameters and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""

        # TODO: what to do with eval / sys kw
        # TODO: option to preserve state
        Ref.refset_input(refs, icur)
        self.info(f"running with {icur}|{kwargs}")
        self.run_method(*args, **kwargs)
        self.debug(f"{icur} run time: {self._run_time}")

    def run_method(self, eval_kw=None, sys_kw=None, cb=None, **method_kw):
        """runs the case as currently defined by the system. This is the method that is called by the solver matrix to run the system with the input parameters. It will run the system with the input parameters and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""
        # set the values
        from engforge.system import System

        # Transeint
        self._run_start = datetime.datetime.now()
        if isinstance(self, System):
            self.system_references(
                recache=True
            )  # recache is important for iterators, #TODO: only recache when iterators are present
        # steady state
        if self._run_id is None:
            self._run_id = int(uuid.uuid4())
        # Recache system references
        self.eval(cb=cb, eval_kw=eval_kw, sys_kw=sys_kw, **method_kw)

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
            if hasattr(comp,'solver_override') and comp.solver_override:
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
        self, cb=None, eval_kw: dict = None, sys_kw: dict = None, *args, **kw
    ):
        """Evaluates the system with additional inputs for execute()
        :param cb: an optional callback taking the system as an argument of the form (self,eval_kw,sys_kw,*args,**kw)
        """

        if self.log_level < 20:
            self.debug(f"running with X: {self.X} & args:{args} kw:{kw}")

        # 0. run pre execute (signals=pre,both)
        self.pre_execute()

        # prep index
        self.index += 1

        # 1.Runs The Solver
        try:
            out = self.execute(*args, **kw)
        except Exception as e:
            self.error(e, f"solver failed")
            out = None
            raise e

        #Update
        self.update_flow(eval_kw,sys_kw,*args,**kw)

        # Save The Data
        self.save_data(index=self.index)

        if cb:
            cb(self, eval_kw, sys_kw, *args, **kw)

        return out
    
    def update_flow(self,eval_kw,sys_kw,*args,**kw):
        # 2. Update Internal Elements
        self.update_internal(eval_kw=eval_kw, *args, **kw)

        # run components and systems recursively
        self.run_internal_systems(sys_kw=sys_kw)

        # Post Execute (signals=post,both)
        self.post_execute()

        # Post Update Each Internal System
        self.post_update_internal(eval_kw=eval_kw, *args, **kw)        


    def pre_execute(self):
        """runs the solver of the system"""
        if self.log_level <= 10:
            self.msg(f"pre execute")

        # TODO: set system fields
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

    def execute(self, *args, **kw):
        """Solves the system's system of constraints and integrates transients if any exist

        Override this function for custom solving functions, and call `solver` to use default solver functionality.

        :returns: the result of this function is returned from eval()
        """
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
            :param dXdt: can be 0 to indicate steady-state, or None to not run the transient constraints. Otherwise a partial dictionary of parameters for the dynamics rates can be given, those not given will be assumed steady state or 0.

            :param kw: additional options for the solver, such as the solver_option, or the solver method options. Described below
            :param combos: a csv str or list of combos to include, including wildcards. the default means all combos will be run unless ign_combos or only combos alters behavior. The initial selection of combos is made by matching any case with the full name of the combo, or a parial name with a wildcard(s) in the combo name Ignore and only combos will further filter the selection. Wildcards / queries per fnmatch
            :param ign_combos: a list of combo parameters to ignore.
            :param only_combos: a list of combo parameters to include exclusively.
            :param add_var: a csv str or variables to include, including wildcards. the default means all combos will be run unless ign_combos or only combos alters behavior. The initial selection of combos is made by matching any case with the full name of the combo, or a parial name with a wildcard(s) in the combo name Ignore and only combos will further filter the selection. Wildcards / queries per fnmatch
            :param ign_var: a list of combo parameters to ignore.
            :param only_var: a list of combo parameters to include exclusively.            
            :param add_obj: a flag to add the objective to the system constraints, by default will add the objective to the system constraints. If False the objective will be the only constraint.        

            """
            from engforge.attr_solver import Solver

            #change references
            #Extract Extra Kws for our default solver
            extra_kw = self._rmv_extra_kws(kw,_slv_dflt_kwdict)
            add_obj = extra_kw.pop("add_obj", True)

            #TODO: Apply filter to solver attributes
            info = self.collect_solver_refs(check_atr_f=combo_filter,check_kw=extra_kw)
            updts = self.gather_update_refs()
            comps = info.get('comps',{})
            attrx = info.get('attrs',{})
            #print(info)

            #Collect Solver States
            Xref = sys_solver_variables(self,info,extra_kw=extra_kw,as_flat=True)
            if len(Xref) == 0:
                raise ValueError(f"no variables found for solver: {extra_kw}")
            
            Eqref = attrx.get("solver.eq",{})
            InEqref = attrx.get("solver.ineq",{})
            Yref = sys_solver_objectives(self,info,Xref,combo_kw=extra_kw)

            #Dynamic References
            #Time Integration
            #TODO: check parameter conflicts            
            #TODO: time solver integration
            Pref = attrx.get("time.parm",{})
            Dref = attrx.get("time.rate",{})
            Xdref = attrx.get("dynamics.state",{})
            dXdRefDt = attrx.get("dynamics.rate",{}) 

            # get the solver configuration
            has_constraints = True if len(Eqref) + len(InEqref) > 0 else False

            solve_mode = self.solver_option
            if "solver_option" in kw and kw["solver_option"] in SOLVER_OPTIONS:
                solve_mode = kw["solver_option"]
            elif "solver_option" in kw:
                raise ValueError(f"invalid solver option: {kw['solver_option']}")

            #enhance objective with obj_add for argmin(X)|(1+add_obj)*obj

            # Dynamics Configuration
            if dXdt is not None and dXdt is not False:
                if dXdt == 0:
                    pass
                elif dXdt is True:
                    pass
                else:  # dXdt is a dictionary
                    # TODO: add dynamics integration (steady: dXdt==0, dXdt = C)
                    assert isinstance(dXdt, dict), f"not dict for dXdt: {dXdt}"
                    raise NotImplementedError(
                        f"dynamic integration not yet implemented for dictionary based input: {dXdt}"
                    )  # TODO:
            else:
                # nope!
                Ytr = {}
                Xtr = {}


            # Initial States
            if X0 is not None:
                assert isinstance(X0, dict), f"wrong format for state: {X0}"
                X0 = X0
                # TODO: check if X0 is valid
            else:
                X0 = Xref

            parms = list(Xref.keys())
            Xg = Ref.refset_get(Xref)

            constraints = sys_solver_constraints(self,info,Xref,cons,extra_kw=extra_kw)
            con_list = constraints.get('info')
            

            #Minimizer Args
            if solve_mode == "minimize" or has_constraints:
                opts = {"tol": 1e-6, "method": "SLSQP"}
            else:
                raise NotImplementedError('minimize is the only solver option currently implemented')
                opts = {}

            #to your liking sir
            opts.update(kw)
            opts.update(**constraints) #constraints override kw per api spec 
            #TODO: allow kw to override constraints with add/rmv above

            # TODO: add basin hopping method in search_optimization
            output = {
                "Xstart": Xg,
                "parms": parms,
                "Xans": None,
                "dXdt": dXdt,
                "obj": obj,
                "add_obj": add_obj,
                "Yref": Yref,
                "update_methods": updts,
            }
            output['input_cons'] = constraints

            # Solve / Minimize
            #if solve_mode == "root" and not has_constraints and obj is None:
                #return self.solve_root(Xref, Yref, Xg, parms, output, **opts)
            if solve_mode == "minimize" or has_constraints:
                # handle threahold for success depending on if objective provided
                if "thresh" not in opts:
                    # default threshold
                    opts["thresh"] = self.convergence_threshold if not obj else None
                elif obj:
                    opts["thresh"] = None

                if obj:
                    if add_obj:
                        #normi = opts.pop("normalize", None)
                        ffunc = lambda *args, **kw: secondary_obj(obj, *args, **kw)
                        opts["ffunc"] = ffunc
                    else:
                        # TODO: define behavior
                        pass
                sol = self.solve_min(Xref, Yref, Xg, parms, output, **opts)

                x_in = [sol['Xans'][p] for p in parms]
                cur = refset_get(Xref)
                refset_input(Xref, sol['Xans'])
                #print(con_list,constraints['constraints'])

                Ycon = {}
                for c,k in zip(constraints['constraints'],con_list):
                    cv = c['fun'](x_in,self,{})
                    #print(c,k,cv)
                    Ycon[k] = cv
                output['Ycon'] = Ycon
                output['Yobj'] = {k:v.value(self,output) for k,v in Yref.items()}
                refset_input(Xref, cur)

                return sol
            
            else:
                self.warning(
                    f"no solution attempted! for {solve_mode} with {obj} and const: {constraints}"
                )

    def solve_min(
        self, Xref, Yref, Xreset, parms, output=None, fail=True, **kw
    ):
        """
        Solve the minimization problem using the given parameters. And sets the system state to the solution depending on input of 

        Solve the root problem using the given parameters.
        :param Xref: The reference input values.
        :param Yref: The reference output values.
        :param Xreset: The reset input values.
        :param parms: The list of parameter names.
        :param output: The output dictionary to store the results. (default: None)
        :param fail: Flag indicating whether to raise an exception if the solver doesn't converge. (default: True)
        :param kw: Additional keyword arguments.
        :return: The output dictionary containing the results.
        """
        thresh = kw.pop("thresh", self.convergence_threshold)

        if output is None:
            output = {
                "Xstart": Xreset,
                "parms": parms,
                "Xans": None,
                "success": None,
            }

        #print(f'solvemin {Xref} {Yref} {Xreset}')
        self._ans = refmin_solve(self, Xref, Yref, ret_ans=True, **kw)
        output["ans"] = self._ans
        de = abs(self._ans.fun)
        if self._ans.success and de < thresh if thresh else True:
            # Set Values
            Xa = {p: self._ans.x[i] for i, p in enumerate(parms)}
            output["Xans"] = Xa
            Ref.refset_input(Xref, Xa)
            self.pre_execute()
            self._converged = True
            output["success"] = True
        elif self._ans.success:
            # out of threshold condition
            Xa = {p: self._ans.x[i] for i, p in enumerate(parms)}
            output["Xans"] = Xa
            self.warning(
                f"solver didnt fully solve equations! {self._ans.x} -> residual: {self._ans.fun}"
            )
            Ref.refset_input(Xref, Xreset)
            self.pre_execute()
            self._converged = False
            output["success"] = False  # only false with threshold
        else:
            Ref.refset_input(Xref, Xreset)
            self.pre_execute()
            self._converged = False
            if fail:
                raise Exception(f"solver didnt converge: {self._ans}")
            output["success"] = False

        return output

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
    

