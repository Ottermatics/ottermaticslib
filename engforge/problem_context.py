"""The ProblemExec provides a uniform set of options for managing the state of the system and its solvables, establishing the selection of combos or de/active attributes to Solvables. Once once created any further entracnces to ProblemExec will return the same instance until finally the last exit is called. 

The ProblemExec class allows entrance to a its context to the same instance until finally the last exit is called. The first entrance to the context will create the instance, each subsequent entrance will return the same instance. The ProblemExec arguments are set the first time and remove keyword arguments from the input dictionary (passed as a dict ie stateful) to subsequent methods. 
This isn't technically a singleton pattern, but it does provide a similar interface. Instead mutliple problem instances will be clones of the first instance, with the optional difference of input/output/event criteria. The first instance will be returned by each context entry, so for that reason it may always appear to have same instance, however each instance is unique in a recusive setting so it may record its own state and be reverted to its own state as per the options defined.

#TODO: allow update of kwargs on re-entrance

## Example:
.. code-block:: python

    #Application code (arguments passed in kw)
    with ProblemExec(sys,combos='default',slv_vars'*',**kw) as pe:
        pe.sys_refs #get the references and compiled problem
        for i in range(10):
            pe.solve_min(pe.Xref,pe.Yref,**other_args)
            pe.set_checkpoint() #save the state of the system 
            self.save_data()
            

    #Solver Module (can use without knowledge of the runtime system)
    with ProblemExec(sys,{},Xnew=Xnext,ctx_fail_new=True) as pe:
        #do revertable math on the state of the system without concern for the state of the system
...

# Combos Selection
By default no arguments run will select all active items from all combos. The `combos` argument can be used to select a specific set of combos, a outer select. From this set, the `ign_combos` and `only_combos` arguments can be used to ignore or select specific combos based on exclusion or inclusion respectively.

# Parameter Name Selection
The `slv_vars` argument can be used to select a specific set of solvables. From this set, the `ign_vars` and `only_vars` arguments can be used to ignore or select specific solvables based on exclusion or inclusion respectively. The `add_vars` argument can be used to add a specific set of solvables to the solver.

# Active Mode Handiling
The `only_active` argument can be used to select only active items. The `activate` and `deactivate` arguments can be used to activate or deactivate specific solvables.

`add_obj` can be used to add an objective to the solver. 

The `_fail` argument can be used to raise an error if no solvables are selected.

# Exit Mode Handling

The ProblemExec supports the following exit mode handling parameters:

- `fail_revert`: Whether to raise an error if no solvables are selected. Default is True.
- `revert_last`: Whether to revert the last change. Default is True.
- `revert_every`: Whether to revert every change. Default is True.
- `exit_on_failure`: Whether to exit on first failure. Default is True.

These parameters control the behavior of the ProblemExec when an error occurs or when no solvables are selected.

"""

#TODO: define the 

from engforge.logging import LoggingMixin
from engforge.system_reference import Ref
from engforge.solver_utils import *

class ProbLog(LoggingMixin): pass
log = ProbLog()

import fnmatch
import uuid

#TODO: implement add_vars feature, ie it creates a solver variable, or activates one if it doesn't exist from in system.heirarchy.format
#TODO: define the dataframe / data sotrage feature



#The KW Defaults for Solver via kw_dict
slv_dflt_options = dict(combos='*',ign_combos=None,only_combos=None,add_obj=True,slv_vars='*',add_vars=None,ign_vars=None,only_vars=None,only_active=True,activate=None,deactivate=None)
#KW Defaults for context from **opts
dflt_parse_kw = dict(fail_revert=True,revert_last=True,revert_every=True,exit_on_failure=True, pre_exec=True,post_exec=True,raise_on_opt_failure = True,level_name='top',post_callback=None,convergence_threshold=10)

#TODO: output options extend_dataframe=True,return_dataframe=True

#Special exception classes handled in exit
class ProblemExit(Exception):
    """an exception to exit the problem context, without error"""
    revert:bool
    def __init__(self,revert:bool=None):
        self.revert = revert

    def __str__(self) -> str:
        return f'ProblemExit[revert={self.revert}]'        

class ProblemExitAtLevel(ProblemExit):
    """an exception to exit the problem context, without error"""
    level: str
    def __init__(self,level:str,revert=None):
        assert level is not None, 'level must be defined'
        assert isinstance(level,str), 'level must be a string'
        self.level = level
        self.revert = revert

    def __str__(self) -> str:
        return f'ProblemExit[level={self.level},revert={self.revert}]'



class ProblemExec:
    """
    Represents the execution context for a problem in the system. The ProblemExec class provides a uniform set of options for managing the state of the system and its solvables, establishing the selection of combos or de/active attributes to Solvables. Once once created any further entracnces to ProblemExec will return the same instance until finally the last exit is called.
    """
    _class_cache = None #ProblemExec is assigned below

    system: "System"
    _session: "ProblemExec"

    #Store refs for later
    _update_refs: dict 
    _post_update_refs: dict

    #runtime and exit options
    convergence_threshold = 10
    pre_exec: bool=True
    post_exec: bool=False
    fail_revert: bool = True
    revert_last: bool = True
    revert_every: bool = True
    exit_on_failure: bool = True
    raise_on_opt_failure: bool = True

    #exit & level definition
    post_callback: callable = None #callback that will be called on the system each time it is reverted, it should take args(system,current_problem_exec)
    level_name: str = None #target this context with the level name
    level_number: int = 0 #TODO: keep track of level on the global context
    #TODO: add an exit system to handle exiting multiple levels 
    #TODO: add an exit system that allows preservation of state for each level
    #TODO: exit system should abide by update / signals options

    #solver references raw
    sys_refs: dict
    slv_kw: dict

    #collections attrs expanded
    attrs: dict
    type: dict
    comp: dict

    #_sessions: dict = {} #TODO: support multiple systems
    X_start:dict = None
    X_end:dict = None

    

    
    def __init__(self,system,kw_dict=None,Xnew=None,ctx_fail_new=False,**opts):
        """
        Initializes the ProblemExec.

        #TODO: provide data storage options for dataframe / table storage history/ record keeping (ss vs transient data)
        
        #TODO: create an option to copy the system and run operations on it, and options for applying the state from the optimized copy to the original system

        :param system: The system to be executed.
        :param Xnew: The new state of the system to set wrt. reversion, optional
        :param ctx_fail_new: Whether to raise an error if no execution context is available, use in utility methods ect. Default is False.
        :param kw_dict: A keyword argument dictionary to be parsed for solver options, and removed from the outer context. Changes are made to this dictionary, so they are removed automatically from the outer context, and thus no longer passed to interior parameters.

        :param combos: The selection of combos. Default is '*' (select all).
        :param  ign_combos: The combos to be ignored.
        :param  only_combos: The combos to be selected.
        :param  add_obj: Whether to add an objective to the solver. Default is True.
        :param  slv_vars: The selection of solvables. Default is '*' (select all).
        :param  add_vars: The solvables to be added to the solver.
        :param  ign_vars: The solvables to be ignored.
        :param  only_vars: The solvables to be selected.
        :param  only_active: Whether to select only active items. Default is True.
        :param  activate: The solvables to be activated.
        :param  deactivate: The solvables to be deactivated.
        :param  fail_revert: Whether to raise an error if no solvables are selected. Default is True.
        :param  revert_last: Whether to revert the last change. Default is True.
        :param  revert_every: Whether to revert every change. Default is True.
        :param  exit_on_failure: Whether to exit on failure, or continue on. Default is True.
        """
        #print(system)
        #parse the options to change behavior of the context
        level_name = opts.pop('level_name',None) #get arg if not set

        opt_in,opt_out = {},{}
        if opts:
            opt_in = {k:v for k,v in opts.items() if k in dflt_parse_kw}
            opt_out = {k:v for k,v in opts.items() if k not in opt_in}

        if kw_dict is None:
            kw_dict = {}
        
            
        if hasattr(ProblemExec,'_session'):
            #TODO: update options for this instance
            #TODO: record option to revert every change (or just on last change)
            #error if the system is different (it shouldn't be!)
                        
            #mirror the state of session (exactly)
            self.__dict__ = self._class_cache._session.__dict__.copy()
            if self.system is not system:
                raise Exception(f'somethings wrong! change of comp! {self.system} -> {system}')

            #modify things from the input
            self.level_name = level_name
            if opt_in: self.__dict__.update(opt_in) #level options ASAP
            self._temp_state = Xnew #input state exception to this
            if log.log_level < 5:
                self.msg(f'[{self.level_number}-{self.level_name}]setting execution context with {opt_in}| {opt_out}')
            
            #each state request to be reverted, then we need to store the state of each execution context overriding the outer context X_start
            self.set_checkpoint()

        elif ctx_fail_new:
            raise Exception(f'no execution context available')
        
        else:
            #add the prob options to the context
            self.__dict__.update(opt_in)
            #supply the level name default as top if not set
            if self.level_name is None:
                self.level_name = 'top'

            self._temp_state = Xnew
            self.name = system.name + '-' + str(uuid.uuid4())[:8]
            self.establish_system(system,kw_dict=kw_dict,**opt_out)
            
            #Finally we record where we started!
            self.set_checkpoint()        

            self.msg(f'[{self.level_number}-{self.level_name}]new execution context for {system}| {self.slv_kw}')

    def establish_system(self,system,kw_dict,**kwargs):
        """caches the system references, and parses the system arguments"""
        from engforge.components import SolveableInterface
        from engforge.system import System 

        #pass args without creating singleton (yet)
        if log.log_level < 5:
            self.info(f'establish {system} {kw_dict} {kwargs}')
        self.system = system
        assert isinstance(self.system,SolveableInterface), 'only solveable interfaces are supported for execution context'

        #Recache system references
        if isinstance(self.system, System):
            #get the fresh references!
            self.system.system_references(recache=True)
        
        #Extract solver parameters and set them on this object, they will be distributed to any new further execution context's via monkey patch above
        in_kw = self.get_extra_kws(kwargs,slv_dflt_options,use_defaults=False)
        self.slv_kw = self.get_extra_kws(kw_dict,slv_dflt_options,rmv=True)
        self.slv_kw.update(in_kw) #update with input!

        self.sys_refs = self.system.solver_parameters(**self.slv_kw)

        #Grab inputs and set to system
        for k,v in dflt_parse_kw.items():
            if k in self.slv_kw:
                setattr(self,k,self.slv_kw[k])

        #Get the update method refs
        self._update_refs = self.system.gather_update_refs()
        self._post_update_refs = self.system.gather_post_update_refs()
        
        #Problem Variable Definitions
        self.Xref = self.sys_solver_variables()
        self.Yref = self.sys_solver_objectives()
        cons = {} #TODO: parse additional constraints
        self.constraints = self.sys_solver_constraints(self.Xref,cons)

        if log.log_level < 5:
            self.msg(f'established {self.slv_kw}')  

    #Context Manager Interface
    def __enter__(self):
        #Set the new state
        self.activate_temp_state()
        
        if self.pre_exec:
            self.pre_execute()

        #Check for existing session        
        if hasattr(ProblemExec,'_session'):
            self.msg(f'[{self.level_number}-{self.level_name}][{self.level_number}-{self.level_name}] entering existing execution context')
            if not isinstance(self,self._class_cache._session.__class__):
                self.warning(f'change of execution class!')
            #global level number
            self._class_cache._session.__class__.level_number += 1        
            return self._class_cache._session
        
        #return New
        self._class_cache._session = self
        self._class_cache._session.__class__.level_number = 0
   
        self.debug(f'[{self.level_number}-{self.level_name}] creating execution context for {self.system}')
        



        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #define exit action, to handle the error here return True. Otherwise error propigates up to top level      

        #Last opprotunity to update the system
        if self.post_exec:
            self.post_execute()

        if self.post_callback:
            self.post_callback()          

        #Exit Scenerio (boolean return important for context manager exit handling in heirarchy)
        if isinstance(exc_value,ProblemExit):     
            
            #first things first
            if exc_value.revert:
                self.revert_to_start()
                if self.pre_exec:
                    self.pre_execute()
            else:
                #A hack to edit the session state.
                Xcur = self.record_state    
                self._class_cache._session.X_start = Xcur           

            #Decide our exit conditon (if we should exit)
            if isinstance(exc_value,ProblemExitAtLevel):
                #should we stop?
                if exc_value.level == self.level_name:
                    self.debug(f'[{self.level_number}-{self.level_name}] exit at level {exc_value}')
                    ext = True
                else:
                    self.msg(f'[{self.level_number}-{self.level_name}] exit not at level {exc_value}')
                    ext = False

                #Check if we missed a level name and its the top level, if so then we raise a real error!
                if self._class_cache._session is self and not ext:
                    #never ever leave the top level without deleting the session
                    self._class_cache._session.__class__.level_number = 0
                    del self._class_cache._session 
                    raise KeyError(f'cant exit to level! {exc_value.level} not found!!')
            
            #TODO: expand this per options
            else:
                self.info(f'[{self.level_number}-{self.level_name}] problem exit revert={exc_value.revert}')

                ext = True

        elif exc_type is not None:
            ext = self.error_action(exc_value)
        else:
            ext = self.exit_action()

        self.clean_context()

        return ext
    
    #Multi Context Exiting:
    def exit_with_state(self):
        raise ProblemExit(revert=False)
    
    def exit_and_revert(self):
        raise ProblemExit(revert=True)

    def exit_to_level(self,level:str,revert=False):
        raise ProblemExitAtLevel(level=level,revert=revert)

    def exit_action(self):
        """handles the exit action wrt system"""
        if self.revert_last and self._class_cache._session is self:
            self.revert_to_start()
            #run execute 
            if self.pre_exec:
                self.pre_execute()

        elif self.revert_every:
            self.revert_to_start()
            #run execute 
            if self.pre_exec:
                self.pre_execute()

        #TODO: add exit on success option
        return True #continue as normal

    def error_action(self,error):
        """handles the error action wrt to the problem"""
        self.debug(f'[{self.level_number}-{self.level_name}] with input: {self.kwargs}')
        
        if self.fail_revert:
            self.revert_to_start()

        if self.exit_on_failure:
            self.error(error,f'error in execution context')
            return False #send me up
        else:
            self.warning(f'error in execution context: {error}')

        return True #our problem will go on
    
    def clean_context(self):
        if hasattr(ProblemExec,'_session') and self._class_cache._session is self:
            #TODO: restore state from `X_start` 
            self.debug(f'[{self.level_number}-{self.level_name}] closing execution session')
            self._class_cache._session.__class__.level_number = 0
            del self._class_cache._session
        elif hasattr(self._class_cache,'_session'):
            #global level number
            self._class_cache._session.__class__.level_number -= 1  
             

    #State Interfaces
    @property
    def record_state(self)->dict:
        """records the state of the system"""
        #refs = self.all_variable_refs
        refs = self.all_variables
        return Ref.refset_get(refs)     

    def set_checkpoint(self):
        """sets the checkpoint"""
        self.X_start = self.record_state
        if log.log_level < 5:
            self.debug(f'[{self.level_number}-{self.level_name}] set checkpoint: {list(self.X_start.values())}')
        

    def revert_to_start(self):
        if log.log_level < 5:
            xs = list(self.X_start.values())
            rs = list(self.record_state.values())
            self.debug(f'[{self.level_number}-{self.level_name}] reverting to start: {xs} -> {rs}')
        Ref.refset_input(self.all_variables,self.X_start)

    def activate_temp_state(self):
        if self._temp_state:
            Ref.refset_input(self.all_variables,self._temp_state)

            
    #System Events
    def apply_pre_signals(self):
        """applies all pre signals"""
        for signame, sig in self.signals.items():
            if sig.mode == "pre" or sig.mode == "both":
                sig.apply()

    def apply_post_signals(self):
        """applies all post signals"""
        for signame, sig in self.signals.items():
            if sig.mode == "post" or sig.mode == "both":
                sig.apply()

    def update_system(self,*args,**kwargs):
        """updates the system"""
        for ukey,uref in self._update_refs.items():
            uref.value(*args,**kwargs)

    def post_update_system(self,*args,**kwargs):
        """updates the system"""
        for ukey,uref in self._post_update_refs.items():
            uref.value(*args,**kwargs)

    def pre_execute(self,*args,**kwargs):
        """Updates the pre/both signals after the solver has been executed. This is useful for updating the system state after the solver has been executed."""
        if log.log_level < 5:
            self.msg(f"pre execute")
        self.apply_pre_signals()
        self.update_system(*args,**kwargs)


    def post_execute(self,*args,**kwargs):
        """Updates the post/both signals after the solver has been executed. This is useful for updating the system state after the solver has been executed."""
        if log.log_level < 5:
            self.msg(f"post execute")
        self.apply_post_signals()
        self.post_update_system(*args,**kwargs)

    def solve_min(
        self,Xref,Yref,output=None,**kw
    ):
        """
        Solve the minimization problem using the given parameters. And sets the system state to the solution depending on input of 

        Solve the root problem using the given parameters.
        :param Xref: The reference input values.
        :param Yref: The reference objective values to minimize.
        :param output: The output dictionary to store the results. (default: None)
        :param fail: Flag indicating whether to raise an exception if the solver doesn't converge. (default: True)
        :param kw: Additional keyword arguments.
        :return: The output dictionary containing the results.
        """

        thresh = kw.pop("thresh", self.convergence_threshold)

        dflt = {
                "Xstart": Ref.refset_get(Xref),
                "Ystart": Ref.refset_get(Yref,sys=self.system,info=self),
                "Xans": None,
                "success": None,
                "Xans":None,
                "Yobj":None,
                "Ycon":None,
            }

        if output:
            dflt.update(output)
            output = dflt
            

        #override constraints input
        kw.update(self.constraints)
        

        self._ans = refmin_solve(self.system, Xref, Yref, ret_ans=True, **kw)
        output["ans"] = self._ans

        self.handle_solution(self._ans,Xref,Yref,self.X_start,output)

        return output

    def handle_solution(self,answer,Xref,Yref,Xreset,output):
        #TODO: move exit condition handiling somewhere else, reduce cross over from process_ans
        thresh = self.convergence_threshold
        parms = list(Xref)

        #Output Results
        Xa = {p: answer.x[i] for i, p in enumerate(parms)}
        output["Xans"] = Xa
        Ref.refset_input(Xref,Xa)

        Yout = {p: Yref[p].value(self.system,self) for p in Yref}
        output["Yobj"] = Yout

        Ycon = {}
        if self.constraints['constraints']:
            x_in = answer.x
            for c,k in zip(self.constraints['constraints'],self.constraints['info']):
                cv = c['fun'](x_in,self,{})
                Ycon[k] = cv
        output['Ycon'] = Ycon   

        de = answer.fun
        if answer.success and de < thresh if thresh else True:
            # Set Values

            # Ref.refset_input(Xref, Xa)
            # self.pre_execute()
            self.system._converged = True
            output["success"] = True

        elif answer.success:
            # out of threshold condition
            self.warning(
                f"solver didnt fully solve equations! {answer.x} -> residual: {answer.fun}"
            )
            # Ref.refset_input(Xref, Xreset)
            # self.pre_execute()
            self.system._converged = False
            output["success"] = False  # only false with threshold

        else:
            # Ref.refset_input(Xref, Xreset)
            # self.pre_execute()
            self.system._converged = False
            if self.raise_on_opt_failure:
                raise Exception(f"solver didnt converge: {answer}")
            output["success"] = False
    
        return output
    
    #Function assembly
    def sys_solver_variables(self):
        """gathers variables from solver vars, and attempts to locate any input_vars to add as well. use exclude_vars to eliminate a variable from  the solver
        """
        out = dict(dynamics=self.dynamic_state,integration=self.integrator_vars,variables=self.problem_vars)
        
        flt = {}
        for k,v in out.items():
            flt.update(v)
        return flt

    def sys_solver_objectives(self,**kw):
        """gathers variables from solver vars, and attempts to locate any input_vars to add as well. use exclude_vars to eliminate a variable from  the solver
        """
        sys_refs = self.sys_refs

        #Convert result per kind of objective (min/max ect)
        objs = sys_refs.get('attrs',{}).get('solver.obj',{})
        return {k:v for k,v in objs.items()}

    def sys_solver_constraints(self,Xrefs,add_con=None,combo_filter=True, *args, **kw):
        """formatted as arguments for the solver
        """
        from engforge.solver_utils import create_constraint

        system = self.system
        sys_refs = self.sys_refs

        extra_kw = self.kwargs
        
        #TODO: move to kwarg parsing on setup
        deactivated = ext_str_list(extra_kw,'deactivate',[]) if 'deactivate' in extra_kw and extra_kw['deactivate'] else []
        activated = ext_str_list(extra_kw,'activate',[]) if 'activate' in extra_kw and  extra_kw['activate'] else []

        slv_inst = sys_refs.get('type',{}).get('solver',{})
        sys_refs = sys_refs.get('attrs',{})

        if add_con is None:
            add_con = {}
        

        #The official definition of X parameter order
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
            if log.log_level < 7:
                self.debug(f'[{self.level_number}-{self.level_name}] constraints {slvr} {slv_constraints}')
            for ctype in slv_constraints:
                cval = ctype['value']
                kind = ctype['type']       
                parm = ctype['parm']
                if log.log_level < 3:
                    self.msg(f'[{self.level_number}-{self.level_name}]const: {slvr} {ctype}')
                if cval is not None and slvr in Xparms:
                    
                    #Check for combos & activation
                    combos = None
                    if 'combos' in ctype:
                        combos = ctype['combos']
                        combo_parm = ctype['combo_parm']
                        active = ctype.get('active',True)
                        in_activate = any([arg_parm_compare(combo_parm,v) for v in activated]) if activated else False
                        in_deactivate = any([arg_parm_compare(combo_parm,v) for v in deactivated]) if deactivated else False

                        #Check active or activated
                        if not active and not activated:
                            if log.log_level < 3:
                                self.msg(f'[{self.level_number}-{self.level_name}]skip con: inactive {parm} {slvr} {ctype}')
                            continue
                        elif not active and not in_activate:
                            if log.log_level < 3:
                                self.msg(f'[{self.level_number}-{self.level_name}]skip con: inactive {parm} {slvr} {ctype}')
                            continue

                        elif active and in_deactivate:
                            if log.log_level < 3:
                                self.msg(f'[{self.level_number}-{self.level_name}]skip con: deactivated {parm} {slvr} ')
                            continue

                        

                    if combos and combo_filter:
                        filt = filt_combo_vars(combo_parm,slv, extra_kw,combos)
                        if not filt:
                            if log.log_level < 5:
                                self.debug(f'[{self.level_number}-{self.level_name}] filtering constraint={filt} {parm} {slv} {ctype} | {combos} {ext_str_list(extra_kw,"combos",None)}')                        
                            continue
                    
                    if log.log_level < 10:
                        self.debug(f'[{self.level_number}-{self.level_name}] adding var constraint {parm,slvr,ctype,combos}') 

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
                        self.warning(f"bad constraint: {cval} {kind} {slvr}")

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

    # General method to distribute input to internal components
    @classmethod
    def parse_default(self,key,defaults,input_dict,rmv=False):
        """splits strings or lists and returns a list of options for the key, if nothing found returns None if fail set to True raises an exception, otherwise returns the default value"""
        if key in input_dict:
            #kwargs will no longer have key!
            if not rmv:
                option = input_dict.get(key)
            else:
                option = input_dict.pop(key)
            #print(f'removing option {key} {option}')
            if option is None:
                return option,False
            elif isinstance(option,(int,float,bool)):
                return option,False
            elif isinstance(option,str):
                if not option:
                    return None,False
                option = option.split(',')
            else:
                if input_dict.get('_fail',True):
                    assert isinstance(option,list),f"str or list combos: {option}"
                    #this is OK!
                else:
                    log.warning(f'bad option {option} for {key}| {input_dict}')
                    
            return option,False
        elif key in defaults:
            return defaults[key],True
        return None,None
        

    @classmethod
    def get_extra_kws(cls,kwargs,_check_keys:dict=slv_dflt_options,rmv=False,use_defaults=True):
        """extracts the combo input from the kwargs"""
        # extract combo input
        if not _check_keys:
            return {}
        _check_keys = _check_keys.copy()
        #TODO: allow extended check_keys / defaults to be passed in, now every value in check_keys has a default
        cur_in = kwargs
        output = {}
        for p,dflt in _check_keys.items():
            val,is_dflt = cls.parse_default(p,_check_keys,cur_in,rmv=rmv)
            if not is_dflt:
                output[p] = val
            elif use_defaults:
                output[p] = val
            if rmv:
                cur_in.pop(p,None)

        #copy from data
        filtr = dict(list(filter(lambda kv: kv[1] is not None or kv[0] in _check_keys, output.items())))
        #print(f'got {combos} -> {comboos} from {kwargs} with {_check_keys}')
        return filtr              

    #Logging to class logger
    @property
    def identity(self):
        return f'PROB|{self.name}'

    def msg(self,msg):
        if log.log_level < 5:
            log.msg(f'{self.identity}| {msg}')

    def debug(self,msg):
        if log.log_level <= 15:
            log.debug(f'{self.identity}| {msg}')

    def warning(self,msg):
        log.warning(f'{self.identity}| {msg}')

    def info(self,msg):
        log.info(f'{self.identity}| {msg}')

    def error(self,error,msg):
        log.error(error,f'{self.identity}| {msg}')        

    def critical(self,msg):
        log.critical(f'{self.identity}| {msg}')

    #Safe Access Methods
    @property
    def ref_attrs(self):
        return self.sys_refs.get('attrs',{})

    @property
    def attr_inst(self):
        return self.sys_refs.get('type',{})

    @property
    def kwargs(self):
        return self.slv_kw

    #X solver variables
    @property
    def problem_vars(self):
        return self.ref_attrs.get('solver.var',{})   

    @property
    def dynamic_state(self):
        return self.ref_attrs.get('dynamics.state',{})
    
    @property
    def dynamic_rate(self):
        return self.ref_attrs.get('dynamics.rate',{})    
    
    @property
    def problem_input(self):
        return self.ref_attrs.get('dynamics.input',{})      

    @property
    def integrator_vars(self):
        return self.ref_attrs.get('time.parm',{})
    
    @property
    def integrator_rates(self):
        return self.ref_attrs.get('time.rate',{})
    
    #TODO: expose optoin for saving all or part of the system information, for now lets default to all (saftey first, then performance :)
    @property
    def all_variable_refs(self)->dict:
        ing = self.integrator_vars
        stt = self.dynamic_state
        vars = self.problem_vars
        return {**ing,**stt,**vars}
    
    @property
    def all_variables(self)->dict:
        return self.system.system_references()['attributes']
    
    #Y solver variables
    @property
    def problem_objs(self):
        return self.ref_attrs.get('solver.obj',{})   

    @property
    def problem_eq(self):
        return self.ref_attrs.get('solver.eq',{})
    
    @property
    def problem_ineq(self):
        return self.ref_attrs.get('solver.ineq',{})
    
    @property
    def signals_source(self):
        return self.ref_attrs.get('signal.source',{})
    
    @property
    def signals_target(self):
        return self.ref_attrs.get('signal.target',{})

    @property
    def signals(self):
        return self.ref_attrs.get('signal.signal',{})       
    
#subclass before altering please!
ProblemExec._class_cache = ProblemExec


