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

The ProblemExec supports the following exit mode handling vars:

- `fail_revert`: Whether to raise an error if no solvables are selected. Default is True.
- `revert_last`: Whether to revert the last change. Default is True.
- `revert_every`: Whether to revert every change. Default is True.
- `exit_on_failure`: Whether to exit on first failure. Default is True.

These vars control the behavior of the ProblemExec when an error occurs or when no solvables are selected.

"""

#TODO: define the 

from engforge.logging import LoggingMixin
from engforge.system_reference import Ref
from engforge.solver_utils import *

class ProbLog(LoggingMixin): pass
log = ProbLog()

import uuid

#TODO: implement add_vars feature, ie it creates a solver variable, or activates one if it doesn't exist from in system.heirarchy.format
#TODO: define the dataframe / data storage feature



#The KW Defaults for Solver via kw_dict
slv_dflt_options = dict(combos='default',ign_combos=None,only_combos=None,add_obj=True,slv_vars='*',add_vars=None,ign_vars=None,only_vars=None,only_active=True,activate=None,deactivate=None,dxdt=None)
#KW Defaults for context from **opts
dflt_parse_kw = dict(fail_revert=True,revert_last=True,revert_every=True,exit_on_failure=True, pre_exec=True,post_exec=True,raise_on_opt_failure = False,level_name='top',post_callback=None,success_thresh=10)

#TODO: output options extend_dataframe=True,return_dataframe=True



#Special exception classes handled in exit
class IllegalArgument(Exception):
    """an exception to exit the problem context as specified"""
    pass

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
        self.level = level.lower()
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
    _session_id = None
    entered: bool = False
    exited: bool = False

    #Store refs for later
    _update_refs: dict 
    _post_update_refs: dict

    #runtime and exit options
    success_thresh = 10
    pre_exec: bool = True
    post_exec: bool = True
    fail_revert: bool = True
    revert_last: bool = True
    revert_every: bool = True
    exit_on_failure: bool = True
    raise_on_opt_failure: bool = True

    #exit & level definition
    post_callback: callable = None #callback that will be called on the system each time it is reverted, it should take args(system,current_problem_exec)
    level_name: str = None #target this context with the level name
    level_number: int = 0 #TODO: keep track of level on the global context

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
    _dxdt = None #numeric/None/dict (True?)

        
    def __init__(self,system,kw_dict=None,Xnew=None,ctx_fail_new=False,**opts):
        """
        Initializes the ProblemExec.
        
        #TODO: exit system should abide by update / signals options

        #TODO: provide data storage options for dataframe / table storage history/ record keeping (ss vs transient data)
        
        #TODO: create an option to copy the system and run operations on it, and options for applying the state from the optimized copy to the original system

        :param system: The system to be executed.
        :param Xnew: The new state of the system to set wrt. reversion, optional
        :param ctx_fail_new: Whether to raise an error if no execution context is available, use in utility methods ect. Default is False.
        :param kw_dict: A keyword argument dictionary to be parsed for solver options, and removed from the outer context. Changes are made to this dictionary, so they are removed automatically from the outer context, and thus no longer passed to interior vars.
        :param dxdt: The dynamics integration method. Default is None meaning that dynamic vars are not considered for minimization unless otherwise specified. Steady State can be specified by dxdt=0 all dynamic vars are considered as solver variables, with the constraint that their rate of change is zero. If a dictionary is passed then the dynamic vars are considered as solver variables, with the constraint that their rate of change is equal to the value in the dictionary, and all other unspecified rates are zero (steady).

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
        if self.log_level < 15:
            if hasattr(ProblemExec,'_session'):
                self.debug(f'subctx{self.level_number}|  keywords: {kw_dict} and misc: {opts}')
            else:
                self.debug(f'context| keywords: {kw_dict} and misc: {opts}')

        #parse the options to change behavior of the context
        if opts and 'level_name' in opts:
            level_name = opts.pop('level_name').lower()
        elif kw_dict and 'level_name' in kw_dict:
            level_name = kw_dict.pop('level_name').lower()
        else:
            level_name = None

        #Merge kwdict(stateful) and opts (current level)
        opt_in,opt_out = {},{}
        if opts:
            #these go to the context instance optoins
            opt_in = {k:v for k,v in opts.items() if k in dflt_parse_kw}
            #these go to system establishment
            opt_out = {k:v for k,v in opts.items() if k not in opt_in}

        if kw_dict is None:
            kw_dict = {}
        else:
            #these go to the context instance optoins
            kw_in = {k:v for k,v in kw_dict.items() if k in dflt_parse_kw}
            kw_out = {k:v for k,v in kw_dict.items() if k not in opt_in}
            opt_in.update(kw_in)
            #these go to system establishment
            opt_out.update(kw_out)
            #remove from kw_dict
            for k in kw_in.keys():
                kw_dict.pop(k)


        #Define the handiling of rate integrals
        if 'dxdt' in opts:
            dxdt = opts.pop('dxdt')
        elif 'dxdt' in kw_dict:
            dxdt = kw_dict.pop('dxdt')
        else:
            dxdt = None        
        if dxdt is not None and dxdt is not False:
            if dxdt == 0 and dxdt is not False:
                pass
            elif dxdt is True:
                pass
            elif isinstance(dxdt, dict):  # dxdt is a dictionary
                #provide a set of values or function to have the solver solve for
                pass
            else:
                raise IllegalArgument(f'bad dxdt value {dxdt}')
                
        self._dxdt = dxdt                 
            
        if hasattr(ProblemExec,'_session'):
            #TODO: update options for this instance
            #TODO: record option to revert every change (or just on last change)
            #error if the system is different (it shouldn't be!)
                        
            #mirror the state of session (exactly)
            self.__dict__ = self._class_cache._session.__dict__.copy()
            if self.system is not system:
                raise IllegalArgument(f'somethings wrong! change of comp! {self.system} -> {system}')

            #modify things from the input
            if level_name is None:
                self.level_name = 'ctx_'+str(int(uuid.uuid4()))[0:15]
            else:
                self.level_name = level_name

            if opt_in: self.__dict__.update(opt_in) #level options ASAP
            self._temp_state = Xnew #input state exception to this
            if log.log_level < 5:
                self.msg(f'[{self.level_number}-{self.level_name}]setting execution context with {opt_in}| {opt_out}')
            
            #each state request to be reverted, then we need to store the state of each execution context overriding the outer context X_start
            self.set_checkpoint()

        elif ctx_fail_new:
            raise IllegalArgument(f'no execution context available')
        
        else:
            #add the prob options to the context
            self.__dict__.update(opt_in)
            #supply the level name default as top if not set
            if level_name is None:
                self.level_name = 'top'
            else:
                self.level_name = level_name

            self._temp_state = Xnew
            self.name = system.name + '-' + str(uuid.uuid4())[:8]
            self.establish_system(system,kw_dict=kw_dict,**opt_out)
            
            #Finally we record where we started!
            self.set_checkpoint()        

        if log.log_level < 10:
            self.info(f'[{self.level_number}-{self.level_name}]new execution context for {system}| {opts} | {self.slv_kw}')            
        elif log.log_level <= 3:
            self.msg(f'[{self.level_number}-{self.level_name}]new execution context for {system}| {self.slv_kw}')

    def establish_system(self,system,kw_dict,**kwargs):
        """caches the system references, and parses the system arguments"""
        from engforge.components import SolveableInterface
        from engforge.system import System 

        #pass args without creating singleton (yet)
        self._session_id = uuid.uuid4()        
        if log.log_level < 5:
            self.info(f'establish {system}| {kw_dict} {kwargs}')

        self.system = system
        assert isinstance(self.system,SolveableInterface), 'only solveable interfaces are supported for execution context'
        self.system._last_context = self

        #Recache system references
        if isinstance(self.system, System):
            #get the fresh references!
            self.system.system_references(recache=True)
        
        #Extract solver vars and set them on this object, they will be distributed to any new further execution context's via monkey patch above
        in_kw = self.get_extra_kws(kwargs,slv_dflt_options,use_defaults=False)
        self.slv_kw = self.get_extra_kws(kw_dict,slv_dflt_options,rmv=True)
        self.slv_kw.update(in_kw) #update with input!

        self.sys_refs = self.system.solver_vars(**self.slv_kw)

        #Grab inputs and set to system
        for k,v in dflt_parse_kw.items():
            if k in self.slv_kw:
                setattr(self,k,self.slv_kw[k])

        #Get the update method refs
        self._update_refs = self.system.collect_update_refs()
        #TODO: find subsystems that are not-subsolvers and execute them
        self._post_update_refs = self.system.collect_post_update_refs()
        
        #Problem Variable Definitions
        #self.solver_vars = self.sys_solver_variables()
        self.Xref = self.problem_opt_vars
        self.Yref = self.sys_solver_objectives()
        cons = {} #TODO: parse additional constraints
        self.constraints = self.sys_solver_constraints(self.Xref,cons)

        if log.log_level < 5:
            self.msg(f'established sys context: {self} {self.slv_kw}')  

    #Context Manager Interface
    def __enter__(self):
        #Set the new state
        
        self.activate_temp_state()
        
        if self.pre_exec:
            self.pre_execute()

        #Check for existing session        
        if hasattr(ProblemExec,'_session'):
            self.msg(f'[{self.level_number}-{self.level_name}][{self.level_number}-{self.level_name}] entering existing execution context')
            if not isinstance(self,self._class_cache):
                self.warning(f'change of execution class!')
            #global level number
            self._class_cache.level_number += 1     
            self.entered = True   
            return self._class_cache._session
        
        #return New
        self._class_cache._session = self
        self._class_cache.level_number = 0

        if self.log_level < 10:
            refs = {k:v for k,v in self.sys_refs.get('attrs',{}).items() if v}
            self.debug(f'[{self.level_number}-{self.level_name}] creating execution context for {self.system}| {self.slv_kw}| {refs}')
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #define exit action, to handle the error here return True. Otherwise error propigates up to top level      

        #Last opprotunity to update the system  at tsate
        if self.post_exec:
            #a component cutsom callback + signals
            self.post_execute()

        if self.post_callback:
            #a context custom callback
            self.post_callback()

        # TODO: save state to dataframe

        #Exit Scenerio (boolean return important for context manager exit handling in heirarchy)
        if isinstance(exc_value,ProblemExit):     
            
            #first things first
            if exc_value.revert:
                self.revert_to_start()
                if self.pre_exec:
                    self.pre_execute()         

            #Decide our exit conditon (if we should exit)
            if isinstance(exc_value,ProblemExitAtLevel):
                #should we stop?
                if exc_value.level == self.level_name:
                    if self.log_level <= 11:
                        self.debug(f'[{self.level_number}-{self.level_name}] exit at level {exc_value}')
                    ext = True                     
                else:
                    if self.log_level <= 5:
                        self.msg(f'[{self.level_number}-{self.level_name}] exit not at level {exc_value}')
                    ext = False

                #Check if we missed a level name and its the top level, if so then we raise a real error!
                #always exit with level_name='top' at outer context
                if not ext and self._class_cache._session is self and exc_value.level=='top':
                    if self.log_level <= 11:
                        self.debug(f'[{self.level_number}-{self.level_name}] exit at top')                
                    ext = True

                elif self._class_cache._session is self and not ext:
                    #never ever leave the top level without deleting the session
                    self._class_cache.level_number = 0
                    del self._class_cache._session 
                    self.exited = True
                    raise KeyError(f'cant exit to level! {exc_value.level} not found!!')
            
            #TODO: expand this per options
            else:
                if self.log_level <= 18:
                    self.info(f'[{self.level_number}-{self.level_name}] problem exit revert={exc_value.revert}')

                ext = True

        elif exc_type is not None:
            ext = self.error_action(exc_value)
        else:
            ext = self.exit_action()

        self.clean_context()

        self.exited = True
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
        if self.revert_last and (self._class_cache._session is self or self.level_name == 'top'):
            if self.log_level <= 8:
                self.debug(f'[{self.level_number}-{self.level_name}] reverting to start')
            self.revert_to_start()
            
            #run execute
            if self.pre_exec:
                self.pre_execute()

        elif self.revert_every:
            if self.log_level <= 8:
                self.debug(f'[{self.level_number}-{self.level_name}] to {self.X_start}')
            self.revert_to_start()
            
            #run execute 
            if self.pre_exec:
                self.pre_execute()

        #TODO: add exit on success option
        return True #continue as normal

    def error_action(self,error):
        """handles the error action wrt to the problem"""
        if self.log_level <= 11:
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
            if self.log_level <= 10:
                self.debug(f'[{self.level_number}-{self.level_name}] closing execution session')
            self._class_cache.level_number = 0
            del self._class_cache._session
        elif hasattr(self._class_cache,'_session'):
            #global level number
            self._class_cache.level_number -= 1  
             

    #State Interfaces
    @property
    def record_state(self)->dict:
        """records the state of the system"""
        #refs = self.all_variable_refs
        refs = self.all_variables
        return Ref.refset_get(refs)     
    
    def get_ref_values(self,refs=None):
        """returns the values of the refs"""
        if refs is None:
            refs = self.all_system_references
        return Ref.refset_get(refs)
    
    def set_ref_values(self,values,refs=None):
        """returns the values of the refs"""
        #TODO: add checks for the refs
        if refs is None:
            refs = self.all_variables        
        return Ref.refset_input(refs,values)    

    def set_checkpoint(self):
        """sets the checkpoint"""
        self.X_start = self.record_state
        if log.log_level <= 7:
            self.debug(f'[{self.level_number}-{self.level_name}] set checkpoint: {list(self.X_start.values())}')

    def revert_to_start(self):
        if log.log_level < 5:
            xs = list(self.X_start.values())
            rs = list(self.record_state.values())
            self.debug(f'[{self.level_number}-{self.level_name}] reverting to start: {xs} -> {rs}')
        Ref.refset_input(self.all_variables,self.X_start)

    def activate_temp_state(self,new_state=None):
        if new_state:
            Ref.refset_input(self.all_variables,new_state)
        elif self._temp_state:
            self.debug(f'[{self.level_number}-{self.level_name}] activating temp state: {self._temp_state}')
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
            self.debug(f'context updating {ukey}')
            uref.value(*args,**kwargs)

    def post_update_system(self,*args,**kwargs):
        """updates the system"""
        for ukey,uref in self._post_update_refs.items():
            self.debug(f'context post updating {ukey}')
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
        Solve the minimization problem using the given vars and constraints. And sets the system state to the solution depending on input of the following:

        Solve the root problem using the given vars.
        :param Xref: The reference input values.
        :param Yref: The reference objective values to minimize.
        :param output: The output dictionary to store the results. (default: None)
        :param fail: Flag indicating whether to raise an exception if the solver doesn't converge. (default: True)
        :param kw: Additional keyword arguments.
        :return: The output dictionary containing the results.
        """

        thresh = kw.pop("thresh", self.success_thresh)

        dflt = {
                "Xstart": Ref.refset_get(Xref,sys=self.system,prob=self),
                "Ystart": Ref.refset_get(Yref,sys=self.system,prob=self),
                "Xans": None,
                "success": None,
                "Xans":None,
                "Yobj":None,
                "Ycon":None,
            }

        if output:
            dflt.update(output)
            output = dflt
        else:
            output = dflt
            

        #override constraints input
        kw.update(self.constraints)

        if len(kw['bounds']) != len(Xref):
            raise ValueError(f"bounds {len(self.constraints['bounds'])} != Xref {len(Xref)}")

        if self.log_level < 10:
            self.debug(f"minimize {Xref} {Yref} {kw}")
        

        self._ans = refmin_solve(self.system,self,Xref, Yref, **kw)
        output["ans"] = self._ans

        self.handle_solution(self._ans,Xref,Yref,output)

        return output

    def handle_solution(self,answer,Xref,Yref,output):
        #TODO: move exit condition handiling somewhere else, reduce cross over from process_ans
        thresh = self.success_thresh
        vars = list(Xref)

        #Output Results
        Xa = {p: answer.x[i] for i, p in enumerate(vars)}
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
            self.system._converged = True
            output["success"] = True

        elif answer.success:
            # out of threshold condition
            self.warning(
                f"solver didnt fully solve equations! {answer.x} -> residual: {answer.fun}"
            )
            self.system._converged = False
            output["success"] = False  # only false with threshold

        else:
            self.system._converged = False
            if self.raise_on_opt_failure:
                raise Exception(f"solver didnt converge: {answer}")
            else:
                self.warning(f"solver didnt converge: {answer}")
            output["success"] = False
    
        return output  
    
    #Dynamics Interface
    def filter_vars(self,refs:list):
        '''selects only settable refs'''
        return {v.key:v for k,v in refs.items() if v.allow_set}
    
    def filter_prop(self,refs:list):
        '''selects only settable refs'''
        return {k:v for k,v in refs.items() if not v.allow_set}        

    #X solver variable refs
    @property
    def problem_opt_vars(self)->dict:
        """solver variables"""
        return self.ref_attrs.get('solver.var',{}).copy()
    
    @property
    def all_problem_vars(self)->dict:
        """solver variables + dynamics states when dynamic_solve is True"""
        varx = self.ref_attrs.get('solver.var',{}).copy()
        #Add the dynamic states
        if self.dynamic_solve:
            varx.update(self.filter_vars(self.dynamic_state))
            varx.update(self.filter_vars(self.integrator_vars))
        return varx

    @property
    def dynamic_solve(self)->bool:
        """indicates if the system is dynamic"""
        dxdt = self._dxdt
        in_type = isinstance(dxdt,(dict,float,int))
        bool_type = (isinstance(dxdt,bool) and dxdt == True)
        if in_type or bool_type:
            return True
        return False

    #Logging to class logger
    @property
    def identity(self):
        return f'PROB|{self.level_name}|{str(self._session_id)[0:5]}'

    @property
    def log_level(self):
        return log.log_level

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
        return self.sys_refs.get('attrs',{}).copy()

    @property
    def attr_inst(self):
        return self.sys_refs.get('type',{}).copy()
    
    @property
    def dynamic_comps(self):
        return self.sys_refs.get('dynamic_comps',{}).copy()
    
    #Instances
    @property
    def integrators(self):
        return self.attr_inst.get('time',{}).copy()
    
    @property
    def signal_inst(self):
        return self.attr_inst.get('signal',{}).copy()

    @property
    def solver_inst(self):
        return self.attr_inst.get('solver',{}).copy()

    @property
    def kwargs(self):
        return self.slv_kw


    @property
    def dynamic_state(self):
        return self.ref_attrs.get('dynamics.state',{}).copy()
    
    @property
    def dynamic_rate(self):
        return self.ref_attrs.get('dynamics.rate',{}).copy()    
    
    @property
    def problem_input(self):
        return self.ref_attrs.get('dynamics.input',{}).copy()      

    @property
    def integrator_vars(self):
        return self.ref_attrs.get('time.var',{}).copy()
    
    @property
    def integrator_rates(self):
        return self.ref_attrs.get('time.rate',{}).copy()
    
    #Y solver variables
    @property
    def problem_objs(self):
        return self.ref_attrs.get('solver.obj',{}).copy()   

    @property
    def problem_eq(self):
        return self.ref_attrs.get('solver.eq',{}).copy()
    
    @property
    def problem_ineq(self):
        return self.ref_attrs.get('solver.ineq',{}).copy()
    
    @property
    def signals_source(self):
        return self.ref_attrs.get('signal.source',{}).copy()
    
    @property
    def signals_target(self):
        return self.ref_attrs.get('signal.target',{}).copy()

    @property
    def signals(self):
        return self.ref_attrs.get('signal.signal',{}).copy()

    #formatted output
    @property
    def is_active(self):
        """checks if the context has been entered and not exited"""
        return self.entered and not self.exited
    
    @property
    def solveable(self):
        """checks the system's references to determine if its solveabl"""
        if self.problem_opt_vars:
            #TODO: expand this
            return True
        return False

    @property
    def integrator_rate_refs(self):
        """combine the dynamic state and the integrator rates to get the transient state of the system, but convert their keys to the target var names """
        dc  = self.dynamic_state.copy()
        for int_name,intinst in self.integrators.items():
            if intinst.var in dc:
                raise KeyError(f'conflict with integrator name {intinst.var} and dynamic state')
            dc.update({intinst.var:intinst.rate_ref})
        return dc
    
    @property
    def integrator_var_refs(self):
        """combine the dynamic state and the integrator rates to get the transient state of the system, but convert their keys to the target var names """
        dc  = self.dynamic_state.copy()
        for int_name,intinst in self.integrators.items():
            if intinst.var_ref in dc:
                raise KeyError(f'conflict with integrator name {intinst.var_ref} and dynamic state')
            dc.update({intinst.var:intinst.var_ref})
        return dc    
    
    #TODO: expose optoin for saving all or part of the system information, for now lets default to all (saftey first, then performance :)
    @property
    def all_variable_refs(self)->dict:
        ing = self.integrator_vars
        stt = self.dynamic_state
        vars = self.problem_opt_vars
        return {**ing,**stt,**vars}
    
    @property
    def all_variables(self)->dict:
        #TODO: ensure system refes are fresh per system runtime events
        return self.system.system_references(False)['attributes']
    
    @property
    def all_system_references(self)->dict:
        refs = self.system.system_references(False)
        out = {}
        out.update(refs['attributes'])
        out.update(refs['properties'])
        return out
    

    #Solver Parsing Methods
    def sys_solver_variables(self,as_set=False,as_flat=False,**kw):
        """gathers variables from solver vars, and attempts to locate any input_vars to add as well. use exclude_vars to eliminate a variable from  the solver
        """
        out = dict(dynamics=self.dynamic_state,integration=self.integrator_vars,variables=self.problem_opt_vars)

        if as_set==True:
            vars = set.union(*(set(v) for k,v in out.items()))
            return vars
        
        if as_flat==True:
            flt = {}
            for k,v in out.items():
                flt.update(v)
            return flt
        
        return out

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
        trv_inst = {v.var:v for v in sys_refs.get('type',{}).get('time',{}).values()}
        sys_refs = sys_refs.get('attrs',{})

        if add_con is None:
            add_con = {}
        

        #The official definition of X var order
        Xvars = list(Xrefs)

        # constraints lookup
        bnd_list = [[None, None]] * len(Xvars)
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
        for slvr, ref in self.problem_opt_vars.items():
            assert not all((slvr in slv_inst,slvr in trv_inst)), f'solver and integrator share parameter {slvr} '
            if slvr in slv_inst:
                slv = slv_inst[slvr]
                slv_var = True #mark a static varible
            elif slvr in trv_inst:
                slv = trv_inst[slvr]
                slv_var = False #a dynamic variable
            else:
                self.warning(f'no solver instance for {slvr} ')
                continue
            
            slv_constraints = slv.constraints
            if log.log_level < 7:
                self.debug(f'[{self.level_number}-{self.level_name}] constraints {slvr} {slv_constraints}')
            for ctype in slv_constraints:
                cval = ctype['value']
                kind = ctype['type']       
                var = ctype['var']
                if log.log_level < 3:
                    self.msg(f'[{self.level_number}-{self.level_name}]const: {slvr} {ctype}')
                if cval is not None and slvr in Xvars:
                    
                    #Check for combos & activation
                    combos = None
                    if 'combos' in ctype:
                        combos = ctype['combos']
                        combo_var = ctype['combo_var']
                        active = ctype.get('active',True)
                        in_activate = any([arg_var_compare(combo_var,v) for v in activated]) if activated else False
                        in_deactivate = any([arg_var_compare(combo_var,v) for v in deactivated]) if deactivated else False

                        #Check active or activated
                        if not active and not activated:
                            if log.log_level < 3:
                                self.msg(f'[{self.level_number}-{self.level_name}]skip con: inactive {var} {slvr} {ctype}')
                            continue

                        elif not active and not in_activate:
                            if log.log_level < 3:
                                self.msg(f'[{self.level_number}-{self.level_name}]skip con: inactive {var} {slvr} {ctype}')
                            continue

                        elif active and in_deactivate:
                            if log.log_level < 3:
                                self.msg(f'[{self.level_number}-{self.level_name}]skip con: deactivated {var} {slvr} ')
                            continue

                    if combos and combo_filter:
                        filt = filt_combo_vars(combo_var,slv, extra_kw,combos)
                        if not filt:
                            if log.log_level < 5:
                                self.debug(f'[{self.level_number}-{self.level_name}] filtering constraint={filt} {var} {slv} {ctype} | {combos} {ext_str_list(extra_kw,"combos",None)}')                        
                            continue
                    
                    if log.log_level < 10:
                        self.debug(f'[{self.level_number}-{self.level_name}] adding var constraint {var,slvr,ctype,combos}') 

                    
                    x_inx = Xvars.index(slvr)        
                    
                    #lookup rates
                    if isinstance(self._dxdt,dict) and not slv_var:
                        rate_val = self._dxdt.get(slvr,0)
                    elif not slv_var:
                        rate_val = 0
                    else:
                        rate_val = None

                    #add the dynamic parameters when configured
                    if not slv_var and rate_val is not None:
                        #if kind in ('min','max') and slvr in Xvars:
                        varref = Xrefs[slvr]
                        #varref = slv.rate_ref
                        #Ref Case
                        ccst = ref_to_val_constraint(system,system.last_context,Xrefs,varref,kind,rate_val,*args,**kw)
                        #con_list.append(ccst)
                        con_info.append(f'dxdt_{varref.comp.classname}.{slvr}_{kind}_{cval}')
                        con_list.append(ccst) 

                    elif slv_var:                       
                        #establish simple bounds w/ solver  
                        if ( 
                            kind in ("min", "max")
                            and slvr in Xvars
                            and isinstance(cval, (int, float))
                        ):
                            minv, maxv = bnd_list[x_inx]
                            bnd_list[x_inx] = [
                                cval if kind == "min" else minv,
                                cval if kind == "max" else maxv,
                                ]
                                
                        #add the bias of cval to the objective function
                        elif  kind in ('min','max') and slvr in Xvars:
                            varref = Xrefs[slvr]
                            #Ref Case
                            ccst = ref_to_val_constraint(system,system.last_context,Xrefs,varref,kind,cval,*args,**kw)
                            con_info.append(f'val_{ref.comp.classname}_{kind}_{slvr}')
                            con_list.append(ccst)

                        else:
                            self.warning(f"bad constraint: {cval} {kind} {slv_var}|{slvr}")

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

    
#subclass before altering please!
ProblemExec._class_cache = ProblemExec


