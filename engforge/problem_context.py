"""The ProblemExec provides a uniform set of options for managing the state of the system and its solvables, establishing the selection of combos or de/active attributes to Solvables. Once once created any further entracnces to ProblemExec will return the same instance until finally the last exit is called. 

The ProblemExec class allows entrance to a its context to the same instance until finally the last exit is called. The first entrance to the context will create the instance, each subsequent entrance will return the same instance. The ProblemExec arguments are set the first time and remove keyword arguments from the input dictionary (passed as a dict ie stateful) to subsequent methods. 
This isn't technically a singleton pattern, but it does provide a similar interface. Instead mutliple problem instances will be clones of the first instance, with the optional difference of input/output/event criteria. The first instance will be returned by each context entry, so for that reason it may always appear to have same instance, however each instance is unique in a recusive setting so it may record its own state and be reverted to its own state as per the options defined.

#TODO: allow update of kwargs on re-entrance

## Example:
.. code-block:: python

    #Application code (arguments passed in kw)
    with ProblemExec(sys,combos='default',slv_vars'*',**kw) as pe:
        pe._sys_refs #get the references and compiled problem
        for i in range(10):
            pe.solve_min(pe.Xref,pe.Yref,**other_args)
            pe.set_checkpoint() #save the state of the system 
            self.save_data()
            

    #Solver Module (can use without knowledge of the runtime system)
    with ProblemExec(sys,{},Xnew=Xnext,ctx_fail_new=True) as pe:
        #do revertable math on the state of the system without concern for the state of the system
...

# Combos Selection
By default no arguments run will select all active items with combo="default". The `combos` argument can be used to select a specific set of combos, a outer select. From this set, the `ign_combos` and `only_combos` arguments can be used to ignore or select specific combos based on exclusion or inclusion respectively.

# Parameter Name Selection
The `slv_vars` argument can be used to select a specific set of solvables. From this set, the `ign_vars` and `only_vars` arguments can be used to ignore or select specific solvables based on exclusion or inclusion respectively. The `add_vars` argument can be used to add a specific set of solvables to the solver.

# Active Mode Handiling
The `only_active` argument can be used to select only active items. The `activate` and `deactivate` arguments can be used to activate or deactivate specific solvables.

`add_obj` can be used to add an objective to the solver. 

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
from engforge.dataframe import DataframeMixin,pandas
from engforge.solver_utils import *
import weakref

from scipy.integrate import solve_ivp
from collections import OrderedDict
import numpy as np
import pandas as pd
import expiringdict
import attr, attrs
import datetime

class ProbLog(LoggingMixin): pass
log = ProbLog()

import uuid

#TODO: implement add_vars feature, ie it creates a solver variable, or activates one if it doesn't exist from in system.heirarchy.format
#TODO: define the dataframe / data storage feature

min_kw_dflt = {"tol": 1e-10, "method": "SLSQP"}


#The KW Defaults for Solver via kw_dict
# IMPORTANT:!!! these group parameter names by behavior, they are as important as the following class, add/remove variables with caution
#these choices affect how solver-items are selected and added to the solver
slv_dflt_options = dict(combos='default',ign_combos=None,only_combos=None,add_obj=True,slv_vars='*',add_vars=None,ign_vars=None,only_vars=None,only_active=True,activate=None,deactivate=None,dxdt=None,weights=None,both_match=True,obj=None)
#KW Defaults for the context
dflt_parse_kw = dict(fail_revert=True,revert_last=True,revert_every=True,exit_on_failure=True, pre_exec=True,post_exec=True,opt_fail = True,level_name='top',post_callback=None,success_thresh=10,copy_system=False,run_solver=False,min_kw=None,save_mode='all',x_start = None,save_data_on_exit=False)
#can be found on session._<parm> or session.<parm>
root_defined = dict( last_time = 0,time = 0,dt = 0,update_refs=None,post_update_refs=None,sys_refs=None,slv_kw=None,minimizer_kw=None,data = None,weights=None,dxdt = None,run_start = None,run_end = None,run_time = None,all_refs=None,num_refs=None,converged=None,comp_changed=False)
save_modes = ['vars','nums','all','prob']
transfer_kw = ['system']

root_possible = list(root_defined.keys()) + list('_'+k for k in root_defined.keys())

#TODO: output options extend_dataframe=True,return_dataframe=True,condensed_dataframe=True,return_system=True,return_problem=True,return_df=True,return_data=True
#TODO: connect save_data() output to _data table.
#TODO: move dataframe mixin here, system should return a dataframe, and the problem should be able to save data to the dataframe, call this class Problem(). With default behavior it could seem like a normal dataframe is returned on problem.return(*state,exit,revert...)

#Special exception classes handled in exit
class IllegalArgument(Exception):
    """an exception to exit the problem context as specified"""
    pass

class ProblemExit(Exception):
    """an exception to exit the problem context, without error"""
    revert:bool
    prob:"ProblemExec"
    def __init__(self,prob:"ProblemExec",revert:bool=None):
        self.revert = revert
        self.prob = prob

    def __str__(self) -> str:
        return f'ProblemExit[{self.prob}|rvt={self.revert}]'        

class ProblemExitAtLevel(ProblemExit):
    """an exception to exit the problem context, without error"""
    level: str
    def __init__(self,prob:"ProblemExec",level:str,revert=None):
        assert level is not None, 'level must be defined'
        assert isinstance(level,str), 'level must be a string'
        self.prob = prob
        self.level = level.lower()
        self.revert = revert

    def __str__(self) -> str:
        return f'ProblemExit[{self.prob}|lvl={self.level}|rvt={self.revert}]'


#TODO: determine when components are updated, and refresh the system references accordingly.
#TODO: Map attributes/properties by component key and then autofix refs! (this is a big one), no refresh required. Min work
class ProblemExec:
    """
    Represents the execution context for a problem in the system. The ProblemExec class provides a uniform set of options for managing the state of the system and its solvables, establishing the selection of combos or de/active attributes to Solvables. Once once created any further entracnces to ProblemExec will return the same instance until finally the last exit is called.

    ## params:
    -  _problem_id: uuid for subproblems, or True for top level, None means uninitialized

    """
    class_cache = None #ProblemExec is assigned below
    
    #this class, wide, dont redefine it
    problems_dict = weakref.WeakValueDictionary()

    system: "System"
    session: "ProblemExec"
    session_id = None

    #problem state / per level
    problem_id = None 
    entered: bool = False
    exited: bool = False  

    #solution control (point to singleton/subproblem via magic getattr)
    _last_time: float = 0
    _time: float = 0
    _dt: float = 0    
    _update_refs: dict 
    _post_update_refs: dict
    _sys_refs: dict
    _slv_kw: dict
    _minimizer_kw: dict = None
    _data: list = None
    _weights: dict
    x_start:dict = None
    _dxdt = None #numeric/None/dict/(integrate=True)
    _run_start = None
    _run_end = None
    _run_time = None
    _converged = None
    

    #Interior Context Options
    save_data_on_exit: bool = False
    save_mode: str = 'all'
    level_name: str = None #target this context with the level name
    level_number: int = 0 #TODO: keep track of level on the global context  
    pre_exec: bool = True
    post_exec: bool = True
    fail_revert: bool = True
    revert_last: bool = True
    revert_every: bool = True
    exit_on_failure: bool = True
    opt_fail: bool = True
    raise_on_unknown: bool = True
    copy_system: bool = False
    success_thresh = 1E6 #if system has `success_thresh` it will be assigned to the context
    post_callback: callable = None#callback that will be called on the system each time it is reverted, it should take args(system,current_problem_exec)
    run_solver: bool = False #for transient #i would love this to be=true, but there's just too much possible variation in application to make it so without some kind of control / continuity strategy. Dynamics are natural responses anyways, so solver use should be an advanced case for now (MPC/Filtering/ect later)

    


    def __getattr__(self, name):
        '''This is a special method that is called when an attribute is not found in the usual places, like when interior contexts (anything not the root (session_id=True)) are created that dont have the top level's attributes. some attributes will look to the parent session'''

        #interior context lookup (when in active context, ie session exists)
        if hasattr(self.class_cache,'session') and name in root_possible:
            #revert to the parent session
            if self.session_id != True and name.startswith('_'):
                return getattr(self.class_cache.session,name)

            elif name in root_defined:
                return getattr(self.class_cache.session,'_'+name)
            
        if name in root_defined: #public interface
            return self.__getattribute__('_'+name)

        # Default behaviour
        return self.__getattribute__(name)
    
#     def __setattr__(self, name: str, value) -> None:
#         """only allow setting data to parent for now (this is in reset which can trigger anywhere)"""
#         if hasattr(self.class_cache,'session') and name=='_data':
#             if self.session_id != True:
#                 self.class_cache.session._data = value
#                 return 
#             elif self.session_id == True:
#                 self._data = value
#                 return 
# 
#         super().__setattr__(name, value)    

        
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

        #### Solver Selection Options
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
        #### Context Options
        :param refresh: Whether to refresh the system references. Default is False, if there is an active session context. New contexts will always refresh the system references.
        """


        if kw_dict is None:
            #kw_dict is stateful so you can mix system & context args together, and ensure context args are removed. in the case this is unused, we'll create an empty dict to avoid errors
            kw_dict = {}
        
        #storage optoins
        if opts.pop('persist',False) or kw_dict.pop('persist',False) :
            self.persist_contexts()

        # temp solver storage
        self.solver_hist = expiringdict.ExpiringDict(100, 60)  

        if self.log_level < 15:
            if hasattr(self.class_cache,'session'):
                self.debug(f'subctx{self.level_number}|  keywords: {kw_dict} and misc: {opts}')
            else:
                self.debug(f'context| keywords: {kw_dict} and misc: {opts}')

        #special cases for parsing
        #parse the options to change behavior of the context
        level_name = None
        if opts and 'level_name' in opts:
            level_name = opts.pop('level_name').lower()
        if kw_dict and 'level_name' in kw_dict:
            level_name = kw_dict.pop('level_name').lower()

        #solver min-args wrt defaults
        min_kw = None
        if opts and 'min_kw' in opts: 
            min_kw = kw_dict.pop('min_kw')
        if kw_dict and 'min_kw' in kw_dict:
            min_kw = kw_dict.pop('min_kw')

        
        mkw =min_kw_dflt.copy()
        if min_kw is None:
            min_kw = mkw
        else:
            mkw.update(min_kw)

        if "refresh" in opts or "refresh" in kw_dict:
            refresh = kw_dict.pop("refresh",opts.get("refresh",False))
        else:
            refresh = False

        self._minimizer_kw = mkw

        #Merge kwdict(stateful) and opts (current level)
        #solver vars should be static for a problem and subcontexts, however the default vars can change. Subproblems allow for the solver vars to be changed on its creation.
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
            opt_in.update(kw_in)
            kw_out = {k:v for k,v in kw_dict.items() if k not in opt_in}
            #these go to system establishment
            opt_out.update(kw_out)
            #remove problem options from dict (otherwise passed along to system!)
            for k in kw_in:
                kw_dict.pop(k)


        #Define the handiling of rate integrals
        if 'dxdt' in opts:
            dxdt = opts.pop('dxdt')
        if 'dxdt' in kw_dict:
            dxdt = kw_dict.pop('dxdt')
        else:
            dxdt = None  #by default dont consider dynamics

        if dxdt is not None and dxdt is not False:
            if dxdt == 0:
                pass
            elif dxdt is True:
                pass
            elif isinstance(dxdt, dict):  # dxdt is a dictionary
                #provide a set of values or function to have the solver solve for
                pass
            else:
                raise IllegalArgument(f'bad dxdt value {dxdt}')
                
        self._dxdt = dxdt              
            
        if hasattr(self.class_cache,'session'):                        
            #mirror the state of session (exactly)
            copy_vals = {k:v for k,v in self.class_cache.session.__dict__.items() if k in dflt_parse_kw or k in transfer_kw}
            self.__dict__.update(copy_vals)
            self._problem_id = int(uuid.uuid4())
            self.problems_dict[self._problem_id] = self #carry that weight
            self.session_id = int(uuid.uuid4())

            self.class_cache.session._prob_levels[self.level_name] = self
            #error if the system is different (it shouldn't be!)
            if self.system is not system:
                raise IllegalArgument(f'somethings wrong! change of comp! {self.system} -> {system}')
            
            if refresh:
                self.class_cache.session.refresh_references()

            #modify things from the input
            if level_name is None:
                #your new id
                self.level_name = 'ctx_'+str(int(self._problem_id))[0:15]
            else:
                self.level_name = level_name

            if opt_in: self.__dict__.update(opt_in) #level options ASAP
            self.temp_state = Xnew #input state exception to this
            if log.log_level < 5:
                self.msg(f'setting execution context with {opt_in}| {opt_out}')
            
            #each state request to be reverted, then we need to store the state of each execution context overriding the outer context x_start
            self.set_checkpoint()

        elif ctx_fail_new:
            raise IllegalArgument(f'no execution context available')
        
        else:
            #add the prob options to the context
            self.__dict__.update(opt_in)
            self._problem_id = True #this is the top level
            self.problems_dict[self._problem_id] = self #carry that weight
            self._prob_levels = {}

            self.reset_data()

            #supply the level name default as top if not set
            if level_name is None:
                self.level_name = 'top'
            else:
                self.level_name = level_name

            self.temp_state = Xnew
            self.establish_system(system,kw_dict=kw_dict,**opt_out)
            #Finally we record where we started!
            self.set_checkpoint()        

        if log.log_level < 10:
            self.info(f'new execution context for {system}| {opts} | {self._slv_kw}')            
        elif log.log_level <= 3:
            self.msg(f'new execution context for {system}| {self._slv_kw}')


    def reset_data(self):
        '''reset the data storage'''
        #the data storage!!
        #TODO: add buffer type, or disk cache
        self._data = {} #index:row_dict 
        self._index = 0# works for time or index


    def establish_system(self,system,kw_dict,**kwargs):
        """caches the system references, and parses the system arguments"""
        from engforge.solver import SolverMixin
        from engforge.system import System 
    
        if self.copy_system:
            system = system.copy_config_at_state()
        
        if system.is_dynamic:
            system.setup_global_dynamics()

        #place me here after system has been modified
        self.system = system
        #cache as much as possible before running the problem (stitch in time saves 9 or whatever)


        #pass args without creating singleton (yet)
        self.session_id = int(uuid.uuid4())
        self._run_start = datetime.datetime.now()
        self.name = system.name + '-' + str(self.session_id)[:8]

        if log.log_level < 5:
            self.info(f'establish {system}| {kw_dict} {kwargs}')

        
        assert isinstance(self.system,SolverMixin), 'only solveable interfaces are supported for execution context'
        self.system._last_context = self #set the last context to this one

        if hasattr(self.system,'success_thresh') and isinstance(self.system.success_thresh,(int,float)):
            self.success_thresh = self.system.success_thresh
        #Extract solver vars and set them on this object, they will be distributed to any new further execution context's via monkey patch above
        in_kw = self.get_extra_kws(kwargs,slv_dflt_options,use_defaults=False)
        self._slv_kw = self.get_extra_kws(kw_dict,slv_dflt_options,rmv=True)
        self._slv_kw.update(in_kw) #update with input!

        self.refresh_references()

        #Get solver weights
        self._weights = self._slv_kw.get('weights',None)

        #Grab inputs and set to system
        for k,v in dflt_parse_kw.items():
            if k in self._slv_kw:
                setattr(self,k,self._slv_kw[k])

        if log.log_level < 5:
            self.msg(f'established sys context: {self} {self._slv_kw}')  

    @property
    def sesh(self):
        """caches the property for the session"""
        if hasattr(self,'inst_sesh'):
            return self.inst_sesh
        sesh = self.get_sesh()
        return sesh

    def get_sesh(self,sesh=None):
        """get the session"""
        out = sesh
        if not sesh:
            if hasattr(self.class_cache,'session'):
                out = self.class_cache.session
            elif self._problem_id == True:
                out = self
        if out: 
            self.inst_sesh = out
        return out

    #Update Methods
    def refresh_references(self,sesh=None):
        """refresh the system references"""
        sesh = self.sesh

        if self.log_level < 5:
            self.warning(f'refreshing system references')
        check_dynamics = sesh.check_dynamics
        sesh._num_refs = sesh.system.system_references(numeric_only=True)
        sesh._sys_refs = sesh.system.solver_vars(check_dynamics=check_dynamics,addable=sesh._num_refs,**sesh._slv_kw)

        sesh.update_methods(sesh=sesh)
        sesh.min_refresh(sesh=sesh)
                

    def update_methods(self,sesh=None):
        #Get the update method refs
        sesh = self.sesh

        sesh._update_refs = sesh.system.collect_update_refs()
        #TODO: find subsystems that are not-subsolvers and execute them
        sesh._post_update_refs = sesh.system.collect_post_update_refs()

        #initial establishment costs / ect
        if sesh.pre_exec:
            sesh.pre_execute()

    def min_refresh(self,sesh=None):
        sesh = self.sesh

        if self.log_level < 5:
            self.info(f'min refresh')

        #final ref's after update
        sesh._all_refs = sesh.system.system_references(recache=True,check_config=False) #after updates

        #Problem Variable Definitions
        sesh.Xref = sesh.all_problem_vars
        sesh.Yref = sesh.sys_solver_objectives()
        
        cons = {} #TODO: parse additional constraints
        sesh.constraints = sesh.sys_solver_constraints(cons)          

    @property
    def check_dynamics(self):
        return self._dxdt is not None and self._dxdt is not False

    #Context Manager Interface
    def __enter__(self):
        #Set the new state
        
        if self.entered:
            #TODO: enable env-var STRICT MODE to fail on things like this
            self.warning(f'context already entered!')
        
        #Important managed updates / refs from Xnew input
        self.activate_temp_state()
        self.entered = True

        #TODO: create a component-slot ref-update graph, and update the system references accordingly.
        #TODO: map the signals to the system references, and update the system references accordingly.
        #TODO: 
        #transients wont update components/ methods dynamically (or shouldn't) so we can just update the system references once and be done with it for other cases, but that is not necessary unless a component changes or a component has in general a unique reference update system (economics / component-iterators)
        sesh = self.sesh
        if not self._dxdt is True:
            sesh.update_methods(sesh=sesh)
            sesh.min_refresh(sesh=sesh)

        #Check for existing session        
        if sesh not in [None,self]:
            self.msg(f'entering existing execution context')
            if not isinstance(self,self.class_cache):
                self.warning(f'change of execution class!')
            #global level number

            self.class_cache.level_number += 1     
            self.class_cache.session._prob_levels[self.level_name] = self
            
            return self.class_cache.session
        
        #return New
        self.class_cache.session = self
        self.class_cache.level_number = 0

        if self.log_level < 10:
            refs = {k:v for k,v in self.sesh._sys_refs.get('attrs',{}).items() if v}
            self.debug(f'creating execution context for {self.system}| {self._slv_kw}| {refs}')
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #define exit action, to handle the error here return True. Otherwise error propigates up to top level
        self.exited = True
        if self.log_level < 10:
            self.debug(f'exit action {exc_type} {exc_value}')

        #Last opprotunity to update the system  at tsate
        if self.post_exec:
            #a component cutsom callback + signals
            self.post_execute()

        if self.post_callback:
            #a context custom callback
            self.post_callback()

        #save state to dataframe
        if self.save_data_on_exit:
            self.save_data()

        #sesh = self.sesh #this should be here
        #if self.level_name in sesh._prob_levels:
        #    sesh._prob_levels.pop(self.level_name)

        #Exit Scenerio (boolean return important for context manager exit handling in heirarchy)
        if isinstance(exc_value,ProblemExit):
            if self.log_level < 7:
                self.debug(f'exit action {exc_type}| {exc_value.__dict__}')
            
            #first things first
            if exc_value.revert:
                self.revert_to_start()
                if self.pre_exec:
                    self.pre_execute()         
            lvl_match = False
            #Decide our exit conditon (if we should exit)
            if isinstance(exc_value,ProblemExitAtLevel):
                #should we stop?
                lvl_match = exc_value.level == self.level_name
                if lvl_match:
                    if self.log_level <= 11:
                        self.debug(f'exit at level {exc_value}')
                    ext = True                     
                else:
                    if self.log_level <= 5:
                        self.msg(f'exit not at level {exc_value}')
                    ext = False

                #Check if we missed a level name and its the top level, if so then we raise a real error!
                #always exit with level_name='top' at outer context
                if not ext and self.class_cache.session is self and exc_value.level=='top':
                    if self.log_level <= 11:
                        self.debug(f'exit at top')

                    ext = True #top override

                elif self.class_cache.session is self and not ext:
                    #never ever leave the top level without deleting the session
                    self.class_cache.level_number = 0
                    del self.class_cache.session 
                    raise KeyError(f'cant exit to level! {exc_value.level} not found!!')
            
            else:
                if self.log_level <= 18:
                    self.info(f'problem exit revert={exc_value.revert}')

                ext = True #basic exit is one level up
            
            self.clean_context()

            return ext
        
        #default exit scenerios
        elif exc_type is not None:
            ext = self.error_action(exc_value)
        else:
            ext = self.exit_action()

        self.clean_context()

        return ext
        
    def debug_levels(self):
        """debug the levels of the context"""
        if hasattr(self.class_cache,'session'):
            for k,v in self.class_cache.session._prob_levels.items():
                self.info(f'level: {k} | {v} | {v.x_start}')

        else:
            raise IllegalArgument(f'no session available')
    
    #Multi Context Exiting:
    def persist_contexts(self):
        """convert all contexts to a new storage format"""
        self.info(f'persisting contexts!')
        current_problems = self.problems_dict 
        ProblemExec.problems_dict = {}
        for k,v in current_problems.items():
            self.problems_dict[k] = v #you will go on!

    def discard_contexts(self):
        """discard all contexts"""
        current_problems = self.problems_dict 
        ProblemExec.problems_dict = weakref.WeakValueDictionary()
        for k,v in current_problems.items():
            ProblemExec.problems_dict[k] = v #you will go on!             

    def reset_contexts(self,fail_if_discardmode=True):
        """reset all contexts to a new storage format"""
        if isinstance(self.problems_dict,dict):
            ProblemExec.problems_dict = {}
        elif fail_if_discardmode:
            raise IllegalArgument(f'cant reset contexts! {self.problems_dict} while not in persistance mode')       

    def exit_with_state(self):
        raise ProblemExit(self,revert=False)
    
    def exit_and_revert(self):
        raise ProblemExit(self,revert=True)

    def exit_to_level(self,level:str,revert=False):
        raise ProblemExitAtLevel(self,level=level,revert=revert)

    def exit_action(self):
        """handles the exit action wrt system"""
        EOL =(self.class_cache.session is self or self.level_name == 'top')
        if self.revert_last and EOL:
            if self.log_level <= 8:
                self.debug(f'revert last!')
                self.debug(f'revert to{self.x_start}')
            self.revert_to_start()
            
            #run execute
            if self.pre_exec:
                self.pre_execute()

        elif self.revert_every:
            if self.log_level <= 8:
                self.debug(f'revert to{self.x_start}')
            self.revert_to_start()
            
            #run execute 
            if self.pre_exec:
                self.pre_execute()

        #TODO: add exit on success option
        return True #continue as normal

    def error_action(self,error):
        """handles the error action wrt to the problem"""
        if self.log_level <= 11:
            self.debug(f' with input: {self.kwargs}')
        
        if self.fail_revert:
            self.revert_to_start()

        if self.exit_on_failure:
            self.error(error,f'error in execution context')
            return False #send me up
        else:
            self.warning(f'error in execution context: {error}')

        return True #our problem will go on
    

    def save_data(self,index=None,force=False,**add_data):
        """save data to the context"""

        if not self.exited and self.post_exec:
            #a context custom callback
            self.post_execute()

        if self.system.anything_changed or not self.data or force:
            if index is None and self._dxdt == True: #integration
                index = self._time
            elif index is None:
                index = self._index
            out = self.output_state
            if add_data: out.update(add_data)
            out['index'] = index
            self.data[index] = out
            #if we are integrating, then we dont increment the index
            if self._dxdt != True:
                self._index += 1
            
            #reset the data for changed items
            self.system._anything_changed = False
            self.debug(f'data saved = {index}')
        elif self.log_level < 15:
            self.warning(f'no data saved, nothing changed')

    
    def clean_context(self):
        if hasattr(self.class_cache,'session') and self.class_cache.session is self:
            if self.log_level <= 8:
                self.debug(f'closing execution session')
            self.class_cache.level_number = 0
            del self.class_cache.session
        elif hasattr(self.class_cache,'session'):
            #global level number
            self.class_cache.level_number -= 1  

        #if we are the top level, then we mark the session runtime/messages
        if self.session_id == True:
            self._run_end = datetime.datetime.now()
            self._run_time = self._run_end - self._run_start
            if self.log_level <= 10:
                self.debug(f"EXIT[{self.system.identity}] run time: {self._run_time}",lvl=5)

    #time context
    def set_time(self,time,dt):
        self._last_time = lt = self._time
        self._time = time
        dt_calc = time - lt
        self._dt = dt if dt_calc <= 0 else dt_calc
        self.system.set_time(time) #system times / subcomponents too
        

    def integrate(self,endtime,dt=0.001,max_step_dt=0.01,X0=None,**kw):
        #Unpack Transient Problem
        sesh = self.sesh
        intl_refs = sesh.integrator_var_refs #order forms problem basis
        sesh.prv_ingtegral_refs = intl_refs #for rate function
        refs = sesh._sys_refs
        system = sesh.system

        min_kw = sesh._minimizer_kw
        if min_kw is None: min_kw = {}

        if dt > max_step_dt:
            self.warning(f'dt {dt} > max_step_dt {max_step_dt}!')
            dt = max_step_dt

        if self.log_level < 15:
            self.info(f'simulating {system},{sesh}| int:{intl_refs} | refs: {refs}' )            

        if not intl_refs:
            raise Exception(f'no transient parameters found')            
        
        x_cur = {k: v.value(sesh.system,sesh) for k, v in intl_refs.items()}

        if self.log_level < 10:
            self.debug(f'initial state {X0} {intl_refs}| {refs}')

        if X0 is None:
            # get current
            X0 = x_cur
        #add any missing solver vars existing in the system
        if set(X0) != set(x_cur):
            X0 = x_cur.update(X0) 
        
        #this will fail if X0 doesn't have solver vars!
        X0 = np.array([X0[p] for p in intl_refs])
        Time = np.arange(sesh.system.time, endtime + dt, dt)

        rate_kw = {'min_kw':min_kw,'dt':dt}

        #get the probelem variables
        Xss = sesh.problem_opt_vars
        Yobj = sesh.final_objectives

        #run the simulation from the current state to the endtime
        ans = solve_ivp(sesh.integral_rate, [sesh.system.time, endtime], X0, method="RK45", t_eval=Time, max_step=max_step_dt,args=(dt,Xss,Yobj),**kw)

        print(ans)

        return ans

    def integral_rate(self,t, x, dt,Xss=None,Yobj=None, **kw):
        """provides the dynamic rate of the system at time t, and state x"""
        sesh = self.sesh
        intl_refs = sesh.prv_ingtegral_refs #cached in self.integral()
        refs = sesh._sys_refs
        system = sesh.system

        out = {p: np.nan for p in intl_refs}
        Xin = {p: x[i] for i, p in enumerate(intl_refs)}

        if self.log_level < 10:
            self.info(f'sim_iter {t} {x} {Xin}')

        with ProblemExec(system,level_name='tr_slvr',Xnew=Xin,revert_last=False,revert_every=False) as pbx:
            # test for record time 
            
            self.set_time(t,dt)
            
            #save data at the start
            pbx.save_data() #TODO: check_enable/ rate_check

            #ad hoc time integration
            for name, trdct in pbx.integrators.items():
                if self.log_level <=10:
                    
                    self.info(f'updating {trdct.var}|{trdct.var_ref.value(self.system,self)}<-{trdct.rate}|{trdct.current_rate}|{trdct.rate_ref.value(self.system,self)}')
                    print( getattr(self.system,trdct.var,None))
                    print( getattr(self.system,trdct.rate,None))
                        
                out[trdct.var] = trdct.current_rate

            # dynamics
            for compnm, compdict in pbx.dynamic_comps.items():
                comp = compdict#["comp"]
                if not comp.dynamic_state_vars and not comp.dynamic_input_vars:
                    continue
                Xds = np.array([r.value() for r in comp.Xt_ref.values()])
                Uds = np.array([r.value() for r in comp.Ut_ref.values()])
                # time updated in step
                #system.info(f'comp {comp} {compnm} {Xds} {Uds}')
                dxdt = comp.step(t, dt, Xds, Uds, True)

                for i, (p, ref) in enumerate(comp.Xt_ref.items()):
                    out[(f"{compnm}." if compnm else "") + p] = dxdt[i]

            #solvers
            if self.run_solver  and Xss and Yobj and self.solveable:
                # TODO: add in any transient
                with ProblemExec(system,level_name='ss_slvr',revert_last=False,revert_every=False) as pbx:

                    ss_out = pbx.solve_min(Xss, Yobj, **self._minimizer_kw)
                    if ss_out['ans'].success:
                        if self.log_level <= 9:
                            self.info(f'exiting solver {t} {ss_out["Xans"]} {ss_out["Xstart"]}')             
                        pbx.set_ref_values(ss_out['Xans'])
                        pbx.exit_to_level('ss_slvr',False)
                    else:
                        self.warning(f'solver failed to converge {ss_out["ans"].message} {ss_out["Xans"]} {ss_out["X0"]}')
                        if pbx.opt_fail:
                            pbx.exit_to_level('sim',pbx.fail_revert)
                        else:
                            pbx.exit_to_level('ss_slvr',pbx.fail_revert)

            V_dxdt =  np.array([out[p] for p in intl_refs])
            if self.log_level <= 10:
                self.info(f'exiting transient {t} {V_dxdt} {Xin}')
            pbx.exit_to_level('tr_slvr',False)

        if any( np.isnan(V_dxdt) ):
            self.warning(f'solver got infeasible: {V_dxdt}|{Xin}')
            pbx.exit_and_revert()
            #TODO: handle this better, seems to cause a warning
            raise ValueError(f'infeasible! nan result {V_dxdt} {out} {Xin}')
        
        elif self.log_level <= 5:
            self.debug(f'rate {self._dt} {t:5.3f}| {x}<-{V_dxdt} {Xin}')

        return V_dxdt

    def solve_min(
        self,Xref=None,Yref=None,output=None,**kw
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
        sesh = self.sesh
        if Xref is None:
            Xref = sesh.Xref

        if Yref is None:
            Yref = sesh.final_objectives

        thresh = kw.pop("thresh", sesh.success_thresh)

        #TODO: options for solver detail in response
        dflt = {
                "Xstart": Ref.refset_get(Xref,sys=sesh.system,prob=sesh),
                "Ystart": Ref.refset_get(Yref,sys=sesh.system,prob=sesh),
                "Xans": None,
                "success": None,
                "Xans":None,
                "Yobj":None,
                "Ycon":None,
                "ans": None,
                "weights":sesh._weights,
                "constraints":sesh.constraints,
            }

        if output:
            dflt.update(output)
            output = dflt
        else:
            output = dflt

        if len(Xref) == 0:
            self.debug(f'no variables found for solver: {kw}')
            #None for `ans` will not trigger optimization failure        
            return output
        
        #override constraints input
        kw.update(sesh.constraints)

        if len(kw['bounds']) != len(Xref):
            raise ValueError(f"bounds {len(sesh.constraints['bounds'])} != Xref {len(Xref)}")

        if self.log_level < 10:
            self.debug(f"minimize {Xref} {Yref} {kw}")
        
        if sesh._weights is not None:
            kw['weights'] = sesh._weights

        sesh._ans = refmin_solve(sesh.system,self,Xref, Yref, **kw)
        output["ans"] = sesh._ans

        sesh.handle_solution(sesh._ans,Xref,Yref,output)

        return output

    def handle_solution(self,answer,Xref,Yref,output):
        #TODO: move exit condition handiling somewhere else, reduce cross over from process_ans
        sesh = self.sesh
        
        thresh = sesh.success_thresh
        vars = list(Xref)

        #Output Results
        Xa = {p: answer.x[i] for i, p in enumerate(vars)}
        output["Xans"] = Xa
        Ref.refset_input(Xref,Xa)

        Yout = {p: Yref[p].value(sesh.system,self) for p in Yref}
        output["Yobj"] = Yout

        Ycon = {}
        if sesh.constraints['constraints']:
            x_in = answer.x
            for c,k in zip(sesh.constraints['constraints'],sesh.constraints['info']):
                cv = c['fun'](x_in,self,{})
                Ycon[k] = cv
        output['Ycon'] = Ycon   

        de = answer.fun
        if answer.success and de < thresh if thresh else True:
            sesh.system._converged = True #TODO: put in context
            output["success"] = True

        elif answer.success:
            # out of threshold condition
            self.warning(
                f"solver didnt meet threshold: {de} <? {thresh} ! {answer.x} -> residual: {answer.fun}"
            )
            sesh.system._converged = False
            output["success"] = False  # only false with threshold

        else:
            sesh.system._converged = False
            if self.opt_fail:
                raise Exception(f"solver didnt converge: {answer}")
            else:
                self.warning(f"solver didnt converge: {answer}")
            output["success"] = False
    
        return output 

    #Solver Parsing Methods
    
    def sys_solver_objectives(self,**kw):
        """gathers variables from solver vars, and attempts to locate any input_vars to add as well. use exclude_vars to eliminate a variable from  the solver
        """
        sys_refs = self.sesh._sys_refs

        #Convert result per kind of objective (min/max ect)
        objs = sys_refs.get('attrs',{}).get('solver.obj',{})
        return {k:v for k,v in objs.items()}
    
    @property
    def final_objectives(self)->dict:
        """returns the final objective of the system"""
        sesh = self.sesh
        Yobj = sesh.problem_objs
        Yeq = sesh.problem_eq
        Xss = sesh.problem_opt_vars
        if Yobj:
            return Yobj #here is your application objective, sir
        
        #now make up an objective, if we have no constraints
        elif not Yobj and Yeq:
            #handle case of Yineq == None: root solve
            self.debug(f'making Yobj from Yeq: {Yeq}')
            Yobj = { k: v.copy(key = lambda sys,prob: (1+v.value(sys,prob)**2)) for k,v in Yeq.items() }

        elif not Yobj:
            #minimize the product of all vars, so the smallest value is the best that satisfies all constraints
            self.debug(f'making Yobj from X: {Xss}')
            def dflt(sys,prob)->float:
                out = 1
                for k,v in prob.problem_opt_vars.items():
                    val = v.value(sys,prob)
                    out = out + (1+val**2)**0.5 #linear norm of positive values > 1 should be very large penalty 
                return 1
            
            Yobj = {'smallness': Ref(sesh.system, dflt)}
        return Yobj #our residual based objective

    def sys_solver_constraints(self,add_con=None,combo_filter=True, **kw):
        """formatted as arguments for the solver
        """
        from engforge.solver_utils import create_constraint
        sesh = self.sesh
        Xrefs = sesh.Xref

        system = sesh.system
        sys_refs = sesh._sys_refs

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
        Nstates = len(Xrefs)
        Xvars = list(Xrefs) #get names of solvers + dynamics

        # constraints lookup
        bnd_list = [[None, None]] * Nstates
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
        for slvr, ref in sesh.problem_opt_vars.items():
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
                self.debug(f'constraints {slvr} {slv_constraints}')
                
            for ctype in slv_constraints:
                cval = ctype['value']
                kind = ctype['type']       
                var = ctype['var']
                if log.log_level < 3:
                    self.msg(f'const: {slvr} {ctype}')

                if cval is not None and slvr in Xvars:
                    
                    #Check for combos & activation
                    combos = None
                    if 'combos' in ctype:
                        combos = ctype['combos']
                        combo_var = ctype['combo_var']
                        active = ctype.get('active',True)
                        in_activate = any([arg_var_compare(combo_var,v) for v in activated]) if activated else False
                        in_deactivate = any([arg_var_compare(combo_var,v) for v in deactivated]) if deactivated else False

                        if log.log_level <= 5:
                            self.debug(f'filter combo: {ctype}=?{extra_kw}') 

                        #Check active or activated
                        if not active and not activated:
                            if log.log_level < 3:
                                self.msg(f'skip con: inactive {var} {slvr} {ctype}')
                            continue

                        elif not active and not in_activate:
                            if log.log_level < 3:
                                self.msg(f'skip con: inactive {var} {slvr} {ctype}')
                            continue

                        elif active and in_deactivate:
                            if log.log_level < 3:
                                self.msg(f'skip con: deactivated {var} {slvr} ')
                            continue

                    if combos and combo_filter:
                        filt = filter_combos(combo_var,slv, extra_kw,combos)
                        if not filt:
                            if log.log_level < 5:
                                self.debug(f'filtering constraint={filt} {var} |{combos}')  
                            continue
                    
                    if log.log_level < 10:
                        self.debug(f'adding var constraint {var,slvr,ctype,combos}') 

                    #get the index of the variable
                    x_inx = Xvars.index(slvr)        
                    
                    #lookup rates
                    rate_val = None
                    if sesh._dxdt is not None:
                        if isinstance(sesh._dxdt,dict) and not slv_var:
                            rate_val = sesh._dxdt.get(slvr,0)
                        elif not slv_var:
                            rate_val = 0

                    #add the dynamic parameters when configured
                    if not slv_var and rate_val is not None:
                        #if kind in ('min','max') and slvr in Xvars:
                        varref = Xrefs[slvr]
                        #varref = slv.rate_ref
                        #Ref Case
                        ccst = ref_to_val_constraint(system,system.last_context,Xrefs,varref,kind,rate_val,**kw)
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
                            ccst = ref_to_val_constraint(system,system.last_context,Xrefs,varref,kind,cval,**kw)
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
                            system,Xrefs, 'ineq', cval, **kw
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
                            system,Xrefs, 'eq', cval, **kw
                        )
                    )


        return constraints 

    # General method to distribute input to internal components
    @classmethod
    def parse_default(self,key,defaults,input_dict,rmv=False,empty_str=True):
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
                if not empty_str and not option:
                     return None,False
                option = option.split(',')
                    
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

    #State Interfaces
    @property
    def record_state(self)->dict:
        """records the state of the system using session"""
        #refs = self.all_variable_refs
        sesh = self.sesh
        refs = sesh.all_comps_and_vars
        return Ref.refset_get(refs,sys=sesh.system,prob=self)

    @property
    def output_state(self)->dict:
        """records the state of the system"""
        sesh = self.sesh
        if 'nums' == sesh.save_mode:
            refs = sesh.num_refs
        elif 'all' == sesh.save_mode:
            refs = sesh.all_system_references
        elif 'vars' == sesh.save_mode:
            refs = self.all_variable_refs
        elif 'prob' == sesh.save_mode:
            raise NotImplementedError(f'problem save mode not implemented')
        else:
            raise KeyError(f'unknown save mode {self.save_mode}, not in {save_modes}')

        out = Ref.refset_get(refs,sys=sesh.system,prob=self)

        if self._dxdt == True:
            out['time'] = self._time

        return out
    
    
    def get_ref_values(self,refs=None):
        """returns the values of the refs"""
        sesh = self.sesh
        if refs is None:
            refs = sesh.all_system_references
        return Ref.refset_get(refs,sys=self.system,prob=self)
    
    def set_ref_values(self,values,refs=None):
        """returns the values of the refs"""
        #TODO: add checks for the refs
        sesh = self.sesh
        if refs is None:
            refs = sesh.all_comps_and_vars        
        return Ref.refset_input(refs,values)    

    def set_checkpoint(self):
        """sets the checkpoint"""
        self.x_start = self.record_state
        if log.log_level <= 7:
            self.debug(f'set checkpoint: {list(self.x_start.values())}')

    def revert_to_start(self):
        sesh = self.sesh
        if log.log_level < 5:
            xs = list(self.x_start.values())
            rs = list(self.record_state.values())
            self.debug(f'reverting to start: {xs} -> {rs}')
        #TODO: STRICT MODE Fail for refset_input
        Ref.refset_input(sesh.all_comps_and_vars,self.x_start,fail=False)

    def activate_temp_state(self,new_state=None):
        #TODO: determine when components change, and update refs accordingly!
        sesh = self.sesh
        #TODO: STRICT MODE Fail for refset_input
        if new_state:
            Ref.refset_input(sesh.all_comps_and_vars,new_state,fail=False)
        elif self.temp_state:
            self.debug(f'activating temp state: {self.temp_state}')
            Ref.refset_input(sesh.all_comps_and_vars,self.temp_state,fail=False)
        elif self.log_level < 9:
            self.debug(f'no state to set: {new_state}')
        

    #System Events
    def apply_pre_signals(self):
        """applies all pre signals"""
        msg_lvl = self.log_level <= 2
        if self.log_level < 5:
            self.msg(f"applying pre signals",lvl=6)
        for signame, sig in self.sesh.signals.items():
            if sig.mode == "pre" or sig.mode == "both":
                if msg_lvl:
                    self.msg(f"applying post signals: {signame}",lvl=3)                
                sig.apply()

    def apply_post_signals(self):
        """applies all post signals"""
        msg_lvl = self.log_level <= 2
        if self.log_level < 5:
            self.msg(f"applying post signals",lvl=6)        
        for signame, sig in self.sesh.signals.items():
            if sig.mode == "post" or sig.mode == "both":
                if msg_lvl:
                    self.msg(f"applying post signals: {signame}",lvl=3)
                sig.apply()

    def update_system(self,*args,**kwargs):
        """updates the system"""
        for ukey,uref in self.sesh._update_refs.items():
            self.debug(f'context updating {ukey}')
            uref.value(*args,**kwargs)

    def post_update_system(self,*args,**kwargs):
        """updates the system"""
        for ukey,uref in  self.sesh._post_update_refs.items():
            self.debug(f'context post updating {ukey}')
            uref.value(*args,**kwargs)

    def pre_execute(self,*args,**kwargs):
        """Updates the pre/both signals after the solver has been executed. This is useful for updating the system state after the solver has been executed."""
        if log.log_level < 5:
            self.msg(f"pre execute")
        sesh = self.sesh
        sesh.apply_pre_signals()
        sesh.update_system(*args,**kwargs)


    def post_execute(self,*args,**kwargs):
        """Updates the post/both signals after the solver has been executed. This is useful for updating the system state after the solver has been executed."""
        if log.log_level < 5:
            self.msg(f"post execute")
        sesh = self.sesh
        sesh.apply_post_signals()
        sesh.post_update_system(*args,**kwargs)

    


    #Logging to class logger
    @property
    def identity(self):
        return f'PROB|{self.level_name}|{str(self.session_id)[0:5]}'

    @property
    def log_level(self):
        return log.log_level

    def msg(self,msg,*a,**kw):
        if log.log_level < 5:
            log.msg(f'{self.identity}|[{self.level_number}-{self.level_name}]  {msg}',*a,**kw)

    def debug(self,msg,*a,**kw):
        if log.log_level <= 15:
            log.debug(f'{self.identity}|[{self.level_number}-{self.level_name}]  {msg}',*a,**kw)

    def warning(self,msg,*a,**kw):
        log.warning(f'{self.identity}|[{self.level_number}-{self.level_name}]  {msg}',*a,**kw)

    def info(self,msg,*a,**kw):
        log.info(f'{self.identity}|[{self.level_number}-{self.level_name}]  {msg}',*a,**kw)

    def error(self,error,msg,*a,**kw):
        log.error(error,f'{self.identity}|[{self.level_number}-{self.level_name}]  {msg}',*a,**kw)        

    def critical(self,msg,*a,**kw):
        log.critical(f'{self.identity}|[{self.level_number}-{self.level_name}]  {msg}',*a,**kw)

    #Safe Access Methods
    @property
    def ref_attrs(self):
        return self.sesh._sys_refs.get('attrs',{}).copy()

    @property
    def attr_inst(self):
        return self.sesh._sys_refs.get('type',{}).copy()
    
    @property
    def dynamic_comps(self):
        return self.sesh._sys_refs.get('dynamic_comps',{}).copy()
    
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
        """copy of slv_kw args"""
        return self.sesh._slv_kw.copy()


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
        if self.sesh.problem_opt_vars:
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
    
    #Dataframe support

    @property
    def dataframe(self)->pd.DataFrame:
        """returns the dataframe of the system"""
        res = pd.DataFrame([kv[-1] for kv in sorted(self.data.items(),
                                                   key=lambda kv:kv[0]) ])
        self.system.format_columns(res)
        return res

    #TODO: expose optoin for saving all or part of the system information, for now lets default to all (saftey first, then performance :)
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
        #Add the dynamic states to be optimized (ignore if integrating)
        sesh = self.sesh
        if sesh.dynamic_solve and not sesh._dxdt is True:
            varx.update(self.filter_vars(sesh.dynamic_state))
            varx.update(self.filter_vars(sesh.integrator_vars))
        return varx

    @property
    def dynamic_solve(self)->bool:
        """indicates if the system is dynamic"""
        sesh = self.sesh
        dxdt = sesh._dxdt

        if dxdt is None or dxdt is False:
            return False
        
        if dxdt is True:
            return True
        
        in_type = isinstance(dxdt,(dict,float,int))
        bool_type = (isinstance(dxdt,bool) and dxdt == True)
        if in_type or bool_type:
            return True
        
        return False

    @property
    def all_variable_refs(self)->dict:
        sesh = self.sesh
        ing = self.integrator_vars
        stt = self.dynamic_state
        vars = self.problem_opt_vars
        return {**ing,**stt,**vars}
    
    @property
    def all_variables(self)->dict:
        """returns all variables in the system"""
        return self.all_refs['attributes']

    @property
    def all_comps_and_vars(self)->dict:
        #TODO: ensure system refes are fresh per system runtime events
        sesh = self.sesh
        refs = sesh.all_refs
        attrs = refs['attributes'].copy()
        comps = refs['components'].copy()
        attrs.update(comps)
        return attrs
    
    @property
    def all_system_references(self)->dict:
        sesh = self.sesh
        refs = sesh.all_refs
        out = {}
        out.update(refs['attributes'])
        out.update(refs['properties'])
        return out
    
    def __str__(self):
        #TODO: expand this
        return f'ProblemContext[{self.level_name:^12}][{str(self.session_id)[0:8]}-{str(self._problem_id)[0:8]}][{self.system.identity}]'
    

#TODO: move all system_reference concept inside problem context, remove from system/tabulation ect.
#TODO: use prob.register(comp,key='') to add components to the problem context, mapping subcomponents to the problem context

    
#subclass before altering please!
ProblemExec.class_cache = ProblemExec

class Problem(ProblemExec,DataframeMixin):
    #TODO: implement checks to ensure that problem is defined as the top level context to be returned to
    #TODO: also define return options for data/system/dataframe and indexing
    pass

    @property
    def level_name(self):
        return 'top' #fixed top output, garuntees exit to here.
    
    @level_name.setter
    def level_name(self,value):
        raise AttributeError(f'cannot set level_name of top level problem context')



