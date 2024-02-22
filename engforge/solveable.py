import attrs
import uuid
import numpy
import numpy as np
import scipy.optimize as sciopt
from contextlib import contextmanager
import copy
import datetime


#from engforge.dynamics import DynamicsMixin
from engforge.engforge_attributes import AttributedBaseMixin
from engforge.properties import *

import collections
import itertools

SOLVER_OPTIONS = ["root", "minimize"]


class SolvableLog(LoggingMixin):
    pass


log = SolvableLog()

def refset_input(refs,delta_dict):
    for k, v in delta_dict.items():
        refs[k].set_value(v)

def refset_get(refs):
    return {k: refs[k].value() for k in refs}

def refmin_solve(Xref:dict,Yref:dict,Xo=None,normalize:np.array=None,reset=True,doset=True,fail=True,**kw):
    """minimize the difference between two dictionaries of refernces, x references are changed in place, y will be solved to zero, options ensue for cases where the solution is not ideal
    
    :param Xref: dictionary of references to the x values
    :param Yref: dictionary of references to the y values
    :param Xo: initial guess for the x values as a list against Xref order, or a dictionary
    :param normalize: a dictionary of values to normalize the x values by, list also ok as long as same length and order as Xref 
    :param reset: if the solution fails, reset the x values to their original state, if true will reset the x values to their original state on failure overiding doset. 
    :param doset: if the solution is successful, set the x values to the solution by default, otherwise follows reset, if not successful reset is checked first, then doset
    """
    parms = list(Xref.keys()) #static x_basis
    yref_parm = list(Yref.keys()) #static y_basis
    if normalize is None:
        normalize = np.ones(len(parms))
    elif isinstance(normalize,(list,tuple,np.ndarray)):
        assert len(normalize)==len(parms), "bad length normalize"
    elif isinstance(normalize,(dict)):
        normalize = np.array([normalize[p] for p in parms])

    #make anonymous function
    def f(x):
        inpt = {}
        for p,xi in zip(parms,x):
            inpt[p] = xi
            Xref[p].set_value(xi)
        
        #if method == 'minimize':
        grp = (yref_parm,x,normalize)
        vals = [(Yref[p].value()/n)**2 for p,x,n in zip(*grp)]
        #print(vals,inpt)
        rs = np.array(vals)
        out = np.sum(rs)**0.5
        return out
    
    #get state
    x_pre = refset_get(Xref) #record before changing

    if Xo is None:
        Xo = [Xref[p].value() for p in parms]

    elif isinstance(Xo,dict):
        Xo = [Xo[p] for p in parms]

    #solve
    #print(Xref,Yref,Xo,normalize)
    #TODO: IO for jacobean and state estimations (complex as function definition, requires learning)
    ans = sciopt.minimize(f,Xo,**kw)
    #print(ans)
    if ans.success:
        ans_dct = {p:a for p,a in zip(parms,ans.x)}
        if doset:
            refset_input(Xref,ans_dct)        
        if reset: 
            refset_input(Xref,x_pre)
        return ans_dct
        
    else:
        min_dict = {p:a for p,a in zip(parms,ans.x)}
        if reset: 
            refset_input(Xref,x_pre)
        if doset:
            refset_input(Xref,min_dict)            
            return min_dict
        if fail:
            raise Exception(f"failed to solve {ans.message}")
        return min_dict

    



class Ref:
    """A way to create portable references to system's and their component's properties, ref can also take a key to a zero argument function which will be evaluated,
    
    A dictionary can be used
    """

    __slots__ = ["comp", "key", "use_call",'use_dict','allow_set','eval_f']
    comp: "TabulationMixin"
    key: str
    use_call: bool
    use_dict: bool
    allow_set: bool
    eval_f: callable

    def __init__(self,component, key, use_call=True,allow_set=True,eval_f=None):
        self.comp = component
        if isinstance(self.comp,dict):
            self.use_dict = True
        else:
            self.use_dict = False
        self.key = key
        self.use_call = use_call
        self.allow_set = allow_set
        self.eval_f = eval_f

    def value(self):
        if self.use_dict:
            o = self.comp.get(self.key)
        else:
            o = getattr(self.comp, self.key)
        if self.use_call and callable(o):
            o = o()
        if self.eval_f:
            return self.eval_f(o)
        return o

    def set_value(self, val):
        if self.allow_set:
            return setattr(self.comp, self.key, val)
        else:
            raise Exception(f'not allowed to set value on {self.key}')

    def __str__(self) -> str:
        if self.use_dict:
            return f"REF[DICT.{self.key}]"
        return f"REF[{self.comp.classname}.{self.key}]"

class SolveableMixin(AttributedBaseMixin): #'Configuration'
    """commonality for components,systems that identifies subsystems and states for solving.
    
    This class defines the update structure of components and systems, and the storage of internal references to the system and its components. It also provides a method to iterate over the internal components and their references.

    Importantly it defines the references to the system and its components, and the ability to set the system state from a dictionary of values across multiple objects. It also provides a method to iterate over the internal components and their references. There are several helper functions to these ends.
    """

    #TODO: add parent state storage
    parent: 'Configuration'
    
    _prv_internal_references: dict 
    _prv_internal_components: dict
    _prv_internal_systems: dict
    _prv_internal_tabs: dict
    _prv_system_references: dict

    #Update Flow
    def update(self,parent,*args,**kwargs):
        pass
        
    def post_update(self,parent,*args,**kwargs):
        pass        
        
    def update_internal(self,eval_kw=None,ignore=None,*args,**kw):
        """update internal elements with input arguments"""
        # Solve Each Internal System
        from engforge.components import Component
        from engforge.component_collections import ComponentIter
        
        #Ignore 
        if ignore is None:
            ignore = set()
        elif self in ignore:
            return
        
        self.update(self.parent,*args,**kw)
        self.debug(f'updating internal {self.__class__.__name__}.{self}')
        for key, comp in self.internal_configurations(False).items():
            
            if ignore is not None and comp in ignore:
                continue

            #provide add eval_kw
            if eval_kw and key in eval_kw:
                eval_kw_comp = eval_kw[key]
            else:
                eval_kw_comp = {}              
            
            #comp update cycle
            self.debug(f"updating {key} {comp.__class__.__name__}.{comp}")
            if isinstance(comp, ComponentIter):
                comp.update(self,**eval_kw_comp)
            elif isinstance(comp, (SolveableMixin,Component)):
                comp.update(self,**eval_kw_comp)
                comp.update_internal(eval_kw=eval_kw_comp,ignore=ignore)
        ignore.add(self) #add self to ignore list

    def post_update_internal(self,eval_kw=None,ignore=None,*args,**kw):
        """Post update all internal components"""
        #Post Update Self
        from engforge.components import Component
        from engforge.component_collections import ComponentIter

        #Ignore 
        if ignore is None:
            ignore = set()
        elif self in ignore:
            return

        self.post_update(self.parent,*args,**kw)
        self.debug(f'post updating internal {self.__class__.__name__}.{self}')
        for key, comp in self.internal_configurations(False).items():
            
            if ignore is not None and comp in ignore:
                continue

            #provide add eval_kw
            if eval_kw and key in eval_kw:
                eval_kw_comp = eval_kw[key]
            else:
                eval_kw_comp = {}   

            self.debug(f"post updating {key} {comp.__class__.__name__}.{comp}")
            if isinstance(comp, ComponentIter):
                comp.post_update(self,**eval_kw_comp)
            elif isinstance(comp, (SolveableMixin,Component)):
                comp.post_update(self,**eval_kw_comp)
                comp.post_update_internal(eval_kw=eval_kw_comp,ignore=ignore)
        ignore.add(self)

    #internals caching
    @instance_cached
    def signals(self):
        return {k: getattr(self, k) for k in self.signals_attributes()}

    @instance_cached
    def solvers(self):
        return {k: getattr(self, k) for k in self.solvers_attributes()}

    @instance_cached(allow_set=True)
    def transients(self):
        #TODO: add dynamics internals
        return {k: getattr(self, k) for k in self.transients_attributes()}

    def internal_components(self,recache=False) -> dict:
        """get all the internal components"""
        if recache == False and hasattr(self,'_prv_internal_components'):
            return self._prv_internal_components 
        from engforge.components import Component
        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, Component)}
        self._prv_internal_components = o
        return o
    
    def internal_systems(self,recache=False) -> dict:
        """get all the internal components"""
        if recache == False and hasattr(self,'_prv_internal_systems'):
            return self._prv_internal_systems 
        from engforge.system import System
        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, System)}
        self._prv_internal_systems = o
        return o    
    
    def internal_tabulations(self,recache=False) -> dict:
        """get all the internal tabulations"""
        from engforge.tabulation import TabulationMixin
        if recache == False and hasattr(self,'_prv_internal_tabs'):
            return self._prv_internal_tabs 
            
        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, TabulationMixin)}
        self._prv_internal_tabs = o
        return o    

    @instance_cached
    def iterable_components(self) -> dict:
        """Finds ComponentIter internal_components that are not 'wide'"""
        from engforge.component_collections import ComponentIter

        return {
            k: v
            for k, v in self.internal_components().items()
            if isinstance(v, ComponentIter) and not v.wide
        }

    def internal_references(self,recache=False) -> dict:
        """get references to all internal attributes and values"""
        if recache == False and hasattr(self,'_prv_internal_references'):
            return self._prv_internal_references  
              
        out = self._gather_references()
        self._prv_internal_references = out
        return out
    
    def _gather_references(self) -> dict:
        out = {}
        out["attributes"] = at = {}
        out["properties"] = pr = {}

        for key in self.system_properties_classdef():
            pr[key] = Ref(self, key,True,False)

        for key in self.input_fields():
            at[key] = Ref(self, key, False,True)

        return out
    
    #Refs Ect.
    @classmethod
    def locate(cls, key, fail=True) -> type:
        """:returns: the class or attribute by key if its in this system class or a subcomponent. If nothing is found raise an error"""
        # Nested
        log.debug(f"locating {cls.__name__} | key: {key}")
        val = None

        if "." in key:
            args = key.split(".")
            comp, sub = args[0], ".".join(args[1:])
            assert comp in cls.slots_attributes(), f"invalid {comp} in {key}"
            comp_cls = cls.slots_attributes()[comp].type.accepted[0]
            val = comp_cls.locate(sub, fail=False)

        elif key in cls.input_fields():
            val = cls.input_fields()[key]

        elif key in cls.system_properties_classdef():
            val = cls.system_properties_classdef()[key]

        # Fail on comand but otherwise return val
        if val is None:
            if fail:
                raise Exception(f"key {key} not found")
            return None
        return val

    def locate_ref(self, key, fail=True,**kw):
        """:returns: the instance assigned to this system. If the key has a `.` in it the comp the lowest level component will be returned"""

        log.debug(f"locating {self.identity} | key: {key}")
        val = None

        if "." in key:
            args = key.split(".")
            comp, sub = args[0], ".".join(args[1:])
            assert comp in self.slots_attributes(), f"invalid {comp} in {key}"
            # comp_cls = cls.slots_attributes()[comp].type.accepted[0]
            comp = getattr(self, comp)
            if "." not in key:
                return Ref(comp, sub,**kw)
            return comp.locate_ref(sub, fail=fail,**kw)

        elif key in self.input_fields():
            # val= cls.input_fields()[key]
            return Ref(self, key,**kw)

        elif key in self.system_properties_classdef():
            # val= cls.system_properties_classdef()[key]
            return Ref(self, key,**kw)
        
        elif key in self.internal_configurations() or key in self.slots_attributes():
            return Ref(self,key,**kw)

        # Fail on comand but otherwise return val
        if val is None:
            if fail:
                raise Exception(f"key {key} not found")
            return None
        return val
    
    #@instance_cached
    @property
    def comp_references(self):
        """A cached set of recursive references to any slot component
        #TODO: work on caching, concern with iterators
        """
        out = {}
        for key, lvl, comp in self.go_through_configurations(parent_level=1):
            if not isinstance(comp, SolveableMixin):
                continue
            out[key] = comp
        return out

    #@property
    def system_references(self,recache=False,child_only=True):
        """gather a list of references to attributes and"""
        if recache == False and hasattr(self,'_prv_system_references'):
            return self._prv_system_references
        
        out = self.internal_references(recache)
        tatr = out["attributes"]
        tprp = out["properties"]

        # component iternals
        for key, comp in self.comp_references.items():
            sout = comp.internal_references(recache)
            satr = sout["attributes"]
            sprp = sout["properties"]

            # Fill in
            for k, v in satr.items():
                tatr[f"{key}.{k}"] = v

            for k, v in sprp.items():
                tprp[f"{key}.{k}"] = v
        
        self._prv_system_references = out
        return out

    @instance_cached
    def all_references(self) -> dict:
        out = {}
        sysref = self.system_references()
        out.update(**sysref["attributes"])
        out.update(**sysref["properties"])
        return out

    @property
    def system_state(self):
        """records all attributes"""
        out = collections.OrderedDict()
        sref = self.system_references()
        for k, v in sref["attributes"].items():
            out[k] = v.value()
        self.debug(f"recording system state: {out}")
        return out

    def set_system_state(self, ignore=None, **kwargs):
        """accepts parital input scoped from system references"""
        sref = self.system_references()
        satr = sref["attributes"]
        self.debug(f"setting system state: {kwargs}")
        for k, v in kwargs.items():
            if ignore and k in ignore:
                continue
            if k not in satr:
                self.debug(f'skipping {k} not in attributes')
                continue
            ref = satr[k]
            ref.set_value(v)


    #Run & Input
    #General method to distribute input to internal components          

    def _iterate_input_matrix(self,method,revert=True, cb=None, sequence:list=None,eval_kw:dict=None,sys_kw:dict=None, force_solve=False,return_results=False,**kwargs):
        """applies a permutation of input parameters for parameters. runs the system instance by applying input to the system and its slot-components, ensuring that the targeted attributes actualy exist. 

        :param revert: will reset the values of X that were recorded at the beginning of the run.
        :param cb: a callback function that takes the system as an argument cb(system)
        :param sequence: a list of dictionaries that should be run in order per the outer-product of kwargs
        :param eval_kw: a dictionary of keyword arguments to pass to the evaluate function of each component by their name and a set of keyword args. Use this to set values in the component that are not inputs to the system. No iteration occurs upon these values, they are static and irrevertable
        :param sys_kw: a dictionary of keyword arguments to pass to the evaluate function of each system by their name and a set of keyword args. Use this to set values in the component that are not inputs to the system. No iteration occurs upon these values, they are static and irrevertable
        :param kwargs: inputs are run on a product basis asusming they correspond to actual scoped parameters (system.parm or system.slot.parm)


        :returns: system or list of systems. If transient a set of systems that have been run with permutations of the input, otherwise a single system with all permutations input run
        """
        from engforge.system import System
        self.debug(f"running [SOLVER].{method} {self.identity} with input {kwargs}")

        # TODO: allow setting sub-component parameters with `slot1.slot2.attrs`. Ensure checking slots exist, and attrs do as well.
        
        #Recache system references
        if isinstance(self,System):
            self.system_references(recache=True)

        #create iterable null for sequence
        if sequence is None or not sequence:
            sequence = [{}]

        #Create Keys List
        sequence_keys = set()
        for seq in sequence:
            sequence_keys = sequence_keys.union(set(seq.keys()))

        # RUN when not solved, or anything changed, or arguments or if forced
        if force_solve or not self.solved or self.anything_changed or kwargs:
            _input = self.parse_run_kwargs(**kwargs)
            
            output = {}
            inputs = {}
            result = {'output':output,'input_sets':inputs}

            if revert:
                # TODO: revert all internal components too with `system_state` and set_system_state(**x,comp.x...)
                revert_x = self.system_state

            # prep references for keys
            refs = {}
            for k, v in _input.items():
                refs[k] = self.locate_ref(k)
            for k in sequence_keys:
                refs[k] = self.locate_ref(k)

            #Pre Run Callback
            self.pre_run_callback(eval_kw=eval_kw,sys_kw=sys_kw,**kwargs)

            # Premute the input as per SS or Transient Logic
            ingrp = list(_input.values())
            keys = list(_input.keys())
            #Iterate over components (they are assigned to the system by call)
            for itercomp in self._iterate_components():
                #Iterate over inputs
                for parms in itertools.product(*ingrp):
                    # Set the reference aliases
                    cur = {k: v for k, v in zip(keys, parms)}
                    #Iterate over Sequence (or run once)
                    for seq in sequence:
                        #apply sequenc evalues
                        icur = cur.copy()
                        if seq:
                            icur.update(**seq)
                        
                        #Run The Method with inputs provisioned
                        out = method(refs,icur,eval_kw,sys_kw,cb=cb)

                        if return_results:
                            #store the output
                            output[max(output)+1 if output else 0] = out
                            #store the input
                            inputs[max(inputs)+1 if inputs else 0] = icur
            
            #nice
            self._solved = True
            
            #Pre Run Callback with current state
            self.post_run_callback(eval_kw=eval_kw,sys_kw=sys_kw,**kwargs)
            
            #pre-revert by default
            if revert and revert_x:
                self.set_system_state(ignore=["index"], **revert_x)
            
            #reapply solved state
            self._solved = True
            
            if return_results:
                return result

        elif not self.anything_changed:
            self.warning(f'nothing changed, not running {self.identity}')
            return
        elif self.solved:
            raise Exception("Analysis Already Solved")
        
    def collect_solver_dynamics(self,conf:"ConfigurationMixin"=None,**kw)->dict:
        """collects the dynamics of the systems"""
        if conf is None:
            conf = self
        
        dynamics = {}
        traces = {}
        signals = {}
        solvers = {}  
        out = dict(
            dynamics = dynamics,
            traces = traces,
            signals = signals,
            solvers = solvers
            )
        
        for key,lvl,conf in conf.go_through_configurations(**kw):
            #FIXME: add a check for the dynamics mixin, that isn't hacky
            #BUG: importing dynamicsmixin resolves as different class in different modules, weird
            if 'dynamicsmixin' in str(conf.__class__.mro()).lower():
                dynamics[key] = {'lvl':lvl,'conf':conf}

            tra = conf.transients_attributes()
            if tra:
                trec = {k:at.type for k,at in tra.items()}
                traces[key] = {'lvl':lvl,'conf':conf,
                               'transients':trec}

            #map signals and slots
            sig = conf.signals_attributes()
            if sig:
                sigc = {k:at.type for k,at in sig.items()}
                signals[key] = {'lvl':lvl,'conf':conf,
                                'signals':sigc}

            solv = conf.solvers_attributes()
            if solv:
                solvc = {k:at.type for k,at in solv.items()}
                solvers[key] = {'lvl':lvl,'conf':conf,
                                'solvers':solvc}

        return out           
    
    #Reference Interaction
    def collect_solver_refs(self,conf:"ConfigurationMixin"=None,by_parm=False,min_set=True,**kw)->dict:
        """collects the solver references of the systems"""
        internal_dynamics = self.collect_solver_dynamics(conf,**kw)

        dynamics = {}
        transients = {}
        solvers = {}
        signals = {}

        #parameter lists
        tr_inputs = {} #TODO: expand user input idea :)
        ss_output = {}
        tr_states = {}
        tr_output = {}
        ss_states = {}
        sig_source = {}
        sig_target = {}
        dyn_comp = {}
        tr_sets = {}

        out = dict(
                tr_inputs = tr_inputs,
                tr_states = tr_states,
                tr_output = tr_output,
                tr_sets = tr_sets,
                ss_states = ss_states,              
                ss_output =   ss_output,
                sig_source = sig_source,
                sig_target = sig_target,            
                dynamics = dynamics,
                transients = transients,
                solvers = solvers,
                signals = signals,
                dyn_comp = dyn_comp
                )
        
        #Collect By Category
        for ck,cv in internal_dynamics['traces'].items():
            comp = cv['conf']
            for k,v in cv['transients'].items():
                
                p_par = comp.locate_ref(v.parameter,True)
                d_par = comp.locate_ref(v.derivative,True)

                #outputs.append(f'{ck+"." if ck else ""}{v.derivative}')
                tr_states[f'{ck+"." if ck else ""}{v.parameter}'] = p_par
                tr_output[f'{ck+"." if ck else ""}{v.derivative}']= d_par
                transients[f'{ck+"." if ck else ""}{k}'] = dict(parm=p_par,dpdt=d_par)
                tr_sets[f'{ck+"." if ck else ""}{v.parameter}'] = {'comp':comp,'parm':p_par,'dpdt':d_par}
            
        for ck,cv in internal_dynamics['solvers'].items():
            comp = cv['conf']
            for k,v in cv['solvers'].items():

                
                p_par = comp.locate_ref(v.dependent)
                d_par = comp.locate_ref(v.independent)

                #ss_states.append(f'{ck+"." if ck else ""}{v.independent}')
                #outputs.append(f'{ck+"." if ck else ""}{v.dependent}')
                ss_output[f'{ck+"." if ck else ""}{v.dependent}'] = p_par
                ss_states[f'{ck+"." if ck else ""}{v.independent}'] = d_par

                solvers[f'{ck+"." if ck else ""}{k}']= dict(dep=p_par,indep=d_par)
                
         
        for ck,cv in internal_dynamics['signals'].items():
            comp = cv['conf']
            for k,v in cv['signals'].items():
                p_par = comp.locate_ref(v.target)
                d_par = comp.locate_ref(v.source)
                signals[f'{ck+"." if ck else ""}{k}']= dict(target=p_par,source=d_par,mode=v.mode)
                sig_source[f'{ck+"." if ck else ""}{v.source}'] = d_par
                sig_target[f'{ck+"." if ck else ""}{v.target}'] = p_par

        
        for ck,cv in internal_dynamics['dynamics'].items():
            comp = cv['conf']
            dynamics[ck] = dyn_ref = {}
            dyni = {'comp':comp,'state':{},'input':{},'output':{}}
            dyn_comp[ck] = dyni
            if comp.dynamic_state_parms:
                dyn_ref['state'] = comp.Xt_ref
                for (p,ref) in comp.Xt_ref.items():
                    tr_states[f'{ck+"." if ck else ""}{p}'] = ref
                    dyni['state'][f'{ck+"." if ck else ""}{p}'] = ref

            if comp.dynamic_output_parms:
                dyn_ref['output'] = comp.Yt_ref
                for p,ref in comp.Yt_ref.items():
                    tr_output[f'{ck+"." if ck else ""}{p}'] = ref
                    dyni['output'][f'{ck+"." if ck else ""}{p}'] = ref

            if comp.dynamic_input_parms:
                dyn_ref['input'] = comp.Ut_ref
                for (p,ref) in comp.Ut_ref.items():
                    tr_inputs[f'{ck+"." if ck else ""}{p}'] = ref
                    dyni['input'][f'{ck+"." if ck else ""}{p}'] = ref
                
            # for k in comp.dynamic_input_parms:
            #     p_par = comp.locate_ref(k)
            #     dyn_ref[k]= p_par
            # 
            # for k in comp.dynamic_output_parms:
            #     p_par = comp.locate_ref(k)
            #     dyn_ref[k]= p_par 
                
        #Orchestrate The Simulation
        if min_set:
            out = dict(
                tr_inputs = tr_inputs,
                tr_states = tr_states,
                tr_output = tr_output, 
                tr_sets = tr_sets,               
                ss_states = ss_states,
                ss_output = ss_output,
                #ig_source = sig_source,
                #sig_target = sig_target,
                dyn_comp = dyn_comp,
                signals=signals
            )
            return out        


        #LINK Parms With Dynamic Component Blocks For Matrix Assignment
        if by_parm:
            #sort by parm and function
            dyns = out
            dyn_st = {f'{comp}.{parm}':v for comp,state in dyns['dynamics'].items() for parm,v in state.get('state',{}).items()}
            dyn_in = {f'{comp}.{parm}':v for comp,state in dyns['dynamics'].items() for parm,v in state.get('input',{}).items()}
            dyn_out = {f'{comp}.{parm}':v for comp,state in dyns['dynamics'].items() for parm,v in state.get('output',{}).items()}
            dyn_sys = {**dyn_st,**dyn_in,**dyn_out}
            trn_int = {f'{intr}':dct['parm'] for intr,dct in dyns['transients'].items()}
            trn_rt = {f'{intr}':dct['dpdt'] for intr,dct in dyns['transients'].items()}
            #signal_tgt = {k:v['target'] for k,v in dyns['signals'].items()}
            #signal_src = {k:v['source'] for k,v in dyns['signals'].items()}
            solver_indep = {k:v['indep'] for k,v in dyns['solvers'].items()}
            solver_dep = {k:v['dep'] for k,v in dyns['solvers'].items()}

            #remove leading dot (ie, its property of self)
            conv_dict = lambda dct: {p if not p.startswith('.') else p[1:]:v for p,v in dct.items()}
            

            comp_dync = {k:d['conf'] for k,d in internal_dynamics['dynamics'].items()}
            comp_tr = {k:d['conf'] for k,d in internal_dynamics['traces'].items()}

            #unify output #TODO: io weirdness
            out = dict(
                tr_inputs = tr_inputs,
                tr_states = tr_states,
                tr_output = tr_output, 
                tr_sets = tr_sets,               
                ss_states = ss_states,
                ss_output = ss_output,
                sig_source = sig_source,
                sig_target = sig_target,                
                comps_dyn = comp_dync,
                comps_tr = comp_tr,
                dynamic_sys = conv_dict(dyn_sys),
                dynamic_state = conv_dict(dyn_st),
                dynamic_input = conv_dict(dyn_in),
                dynamic_output = conv_dict(dyn_out),
                transient_parm = conv_dict(trn_int),
                transient_rate = conv_dict(trn_rt),
                #signal_tgt = conv_dict(signal_tgt),
                #signal_src = conv_dict(signal_src),
                solver_dep = conv_dict(solver_dep),
                solver_indep = conv_dict(solver_indep),
                dyn_comp=dyn_comp              
            )
            return out

        return out        
    
    #IO Functions
    def parse_simulation_input(self,**kwargs):
        """parses the simulation input
        
        :param dt: timestep in s, required for transients
        :param endtime: when to end the simulation
        """
        # timestep
        if "dt" not in kwargs:
            raise Exception("transients require `dt` to run")
        # f"transients require timestep input `dt`"
        dt = float(kwargs.pop("dt"))

        # endtime
        if "endtime" not in kwargs:
            raise Exception("transients require `endtime` to run")

        # add data
        _trans_opts = {"dt": None, "endtime": None}
        _trans_opts["dt"] = dt

        # f"transients require `endtime` to specify "
        _trans_opts["endtime"] = endtime = float(kwargs.pop("endtime"))
        _trans_opts["Nrun"] = max(int(endtime / dt) + 1, 1)

        #TODO: expose integrator choices
        #TODO: add delay and signal & feedback options

        return _trans_opts  

    def parse_run_kwargs(self, **kwargs):
        """ensures correct input for simulation.
        :returns: first set of input for initalization, and all input dictionaries as tuple.

        """

        #use eval_kw and sys_kw to handle slot argument and sub-system arguments

        # Validate OTher Arguments By Parameter Or Comp-Recursive
        parm_args = {k: v for k, v in kwargs.items() if "." not in k}
        comp_args = {k: v for k, v in kwargs.items() if "." in k}

        # check parms
        inpossible = set.union(set(self.input_fields()),set(self.slots_attributes()))
        argdiff = set(parm_args).difference(inpossible)
        assert not argdiff, f"bad input {argdiff}"

        # check components
        comps = set([k.split(".")[0] for k in comp_args.keys()])
        compdiff = comps.difference(set(self.slots_attributes()))
        assert not compdiff, f"bad slot references {compdiff}"

        _input = {}
        test = lambda v,add: isinstance(v, (int, float, str,*add)) or v is None

        # parameters input
        for k, v in kwargs.items():
            
            #If a slot check the type is applicable
            subslot = self.check_ref_slot_type(k)
            if subslot is not None:
                #log.debug(f'found subslot {k}: {subslot}')
                addty = subslot
            else:
                addty = []

            # Ensure Its a List
            if isinstance(v, numpy.ndarray):
                v = v.tolist()

            if not isinstance(v, list):
                assert test(v,addty), f"bad values {k}:{v}"
                v = [v]
            else:
                assert all([test(vi,addty) for vi in v]), f"bad values: {k}:{v}"

            if k not in _input:
                _input[k] = v
            else:
                _input[k].extend(v)

        return _input
            