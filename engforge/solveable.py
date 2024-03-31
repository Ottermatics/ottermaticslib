import attrs,attr
import uuid
import numpy
import numpy as np
import scipy.optimize as sciopt
from contextlib import contextmanager
import copy
import datetime
import typing


# from engforge.dynamics import DynamicsMixin
from engforge.attributes import AttributeInstance
from engforge.engforge_attributes import AttributedBaseMixin
from engforge.configuration import Configuration, forge
from engforge.properties import *
from engforge.system_reference import *
from engforge.system_reference import Ref
from engforge.solver_utils import *
import collections
import itertools

SOLVER_OPTIONS = ["root", "minimize"]


class SolvableLog(LoggingMixin):
    pass


SKIP_REF = ["run_id", "converged", "name", "index"]

log = SolvableLog()

#Anonymous Update Funcs (solve scoping issue)
def _update_func(comp,eval_kw):
    def updt(*args,**kw):
        eval_kw.update(kw)
        return comp.update(comp.parent, *args, **eval_kw)
    if log.log_level <= 5:
        log.msg(f'create method| {comp.name}| {eval_kw}')
    return updt

def _post_update_func(comp,eval_kw):
    def updt(*args,**kw):
        eval_kw.update(kw)
        return comp.update(comp.parent, *args, **eval_kw)
    if log.log_level <= 5:
        log.msg(f'create post method| {comp.name}| {eval_kw}')
    return updt


    
class SolveableMixin(AttributedBaseMixin):  #'Configuration'
    """commonality for components,systems that identifies subsystems and states for solving.

    This class defines the update structure of components and systems, and the storage of internal references to the system and its components. It also provides a method to iterate over the internal components and their references.

    Importantly it defines the references to the system and its components, and the ability to set the system state from a dictionary of values across multiple objects. It also provides a method to iterate over the internal components and their references. There are several helper functions to these ends.
    """

    # TODO: add parent state storage
    parent: "Configuration"

    _prv_internal_references: dict
    _prv_internal_components: dict
    _prv_internal_systems: dict
    _prv_internal_tabs: dict
    _prv_system_references: dict

    # Update Flow
    def update(self, parent, *args, **kwargs):
        """Kwargs comes from eval_kw in solver"""
        if log.log_level <= 5:
            log.debug(f"void updating {self.__class__.__name__}.{self}")

    def post_update(self, parent, *args, **kwargs):
        """Kwargs comes from eval_kw in solver"""
        if log.log_level <= 5:
            log.debug(f"void post-updating {self.__class__.__name__}.{self}")

    def update_internal(self, eval_kw=None, ignore=None, *args, **kw):
        """update internal elements with input arguments"""
        # Solve Each Internal System
        from engforge.components import Component
        from engforge.component_collections import ComponentIter

        # Ignore
        if ignore is None:
            ignore = set()
        elif self in ignore:
            return

        self.update(self.parent, *args, **kw)
        self.debug(f"updating internal {self.__class__.__name__}.{self}")
        for key, comp in self.internal_configurations(False).items():
            if ignore is not None and comp in ignore:
                continue

            # provide add eval_kw
            if eval_kw and key in eval_kw:
                eval_kw_comp = eval_kw[key]
            else:
                eval_kw_comp = {}

            # comp update cycle
            self.debug(f"updating {key} {comp.__class__.__name__}.{comp}")
            if isinstance(comp, ComponentIter):
                comp.update(self, **eval_kw_comp)
            elif isinstance(comp, (SolveableMixin, Component)):
                comp.update(self, **eval_kw_comp)
                comp.update_internal(eval_kw=eval_kw_comp, ignore=ignore)
        ignore.add(self)  # add self to ignore list

    def post_update_internal(self, eval_kw=None, ignore=None, *args, **kw):
        """Post update all internal components"""
        # Post Update Self
        from engforge.components import Component
        from engforge.component_collections import ComponentIter

        # Ignore
        if ignore is None:
            ignore = set()
        elif self in ignore:
            return

        self.post_update(self.parent, *args, **kw)
        self.debug(f"post updating internal {self.__class__.__name__}.{self}")
        for key, comp in self.internal_configurations(False).items():
            if ignore is not None and comp in ignore:
                continue

            # provide add eval_kw
            if eval_kw and key in eval_kw:
                eval_kw_comp = eval_kw[key]
            else:
                eval_kw_comp = {}

            self.debug(f"post updating {key} {comp.__class__.__name__}.{comp}")
            if isinstance(comp, ComponentIter):
                comp.post_update(self, **eval_kw_comp)
            elif isinstance(comp, (SolveableMixin, Component)):
                comp.post_update(self, **eval_kw_comp)
                comp.post_update_internal(eval_kw=eval_kw_comp, ignore=ignore)
        ignore.add(self)

    def collect_update_refs(self,eval_kw=None,ignore=None):
        """checks all methods and creates ref's to execute them later"""
        from engforge.eng.costs import CostModel

        updt_refs = {}
        from engforge.components import Component
        from engforge.component_collections import ComponentIter

        # Ignore
        if ignore is None:
            ignore = set()
        elif self in ignore:
            return
        
        for key, comp in self.internal_configurations(False).items():
            if ignore is not None and comp in ignore:
                continue

            if not isinstance(comp,SolveableMixin):
                continue            

            #provide add eval_kw
            if eval_kw and key in eval_kw:
                eval_kw_comp = eval_kw[key]
            else:
                eval_kw_comp = {}
            ekw = eval_kw_comp

            #Add if its a unique update method (not the passthrough)
            if comp.__class__.update != SolveableMixin.update:
                ref = Ref(comp,_update_func(comp,ekw))
                updt_refs[key] =  ref

            if isinstance(comp,CostModel):
                ref = Ref(comp,lambda *a,**kw: comp.update_dflt_costs() )
                updt_refs[key+'._cost_model_'] =  ref

        ignore.add(self)
        return updt_refs

    def collect_post_update_refs(self,eval_kw=None,ignore=None):
        """checks all methods and creates ref's to execute them later"""
        updt_refs = {}
        from engforge.components import Component
        from engforge.component_collections import ComponentIter

        # Ignore
        if ignore is None:
            ignore = set()
        elif self in ignore:
            return

        for key, comp in self.internal_configurations(False).items():
            if ignore is not None and comp in ignore:
                continue

            if not isinstance(comp,SolveableMixin):
                continue            

            # provide add eval_kw
            if eval_kw and key in eval_kw:
                eval_kw_comp = eval_kw[key]
            else:
                eval_kw_comp = {}
            
            ekw = eval_kw_comp
                
            #Add if its a unique update method (not the passthrough)
            if comp.__class__.post_update != SolveableMixin.post_update:
                ref = Ref(comp,_post_update_func(comp,ekw))
                updt_refs[key] =  ref

        ignore.add(self)
        return updt_refs

    # internals caching
    # instance attributes

    @instance_cached
    def signals(self):
        """this is just a record of signals from this solvable. dont use this to get the signals, use high level signals strategy #TODO: add signals strategy"""
        return {k: getattr(self, k) for k in self.signals_attributes()}

    @instance_cached
    def solvers(self):
        """this is just a record of any solvable attribute. dont use this to get the attribute, use another strategy"""        
        return {k: getattr(self, k) for k in self.solvers_attributes()}

    @instance_cached(allow_set=True)
    def transients(self):
        """this is just a record of any transient attribute. dont use this to get the transient, use another strategy"""   
        return {k: getattr(self, k) for k in self.transients_attributes()}

    def internal_components(self, recache=False) -> dict:
        """get all the internal components"""
        if recache == False and hasattr(self, "_prv_internal_components"):
            return self._prv_internal_components
        from engforge.components import Component

        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, Component)}
        self._prv_internal_components = o
        return o

    def internal_systems(self, recache=False) -> dict:
        """get all the internal components"""
        if recache == False and hasattr(self, "_prv_internal_systems"):
            return self._prv_internal_systems
        from engforge.system import System

        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, System)}
        self._prv_internal_systems = o
        return o

    def internal_tabulations(self, recache=False) -> dict:
        """get all the internal tabulations"""
        from engforge.tabulation import TabulationMixin

        if recache == False and hasattr(self, "_prv_internal_tabs"):
            return self._prv_internal_tabs

        o = {k: getattr(self, k) for k in self.slots_attributes()}
        o = {k: v for k, v in o.items() if isinstance(v, TabulationMixin)}
        self._prv_internal_tabs = o
        return o

    # recursive references
    @instance_cached
    def iterable_components(self) -> dict:
        """Finds ComponentIter internal_components that are not 'wide'"""
        from engforge.component_collections import ComponentIter

        return {
            k: v
            for k, v in self.internal_components().items()
            if isinstance(v, ComponentIter) and not v.wide
        }

    def internal_references(self, recache=False) -> dict:
        """get references to all internal attributes and values"""
        if recache == False and hasattr(self, "_prv_internal_references"):
            return self._prv_internal_references

        out = self._gather_references()
        self._prv_internal_references = out
        return out

    def _gather_references(self) -> dict:
        out = {}
        out["attributes"] = at = {}
        out["properties"] = pr = {}

        for key in self.system_properties_classdef():
            pr[key] = Ref(self, key, True, False)

        for key in self.input_fields():
            at[key] = Ref(self, key, False, True)

        return out

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

    # @instance_cached 
    @property
    def comp_references(self):
        """A cached set of recursive references to any slot component
        #TODO: work on caching, concern with iterators 
        #FIXME: by instance recache on iterative component change or other signals
        """
        out = {}
        for key, lvl, comp in self.go_through_configurations(parent_level=1):
            if not isinstance(comp, SolveableMixin):
                continue
            out[key] = comp
        return out

    # Run & Input
    def _iterate_input_matrix(
        self,
        method,
        cb=None,
        sequence: list = None,
        eval_kw: dict = None,
        sys_kw: dict = None,
        force_solve=False,
        return_results=False,
        method_kw: dict = None,
        **kwargs,
    ):
        """applies a permutation of input vars for vars. runs the system instance by applying input to the system and its slot-components, ensuring that the targeted attributes actualy exist.

        :param revert: will reset the values of X that were recorded at the beginning of the run.
        :param cb: a callback function that takes the system as an argument cb(system)
        :param sequence: a list of dictionaries that should be run in order per the outer-product of kwargs
        :param eval_kw: a dictionary of keyword arguments to pass to the eval function of each component by their name and a set of keyword args. Use this to set values in the component that are not inputs to the system. No iteration occurs upon these values, they are static and irrevertable
        :param sys_kw: a dictionary of keyword arguments to pass to the eval function of each system by their name and a set of keyword args. Use this to set values in the component that are not inputs to the system. No iteration occurs upon these values, they are static and irrevertable
        :param kwargs: inputs are run on a product basis asusming they correspond to actual scoped vars (system.var or system.slot.var)


        :returns: system or list of systems. If transient a set of systems that have been run with permutations of the input, otherwise a single system with all permutations input run
        """
        from engforge.system import System

        self.debug(
            f"running [SOLVER].{method} {self.identity} with input {kwargs}"
        )

        # TODO: allow setting sub-component vars with `slot1.slot2.attrs`. Ensure checking slots exist, and attrs do as well.

        # create iterable null for sequence
        if sequence is None or not sequence:
            sequence = [{}]

        if method_kw is None:
            method_kw = {}

        # Create Keys List
        sequence_keys = set()
        for seq in sequence:
            sequence_keys = sequence_keys.union(set(seq.keys()))

        # RUN when not solved, or anything changed, or arguments or if forced
        if force_solve or not self.solved or self.anything_changed or kwargs:
            _input = self.parse_run_kwargs(**kwargs)

            output = {}
            inputs = {}
            result = {"output": output, "input_sets": inputs}

            # if revert:
            #     # revert all internal components too with `system_state` and set_system_state(**x,comp.x...)
            #     sys_refs = self.get_system_input_refs(all=True)
            #     revert_x = Ref.refset_get(sys_refs)

            # prep references for keys
            refs = {}
            for k, v in _input.items():
                refs[k] = self.locate_ref(k)
            for k in sequence_keys:
                refs[k] = self.locate_ref(k)

            # Pre Run Callback
            self.pre_run_callback(eval_kw=eval_kw, sys_kw=sys_kw, **kwargs)

            # Premute the input as per SS or Transient Logic
            ingrp = list(_input.values())
            keys = list(_input.keys())

            # Iterate over components (they are assigned to the system by call)
            for itercomp in self._iterate_components():
                # Iterate over inputs
                for vars in itertools.product(*ingrp):
                    # Set the reference aliases
                    cur = {k: v for k, v in zip(keys, vars)}

                    # Iterate over Sequence (or run once)
                    for seq in sequence:
                        #apply sequence values
                        icur = cur.copy()
                        #apply sequence values
                        if seq:
                            icur.update(**seq)

                        # Run The Method with inputs provisioned
                        #TODO: wrap with ProblemExec 
                        out = method(
                            refs, icur, eval_kw, sys_kw, cb=cb, **method_kw
                        )

                        if return_results:
                            # store the output
                            output[max(output) + 1 if output else 0] = out
                            # store the input
                            inputs[max(inputs) + 1 if inputs else 0] = icur

            # nice
            #TODO: wrap this with context manager
            self._solved = True

            # Pre Run Callback with current state
            self.post_run_callback(eval_kw=eval_kw, sys_kw=sys_kw, **kwargs)

            if return_results:
                return result

        elif not self.anything_changed:
            self.warning(f"nothing changed, not running {self.identity}")

            return
        elif self.solved:
            raise Exception("Analysis Already Solved")

    # IO Functions
    def parse_simulation_input(self, **kwargs):
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

        # TODO: expose integrator choices
        # TODO: add delay and signal & feedback options

        return _trans_opts

    def parse_run_kwargs(self, **kwargs):
        """ensures correct input for simulation.
        :returns: first set of input for initalization, and all input dictionaries as tuple.

        """
        # Validate OTher Arguments By Parameter Or Comp-Recursive
        var_args = {k: v for k, v in kwargs.items() if "." not in k}
        comp_args = {k: v for k, v in kwargs.items() if "." in k}

        # check vars
        inpossible = set.union(
            set(self.input_fields()), set(self.slots_attributes())
        )
        argdiff = set(var_args).difference(inpossible)
        assert not argdiff, f"bad input {argdiff}"

        # check components
        comps = set([k.split(".")[0] for k in comp_args.keys()])
        compdiff = comps.difference(set(self.slots_attributes()))
        assert not compdiff, f"bad slot references {compdiff}"

        _input = {}
        test = (
            lambda v, add: isinstance(v, (int, float, str, *add)) or v is None
        )

        # vars input
        for k, v in kwargs.items():
            # If a slot check the type is applicable
            subslot = self.check_ref_slot_type(k)
            if subslot is not None:
                # log.debug(f'found subslot {k}: {subslot}')
                addty = subslot
            else:
                addty = []

            # Ensure Its a List
            if isinstance(v, numpy.ndarray):
                v = v.tolist()

            if not isinstance(v, list):
                assert test(v, addty), f"bad values {k}:{v}"
                v = [v]
            else:
                assert all(
                    [test(vi, addty) for vi in v]
                ), f"bad values: {k}:{v}"

            if k not in _input:
                _input[k] = v
            else:
                _input[k].extend(v)

        return _input

    # REFERENCE FUNCTIONS
    # Location Funcitons
    @classmethod
    def locate(cls, key, fail=True) -> type:
        """:returns: the class or attribute by key if its in this system class or a subcomponent. If nothing is found raise an error"""
        # Nested
        log.msg(f"locating {cls.__name__} | key: {key}")
        val = None

        if "." in key:
            args = key.split(".")
            comp, sub = args[0], ".".join(args[1:])
            assert comp in cls.slots_attributes(), f"invalid {comp} in {key}"
            comp_cls = cls.slots_attributes()[comp].type.accepted[0]
            val = comp_cls.locate(sub, fail=True)

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

    def locate_ref(self, key, fail=True, **kw):
        """Pass a string of a relative var or property on this system or pass a callable to get a reference to a function. If the key has a `.` in it the comp the lowest level component will be returned, unless a callable is passed in which case this component will be used or the `comp` passed in the kw will be used.
        :param key: the key to locate, or a callable to be used as a reference
        :param comp: the component to use if a callable is passed
        :returns: the instance assigned to this system. If the key has a `.` in it the comp the lowest level component will be returned
        """

        log.msg(f"locating {self.identity} | key: {key}")
        val = None

        #Handle callable key override
        if callable(key):
            comp = kw.pop("comp", self)
            func = copy.copy(key)
            return Ref(comp, func, **kw)
        else:
            assert 'comp' not in kw, f"comp kwarg not allowed with string key {key}"

        if "." in key:
            args = key.split(".")
            comp, sub = args[0], ".".join(args[1:])
            assert comp in self.slots_attributes(), f"invalid {comp} in {key}"
            # comp_cls = cls.slots_attributes()[comp].type.accepted[0]
            comp = getattr(self, comp)
            if "." not in key:
                return Ref(comp, sub, **kw)
            return comp.locate_ref(sub, fail=fail, **kw)

        elif key in self.input_fields():
            # val= cls.input_fields()[key]
            return Ref(self, key, **kw)

        elif key in self.system_properties_classdef():
            # val= cls.system_properties_classdef()[key]
            return Ref(self, key, **kw)

        elif (
            key in self.internal_configurations()
            or key in self.slots_attributes()
        ):
            return Ref(self, key, **kw)

        # Fail on comand but otherwise return val
        if val is None:
            if fail:
                raise Exception(f"key {key} not found")
            return None
        return val

    # Reference Caching
    def system_references(self, recache=False):
        """gather a list of references to attributes and"""
        if recache == False and hasattr(self, "_prv_system_references"):
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
    
    def collect_comp_refs(self,conf:"Configuration"=None,**kw):
        """collects all the references for the system grouped by component"""    
        if conf is None:
            conf = self
        comp_dict = {}
        attr_dict = {}
        cls_dict = {}
        out = {'comps':comp_dict,'attrs':attr_dict,'type':cls_dict}
        for key, lvl, conf in self.go_through_configurations(**kw):
            comp_dict[key] = conf
            attr_dict[key] = conf.collect_inst_attributes()

        return out
    
    
    def collect_solver_refs(self,conf:"Configuration"=None,conv=None,check_atr_f=None,check_kw=None,**kw):
        """collects all the references for the system grouped by function and prepended with the system key"""    
        from engforge.attributes import ATTR_BASE
        from engforge.engforge_attributes import AttributedBaseMixin
        
        confobj = conf
        if confobj is None:
            confobj = self

        comp_dict = {}
        attr_dict = {}
        cls_dict = {}
        skipped = {}

        conv = conv if conv is not None else None

        out = {'comps':comp_dict,'attrs':attr_dict,'type':cls_dict,'skipped':skipped}
        
        #Go through all components
        for key, lvl, conf in confobj.go_through_configurations(**kw):

            if conf is None:
                continue
            
            if hasattr(conf,'_solver_override') and conf._solver_override:
                continue

            #Get attributes & attribute instances
            atrs = conf.collect_inst_attributes()
            rawattr = conf.collect_inst_attributes(handle_inst=False)
            key = f'{key}.' if key else ''
            comp_dict[key] = conf
            #Gather attribute heirarchy and make key.var the dictionary entry
            for atype,aval in atrs.items():
                
                ck_type = rawattr[atype]
                if atype not in cls_dict:
                    cls_dict[atype] = {}

                if isinstance(aval,dict) and aval:
                                       
                    for k,pre,val in ATTR_BASE.unpack_atrs(aval,atype):

                        #No Room For Components (SLOTS feature)
                        if isinstance(val,(AttributedBaseMixin,ATTR_BASE)):
                            if conf.log_level <= 5:
                                conf.debug(f'skipping {val}')
                            continue   

                        if val is None:
                            continue

                        if conf.log_level <= 5:
                            conf.msg(f'')
                            conf.msg(f'got val: {k} {pre} {val}')  

                        slv_type = None
                        pre_var = pre.split('.')[-1]
                        if hasattr(conf,pre_var):
                            _var = getattr(conf,pre_var)
                            if isinstance(_var,AttributeInstance):
                                slv_type = _var
                            conf.msg(f'slv type: {conf.classname}.{pre_var} -> {_var}')

                        val_type = ck_type[pre_var]

                        #Otherwise assign the data from last var and the compoenent name
                        var_name = pre_var #prer alias
                        if slv_type:
                            var_name = slv_type.get_alias(pre)
                        
                        #Keep reference to the original type and name
                        scope_name = f'{key}{var_name}'
                        cls_dict[atype][scope_name] = val_type
                        
                        if conf.log_level <= 5:
                            conf.msg(f'rec: {var_name} {k} {pre} {val} {slv_type}')

                        #Check to skip this item
                        #keep references even if null
                        pre = f'{atype}.{k}' #pre switch
                        if pre not in attr_dict:
                            attr_dict[pre] = {}
                        
                        if isinstance(val,Ref) and val.allow_set:
                            #its a var, skip it if it's already been skipped
                            current_skipped = [set(v) for v in skipped.values()]
                            if current_skipped and val.key in set.union(*current_skipped):
                                continue
                        
                        #Perform the check
                        if check_atr_f and not check_atr_f(pre,scope_name,val_type,check_kw):
                            if conf.log_level <= 5:
                                conf.msg(f'chk skip {scope_name} {k} {pre} {val}')
                            if pre not in skipped:
                                skipped[pre] = []

                            if isinstance(val,Ref) and val.allow_set:
                                skipped[pre].append(f'{key}{val.key}') #its a var
                            else:
                                #not objective or settable, must be a obj/cond
                                skipped[pre].append(scope_name)
                            continue

                        #if the value is a dictionary, unpack it with comp key
                        if val:
                            attr_dict[pre].update({scope_name:val})
                        else:
                            if attr_dict[pre]:
                                continue #keep it!
                            else:
                                attr_dict[pre] = {} #reset it

                elif atype not in attr_dict or not attr_dict[atype]:
                   #print(f'unpacking {atype} {aval}')
                   attr_dict[atype] = {}
        
        #Dynamic Variables Add, following the skipped items
        dyn_refs = self.collect_dynamic_refs(confobj)
        #house keeping to organize special returns 
        #TODO: generalize this with a function to update the attr dict serach result
        out['dynamic_comps'] = dyn_refs.pop('dynamic_comps',{})
        

        if check_atr_f or any([v for v in skipped.values()]):
            #print('skipped',skipped)
            skipd = set().union(*(set(v) for v in skipped.values()))
            for pre,refs in dyn_refs.items():
                
                if pre not in attr_dict:
                    attr_dict[pre] = {}

                for var,ref in refs.items():
                    
                    is_ref = isinstance(ref,Ref) and ref.allow_set
                    if not is_ref:
                        conf.info(f'bad refs skip {pre,ref,val_type,skipd}')
                        attr_dict[pre].update(**{var:ref})
                        continue

                    scoped_name = f'{key}{var}'
                    val_setskip = ( is_ref and ref.key in skipd)
                    val_type = None
                    
                    if scoped_name in skipd:
                        conf.msg(f'dynvar skip {pre,ref.key,val_type,skipd}')
                        continue

                    elif val_setskip:
                        conf.msg(f'dynvar r-skip {pre,ref.key,val_type,skipd}')
                        continue
                        
                    elif check_atr_f and check_atr_f(pre,ref.key,val_type,check_kw):
                        conf.msg(f'dynvar add {pre,ref.key,val_type,skipd}')
                        attr_dict[pre].update(**{var:ref})
                    else:
                        conf.msg(f'dynvar endskip {pre,ref.key,val_type,skipd}')
                        if check_atr_f: #then it didn't checkout
                            skipped[pre].append(scope_name)

        else:
            #There's no checks to be done, just add these
            attr_dict.update(**dyn_refs)                   

        return out

    #Dynamics info refs
    def collect_dynamic_refs(
        self, conf: "ConfigurationMixin" = None, **kw
    ) -> dict:
        """collects the dynamics of the systems
        1. Time.integrate
        2. Dynamic Instances
        """
        if conf is None:
            conf = self

        dynamics = {'dynamics.state':{},'dynamics.input':{},'dynamics.rate':{},'dynamics.output':{},'dynamic_comps':{}}

        for key, lvl, conf in conf.go_through_configurations(**kw):
            # FIXME: add a check for the dynamics mixin, that isn't hacky
            # BUG: importing dynamicsmixin resolves as different class in different modules, weird
            if "dynamicsmixin" not in str(conf.__class__.mro()).lower():
                continue
            sval = f'{key}.' if key else ''
            scope = lambda d: {f'{sval}{k}':v for k,v in d.items()}
            dynamics['dynamics.state'].update(scope(conf.Xt_ref))
            dynamics['dynamics.input'].update(scope(conf.Ut_ref))
            dynamics['dynamics.output'].update(scope(conf.Yt_ref))
            dynamics['dynamics.rate'].update(scope(conf.dXtdt_ref))
            dynamics['dynamic_comps'][key] = conf

        return dynamics


    def get_system_input_refs(
        self,
        strings=False,
        numeric=True,
        misc=False,
        all=False,
        boolean=False,
        **kw,
    ) -> dict:
        """
        Get the references to system input based on the specified criteria.

        :param strings: Include system properties of string type.
        :param numeric: Include system properties of numeric type (float, int).
        :param misc: Include system properties of miscellaneous type.
        :param all: Include all system properties regardless of type.
        :param boolean: Include system properties of boolean type.
        :param kw: Additional keyword arguments passed to recursive config loop
        :return: A dictionary of system property references.
        :rtype: dict
        """
        refs = {}
        for ckey, lvl, comp in self.go_through_configurations(**kw):
            if comp is None:
                continue
            for p, atr in comp.input_fields().items():
                if p in SKIP_REF and not all:
                    continue
                if all:
                    refs[(f"{ckey}." if ckey else "") + p] = Ref(
                        comp, p, False, True
                    )
                    continue
                elif atr.type:
                    ty = atr.type
                    if issubclass(ty, (bool)):
                        if not boolean:
                            continue  # prevent catch at int type
                        refs[(f"{ckey}." if ckey else "") + p] = Ref(
                            comp, p, True, False
                        )
                    elif issubclass(ty, (float, int)) and numeric:
                        refs[(f"{ckey}." if ckey else "") + p] = Ref(
                            comp, p, False, True
                        )
                    elif issubclass(ty, (str)) and strings:
                        refs[(f"{ckey}." if ckey else "") + p] = Ref(
                            comp, p, False, True
                        )
                    elif misc:
                        refs[(f"{ckey}." if ckey else "") + p] = Ref(
                            comp, p, False, True
                        )

        return refs

    def get_system_property_refs(
        self,
        strings=False,
        numeric=True,
        misc=False,
        all=False,
        boolean=False,
        **kw,
    ):
        """
        Get the references to system properties based on the specified criteria.

        :param strings: Include system properties of string type.
        :param numeric: Include system properties of numeric type (float, int).
        :param misc: Include system properties of miscellaneous type.
        :param all: Include all system properties regardless of type.
        :param boolean: Include system properties of boolean type.
        :param kw: Additional keyword arguments passed to recursive config loop
        :return: A dictionary of system property references.
        :rtype: dict
        """
        refs = {}
        for ckey, lvl, comp in self.go_through_configurations(**kw):
            if not isinstance(comp, SolveableMixin):
                continue
            for p, atr in comp.system_properties_classdef().items():
                if p in SKIP_REF and not all:
                    continue
                ty = atr.return_type
                if all:
                    refs[(f"{ckey}." if ckey else "") + p] = Ref(
                        comp, p, True, False
                    )
                    continue
                if issubclass(ty, bool):
                    if not boolean:
                        continue  # prevent catch at int type
                    refs[(f"{ckey}." if ckey else "") + p] = Ref(
                        comp, p, True, False
                    )
                elif issubclass(ty, (float, int)) and numeric:
                    refs[(f"{ckey}." if ckey else "") + p] = Ref(
                        comp, p, True, False
                    )
                elif issubclass(ty, str) and strings:
                    refs[(f"{ckey}." if ckey else "") + p] = Ref(
                        comp, p, True, False
                    )
                elif misc:
                    refs[(f"{ckey}." if ckey else "") + p] = Ref(
                        comp, p, True, False
                    )

        return refs

