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
        pass

    def post_update(self, parent, *args, **kwargs):
        pass

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

    def gather_update_refs(self,eval_kw=None,ignore=None,*args,**kw):
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

            if comp.__class__.update != SolveableMixin.update:
                f = lambda : comp.update(comp.parent, *args, **kw)
                f.__name__ = f'{comp.name}_update'
                ref = Ref(comp,f)
                updt_refs[key] =  ref

        ignore.add(self)
        return updt_refs

    def gather_post_update_refs(self,eval_kw=None,ignore=None,*args,**kw):
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

            if comp.__class__.post_update != SolveableMixin.post_update:
                f = lambda : comp.update(comp.parent, *args, **kw)
                f.__name__ = f'{comp.name}_post_update'
                ref = Ref(comp,f)
                updt_refs[key] =  ref                
        ignore.add(self)
        return updt_refs

    # internals caching
    # instance attributes
    #TODO: make a global signals call
    @instance_cached
    def signals(self):
        return {k: getattr(self, k) for k in self.signals_attributes()}

    @instance_cached
    def solvers(self):
        return {k: getattr(self, k) for k in self.solvers_attributes()}

    @instance_cached(allow_set=True)
    def transients(self):
        # TODO: add dynamics internals
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
        """
        out = {}
        for key, lvl, comp in self.go_through_configurations(parent_level=1):
            if not isinstance(comp, SolveableMixin):
                continue
            out[key] = comp
        return out

    # Run & Input
    # General method to distribute input to internal components
    def _parse_default(self,key,defaults,input_dict):
        """splits strings or lists and returns a list of options for the key, if nothing found returns None if fail set to True raises an exception, otherwise returns the default value"""
        if key in input_dict:
            option = input_dict.pop(key)
            #print(f'removing option {key} {option}')
            if isinstance(option,(int,float,bool)):
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
                    self.warning(f'bad option {option} for {key}| {input_dict}')
                    
            return option,False
        elif key in defaults:
            return defaults[key],True
        elif input_dict.get('_fail',True):
            raise Exception(f'no option for {key}')
        return None,None
        

    def _rmv_extra_kws(self,kwargs,_check_keys:dict,_fail=True):
        """extracts the combo input from the kwargs"""
        # extract combo input
        if not _check_keys:
            return {}
        _check_keys = _check_keys.copy()
        #TODO: allow extended check_keys / defaults to be passed in, now every value in check_keys has a default
        cur_in = kwargs
        cur_in['_fail'] = _fail
        combos = {}
        for p,dflt in _check_keys.items():
            val,is_dflt = self._parse_default(p,_check_keys,cur_in)
            combos[p] = val
            cur_in.pop(p,None)

        comboos = dict(list(filter(lambda kv: kv[1] is not None or kv[0] in _check_keys, combos.items())))
        #print(f'got {combos} -> {comboos} from {kwargs} with {_check_keys}')
        return combos

    def _iterate_input_matrix(
        self,
        method,
        revert=True,
        cb=None,
        sequence: list = None,
        eval_kw: dict = None,
        sys_kw: dict = None,
        force_solve=False,
        return_results=False,
        method_kw: dict = None,
        **kwargs,
    ):
        """applies a permutation of input parameters for parameters. runs the system instance by applying input to the system and its slot-components, ensuring that the targeted attributes actualy exist.

        :param revert: will reset the values of X that were recorded at the beginning of the run.
        :param cb: a callback function that takes the system as an argument cb(system)
        :param sequence: a list of dictionaries that should be run in order per the outer-product of kwargs
        :param eval_kw: a dictionary of keyword arguments to pass to the eval function of each component by their name and a set of keyword args. Use this to set values in the component that are not inputs to the system. No iteration occurs upon these values, they are static and irrevertable
        :param sys_kw: a dictionary of keyword arguments to pass to the eval function of each system by their name and a set of keyword args. Use this to set values in the component that are not inputs to the system. No iteration occurs upon these values, they are static and irrevertable
        :param kwargs: inputs are run on a product basis asusming they correspond to actual scoped parameters (system.parm or system.slot.parm)


        :returns: system or list of systems. If transient a set of systems that have been run with permutations of the input, otherwise a single system with all permutations input run
        """
        from engforge.system import System

        self.debug(
            f"running [SOLVER].{method} {self.identity} with input {kwargs}"
        )

        # TODO: allow setting sub-component parameters with `slot1.slot2.attrs`. Ensure checking slots exist, and attrs do as well.

        # Recache system references
        if isinstance(self, System):
            self.system_references(recache=True)

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

            if revert:
                # revert all internal components too with `system_state` and set_system_state(**x,comp.x...)
                sys_refs = self.get_system_input_refs(all=True)
                revert_x = Ref.refset_get(sys_refs)

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
                for parms in itertools.product(*ingrp):
                    # Set the reference aliases
                    cur = {k: v for k, v in zip(keys, parms)}
                    # Iterate over Sequence (or run once)
                    for seq in sequence:
                        # apply sequenc evalues
                        icur = cur.copy()
                        if seq:
                            icur.update(**seq)

                        # Run The Method with inputs provisioned
                        out = method(
                            refs, icur, eval_kw, sys_kw, cb=cb, **method_kw
                        )

                        if return_results:
                            # store the output
                            output[max(output) + 1 if output else 0] = out
                            # store the input
                            inputs[max(inputs) + 1 if inputs else 0] = icur

            # nice
            self._solved = True

            # Pre Run Callback with current state
            self.post_run_callback(eval_kw=eval_kw, sys_kw=sys_kw, **kwargs)

            # pre-revert by default
            if revert and revert_x:
                Ref.refset_input(sys_refs, revert_x)

            # reapply solved state
            self._solved = True

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

        # use eval_kw and sys_kw to handle slot argument and sub-system arguments. use gather_kw to gather extra parameters adhoc.

        # Validate OTher Arguments By Parameter Or Comp-Recursive
        parm_args = {k: v for k, v in kwargs.items() if "." not in k}
        comp_args = {k: v for k, v in kwargs.items() if "." in k}

        # check parms
        inpossible = set.union(
            set(self.input_fields()), set(self.slots_attributes())
        )
        argdiff = set(parm_args).difference(inpossible)
        assert not argdiff, f"bad input {argdiff}"

        # check components
        comps = set([k.split(".")[0] for k in comp_args.keys()])
        compdiff = comps.difference(set(self.slots_attributes()))
        assert not compdiff, f"bad slot references {compdiff}"

        _input = {}
        test = (
            lambda v, add: isinstance(v, (int, float, str, *add)) or v is None
        )

        # parameters input
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
        log.debug(f"locating {cls.__name__} | key: {key}")
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
        """Pass a string of a relative parm or property on this system or pass a callable to get a reference to a function. If the key has a `.` in it the comp the lowest level component will be returned, unless a callable is passed in which case this component will be used or the `comp` passed in the kw will be used.
        :param key: the key to locate, or a callable to be used as a reference
        :param comp: the component to use if a callable is passed
        :returns: the instance assigned to this system. If the key has a `.` in it the comp the lowest level component will be returned
        """

        log.debug(f"locating {self.identity} | key: {key}")
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
    def system_references(self, recache=False, child_only=True):
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

        state_parms = ['solver.var','dynamics.state','time.parm']
        comp_dict = {}
        attr_dict = {}
        cls_dict = {}
        skipped = {}
        updates = {}

        conv = conv if conv is not None else None

        out = {'comps':comp_dict,'attrs':attr_dict,'type':cls_dict,'skipped':skipped}
        
        #Go through all components
        for key, lvl, conf in confobj.go_through_configurations():
            
            if hasattr(conf,'solver_override') and conf.solver_override:
                continue

            atrs = conf.collect_inst_attributes()
            rawattr = conf.collect_inst_attributes(handle_inst=False)
            key = f'{key}.' if key else ''
            comp_dict[key] = conf
            #Gather attribute heirarchy and make key.parm the dictionary entry
            for atype,aval in atrs.items():
                
                ck_type = rawattr[atype]
                if atype not in cls_dict:
                    cls_dict[atype] = {}

                if isinstance(aval,dict) and aval:
                                       
                    for k,pre,val in ATTR_BASE.unpack_atrs(aval,atype):

                        #No Room For Components (SLOTS feature)
                        if isinstance(val,(AttributedBaseMixin,ATTR_BASE)):
                            conf.msg(f'skipping {val}')
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

                        #Otherwise assign the data from last parm and the compoenent name
                        parm_name = pre_var #prer alias
                        if slv_type:
                            parm_name = slv_type.get_alias(pre)

                        scope_name = f'{key}{parm_name}'
                        cls_dict[atype][scope_name] = val_type
                        
                        if conf.log_level <= 5:
                            conf.msg(f'rec: {parm_name} {k} {pre} {val} {slv_type}')

                        #Check to skip this item
                        #self.info(f'check {pre} {parm_name} {k} {val}')

                        pre = f'{atype}.{k}' #pre switch
                        if pre not in attr_dict:
                            attr_dict[pre] = {}
                        
                        if isinstance(val,Ref) and val.allow_set:
                            #its a parameter, skip it if it's already been skipped
                            current_skipped = [set(v) for v in skipped.values()]
                            if current_skipped and val.key in set.union(*current_skipped):
                                continue

                        if check_atr_f and not check_atr_f(pre,scope_name,val_type,check_kw):
                            conf.debug(f'skip {scope_name} {k} {pre} {val}')
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
        if check_atr_f and any([v for v in skipped.values()]):
            #print('skipped',skipped)
            skipd = set().union(*(set(v) for v in skipped.values()))
            for pre,refs in self.collect_dynamic_refs(conf).items():
                if pre not in attr_dict:
                    attr_dict[pre] = {}
                for parm,ref in refs.items():
                    
                    val_type = getattr(ref.comp,parm)
                    if f'{key}{parm}' in skipd:
                        conf.debug('dynvar skip',pre,ref.key,val_type,skipd)
                        continue

                    elif isinstance(ref,Ref) and ref.allow_set:
                        if ref.key in skipd:
                            conf.debug('dynvar skip',pre,ref.key,val_type,skipd)
                            continue
                    
                        
                    elif check_atr_f(pre,ref.key,val_type,check_kw):
                        conf.debug('dynvar add',pre,ref.key,val_type,skipd)
                        attr_dict[pre].update(**{parm:ref})
                    else:
                        conf.debug('dynvar skip',pre,ref.key,val_type,skipd)

        else:
            attr_dict.update(**self.collect_dynamic_refs(conf))                   

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

        dynamics = {'dynamics.state':{},'dynamics.input':{},'dynamics.rate':{},'dynamics.output':{}}

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
        :type strings: bool, optional
        :param numeric: Include system properties of numeric type (float, int).
        :type numeric: bool, optional
        :param misc: Include system properties of miscellaneous type.
        :type misc: bool, optional
        :param all: Include all system properties regardless of type.
        :type all: bool, optional
        :param boolean: Include system properties of boolean type.
        :type boolean: bool, optional
        :param kw: Additional keyword arguments passed to recursive config loop
        :type kw: dict, optional
        :return: A dictionary of system property references.
        :rtype: dict
        """
        refs = {}
        for ckey, lvl, comp in self.go_through_configurations(**kw):
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
        :type strings: bool, optional
        :param numeric: Include system properties of numeric type (float, int).
        :type numeric: bool, optional
        :param misc: Include system properties of miscellaneous type.
        :type misc: bool, optional
        :param all: Include all system properties regardless of type.
        :type all: bool, optional
        :param boolean: Include system properties of boolean type.
        :type boolean: bool, optional
        :param kw: Additional keyword arguments passed to recursive config loop
        :type kw: dict, optional
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

