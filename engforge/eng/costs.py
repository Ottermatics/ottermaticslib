
"""Defines a CostModel & Economics Component that define & orchestrate cost accounting respectively.

CostModels can have a `cost_per_item` and additionally calculate a `cumulative_cost` from internally defined `CostModel`s.

CostModel's can have cost_property's which detail how and when a cost should be applied & grouped. By default each CostModel has a `cost_per_item` which is reflected in `item_cost` cost_property set on the `initial` term as a `unit` category. Multiple categories of cost are also able to be set on cost_properties as follows

```
@forge
class Widget(Component,CostModel):

    @cost_property(mode='initial',category='capex,manufacturing')
    def cost_of_XYZ(self):
        return ...
```

Economics models sum CostModel.cost_properties recursively on the parent they are defined. Economics computes the grouped category costs for each item recursively as well as summary properties like annualized values and levalized cost. Economic output is determined by a `fixed_output` or overriding `calculate_production(self,parent)` to dynamically calculate changing economics based on factors in the parent.

The economics term_length applies costs over the term, using the `cost_property.mode` to determine at which terms a cost should be applied.

@forge
class Parent(System,CostModel)

    econ = SLOT.define(Economics) #will calculate parent costs as well
    cost = SLOT.define(Widget) #slots automatically set to none if no input provided

Parent(econ=Economics(term_length=25,discount_rate=0.05,fixed_output=1000))


"""

from engforge.components import Component
from engforge.configuration import forge,Configuration
from engforge.tabulation import TabulationMixin, system_property, Ref
from engforge.properties import instance_cached,solver_cached,cached_system_property
from engforge.logging import LoggingMixin
from engforge.component_collections import ComponentIter
import typing
import attrs
import uuid
import numpy
import collections
import pandas

class CostLog(LoggingMixin):pass
log = CostLog()

#Cost Term Modes are a quick lookup for cost term support
global COST_TERM_MODES,COST_CATEGORIES
COST_TERM_MODES = {'initial': lambda inst,term: True if term < 1 else False,
                   'maintenance': lambda inst,term: True if term >= 1 else False,
                   'always': lambda inst,term: True}

category_type = typing.Union[str,list]
COST_CATEGORIES = set(('uncategorized',))



class cost_property(system_property):
    """A thin wrapper over `system_property` that will be accounted by `Economics` Components and apply term & categorization

    `cost_property` should return a float/int always and will raise an error if the return annotation is different, although annotations are not required and will default to float.

    #Terms:
    Terms start counting at 0 and can be evaluated by the Economic.term_length
    cost_properties will return their value as system_properties do without regard for the term state, however a CostModel's costs at a term can be retrived by `costs_at_term`. The default mode is for `initial` cost

    #Categories:
    Categories are a way to report cost categories and multiple can be applied to a cost. Categories are grouped by the Economics system at reported in bulk by term and over the term_length
    
    """

    cost_categories: list = None
    term_mode: str = None

    _all_modes: dict = COST_TERM_MODES
    _all_categories:set = COST_CATEGORIES

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, desc=None, label=None, stochastic=False, mode:str='initial',category:category_type=None):
        """extends system_property interface with mode & category keywords
        :param mode: can be one of `initial`,`maintenance`,`always` or a function with signature f(inst,term) as an integer and returning a boolean True if it is to be applied durring that term.
        """
        super().__init__(fget, fset, fdel, doc, desc, label, stochastic)
        if isinstance(mode,str):
            mode = mode.lower()
            assert mode in COST_TERM_MODES, f'mode: {mode} is not in {set(COST_TERM_MODES.keys())}'
            self.term_mode = mode
        elif callable(mode):
            fid = str(uuid.uuid4())
            self.__class__._all_modes[fid] = mode
            self.term_mode = fid
        else:
            raise ValueError(f'mode: {mode} must be cost term str or callable')


        if category is not None:
            if isinstance(category,str):
                self.cost_categories = category.split(',')
            elif isinstance(category,list):
                self.cost_categories = category
            else:
                raise ValueError(f'categories: {category} not string or list')
            for cc in self.cost_categories:
                self.__class__._all_categories.add(cc)
        else:
            self.cost_categories = ['uncategorized']

    def apply_at_term(self,inst,term):
        if term < 0:
            raise ValueError(f'negative term!')
        if self.__class__._all_modes[self.term_mode](inst,term):
            return True
        return False

    def get_func_return(self, func):
        """ensures that the function has a return annotation, and that return annotation is in valid sort types"""
        anno = func.__annotations__
        typ = anno.get("return", None)
        if typ is not None and not typ in (int, float):
            raise Exception(
                f"system_property input: function {func.__name__} must have valid return annotation of type: {(int,float)}"
            )
        else:
            self.return_type = float

@forge
class CostModel(TabulationMixin): 
    """CostModel is a mixin for components or systems that reports its costs through the `cost` system property, which by default sums the `item_cost` and `sub_items_cost`.

    `item_cost` is determined by `calculate_item_cost()` which by default uses: `cost_per_item` field to return the item cost, which defaults to `numpy.nan` if not set. Nan values are ignored and replaced with 0.
    
    `sub_items_cost` system_property summarizes the costs of any component in a SLOT that has a `CostModel` or for SLOTS which CostModel.declare_cost(`slot`,default=numeric|CostModelInst|dict[str,float])
    """
    _slot_costs: dict #TODO: insantiate per class

    cost_per_item: float = attrs.field(default=numpy.nan)

    #TODO: add dictionary & category implementations for economics comp to sum groups of
    #cost_category: str = attrs.field(default=None)

    def __on_init__(self):
        self.set_default_costs()

    def update_dflt_costs(self,callback=None):
        """updates internal default slot costs if the current component doesn't exist or isn't a cost model, this is really a component method but we will use it never the less.
        
        The cost model can be updated with a callback(dflt,parent) after update

        This should be called from Component.update() if default costs are used
        """

        if self._slot_costs:
            current_comps = self.internal_components()
            for k,v in self._slot_costs.items():
                #Check if the cost model will  be accessed
                no_comp = k not in current_comps
                is_cost = isinstance(current_comps[k],CostModel)
                dflt_is_cost_comp = all([isinstance(v,CostModel),isinstance(v,Component)])
                if no_comp and not is_cost and dflt_is_cost_comp:
                    self.debug('Updating default {k}')
                    v.update(self)

                    if callback:
                        callback(v,self)
                    

    def set_default_costs(self):
        """set default costs if no costs are set"""
        inter_config = self.internal_configurations()
        for k,dflt in self._slot_costs.items():
            if k not in inter_config and isinstance(dflt,CostModel):
                setattr(self,k,attrs.evolve(dflt,parent=self))
            elif k not in inter_config and isinstance(dflt,type) and issubclass(dflt,CostModel):
                self.warning(f'setting default cost {k} from costmodel class, provide a default instance instead!')
                setattr(self,k,dflt())

        #Reset cache
        self.internal_components(True)
    
    @classmethod
    def subcls_compile(cls):
        assert not issubclass(cls,ComponentIter), 'component iter not supported'
        cls.reset_cls_costs()

    @classmethod
    def reset_cls_costs(cls):
        cls._slot_costs = {}


    @classmethod
    def default_cost(cls,slot_name:str,cost:typing.Union[float,'CostModel'],warn_on_non_costmodel=True):
        """Provide a default cost for SLOT items that are not CostModel's. Cost is applied class wide, but can be overriden with custom_cost per instance"""
        assert not isinstance(cost,type), f'insantiate classes before adding as a cost!'
        assert slot_name in cls.slots_attributes(), f'slot {slot_name} doesnt exist'
        assert isinstance(cost,(float,int,dict)) or isinstance(cost,CostModel), 'only numeric types or CostModel instances supported'

        atrb = cls.slots_attributes()[slot_name]
        atypes = atrb.type.accepted
        if warn_on_non_costmodel and not any([issubclass(at,CostModel) for at in atypes]):
            log.warning(f'assigning cost to non CostModel based slot {slot_name}')

        cls._slot_costs[slot_name] = cost

        #IDEA: create slot if one doesn't exist, for dictionaries and assign a ComponentDict+CostModel in wide mode?

    def custom_cost(self,slot_name:str,cost:typing.Union[float,'CostModel'],warn_on_non_costmodel=True):
        """Takes class costs set, and creates a copy of the class costs, then applies the cost numeric or CostMethod in the same way but only for that instance of"""
        assert not isinstance(cost,type), f'insantiate classes before adding as a cost!'
        assert slot_name in self.slots_attributes(), f'slot {slot_name} doesnt exist'
        assert isinstance(cost,(float,int,dict)) or isinstance(cost,CostModel), 'only numeric types or CostModel instances supported'

        atrb = self.__class__.slots_attributes()[slot_name]
        atypes = atrb.type.accepted
        if warn_on_non_costmodel and not any([issubclass(at,CostModel) for at in atypes]):
            self.warning(f'assigning cost to non CostModel based slot {slot_name}')
            
        if self._slot_costs is self.__class__._slot_costs:
            self._slot_costs =  self.__class__._slot_costs.copy()
        self._slot_costs[slot_name] = cost
        self.set_default_costs()
    
        #if the cost is a cost model, and there's nothing assigned to the slot, assign it
        # if assign_when_missing and isinstance(cost,CostModel):
        #     if hasattr(self,slot_name) and getattr(self,slot_name) is None:
        #         self.info(f'assigning custom cost {slot_name} with {cost}')
        #         setattr(self,slot_name,cost)
        #     elif hasattr(self,slot_name):
        #         self.warning(f'could not assign custom cost to {slot_name} with {cost}, already assigned to {getattr(self,slot_name)}')


    def calculate_item_cost(self)->float:
        """override this with a parametric model related to this systems attributes and properties"""
        return self.cost_per_item
    
    @system_property
    def sub_items_cost(self)->float:
        """calculates the total cost of all sub-items, using the components CostModel if it is provided, and using the declared_cost as a backup"""
        return self.sub_costs()

    @cost_property(mode='initial',category='unit')
    def item_cost(self)->float:
        calc_item = self.calculate_item_cost()
        return numpy.nansum([0,calc_item])

    @system_property
    def combine_cost(self)->float:
        return self.sum_costs()
    
    @system_property
    def itemized_costs(self)->float:
        """sums costs of cost_property's in this item that are present at term=0"""
        initial_costs = self.costs_at_term(0)
        return numpy.nansum( list(initial_costs.values()) )
    
    @system_property
    def future_costs(self)->float:
        """sums costs of cost_property's in this item that do not appear at term=0"""
        initial_costs = self.costs_at_term(0,False)
        return numpy.nansum(list(initial_costs.values()))

    def sum_costs(self,saved:set=None,categories:tuple=None,term=0):
        """sums costs of cost_property's in this item that are present at term=0, and by category if define as input"""
        if saved is None:
            saved = set((self,)) #item cost included!
        elif self not in saved:
            saved.add(self)
        itemcst = list(self.dict_itemized_costs(saved,categories,term).values())
        csts = [self.sub_costs(saved,categories,term),numpy.nansum(itemcst)]
        return numpy.nansum(csts)

    def dict_itemized_costs(self,saved:set=None,categories:tuple=None,term=0,test_val = True)->dict:
        ccp = self.class_cost_properties()
        costs = {k: obj.__get__(self) if obj.apply_at_term(self,term)==test_val else 0 for k,obj in ccp.items() if categories is None or any([cc in categories for cc in obj.cost_categories])}
        return costs
         
    
    def sub_costs(self,saved:set=None,categories:tuple=None,term=0):
        """gets items from CostModel's defined in a SLOT attribute or in a slot default"""
        if saved is None:
            saved = set()

        sub_tot = 0
    
        for slot in self.slots_attributes():
            comp = getattr(self,slot)

            if comp in saved:
                #print(f'skipping {slot}:{comp}')
                continue

            elif isinstance(comp,Configuration):
                saved.add(comp)

            if isinstance(comp,CostModel):
                sub = comp.sum_costs(saved,categories,term)
                log.debug(f'{self} adding: {comp}: {sub}+{sub_tot}')
                cst = [sub_tot,sub]
                sub_tot = numpy.nansum(cst)

            
            elif slot in self._slot_costs and (categories is None or 'unit' in categories) and term==0:
                #Add default costs from direct slots
                dflt = self._slot_costs[slot]
                sub = evaluate_slot_cost(dflt,saved)                
                log.debug(f'{self} adding slot: {comp}.{slot}: {sub}+{sub_tot}')
                cst= [sub_tot,sub]
                sub_tot = numpy.nansum(cst)

            #add base class slot values when comp was nonee
            if comp is None:
                #print(f'skipping {slot}:{comp}')
                comp_cls = self.slots_attributes()[slot].type.accepted
                for cc in comp_cls:
                    if issubclass(cc,CostModel):
                        if cc._slot_costs:
                            for k,v in cc._slot_costs.items():
                                sub = evaluate_slot_cost(v,saved)
                                log.debug(f'{self} adding dflt: {slot}.{k}: {sub}+{sub_tot}')
                                cst= [sub_tot,sub]
                                sub_tot = numpy.nansum(cst)
                            break #only add once
                    

        return sub_tot

    #Cost Term & Category Reporting
    def costs_at_term(self,term:int,test_val=True)->dict:
        """returns a dictionary of all costs at term i, with zero if the mode 
        function returns False at that term"""
        ccp = self.class_cost_properties()
        return {k: obj.__get__(self) if obj.apply_at_term(self,term)==test_val else 0 for k,obj in ccp.items()}        

    @classmethod
    def class_cost_properties(cls)->dict:
        """returns cost_property objects from this class & subclasses"""
        return {k:v for k,v in cls.classmethod_system_properties().items() if isinstance(v,cost_property)}     
    
    @property
    def cost_properties(self)->dict:
        """returns the current values of the current properties"""
        ccp = self.class_cost_properties()
        return {k:obj.__get__(self) for k,obj in ccp.items()}
    
    @property
    def cost_categories(self):
        """returns itemized costs grouped by category"""
        base = {cc:0 for cc in self.all_categories()}
        for k,obj in self.class_cost_properties().items():
            for cc in obj.cost_categories:
                base[cc] += obj.__get__(self)
        return base

    def cost_categories_at_term(self,term:int):
        base = {cc:0 for cc in self.all_categories()}
        for k,obj in self.class_cost_properties().items():
            if obj.apply_at_term(self,term):
                for cc in obj.cost_categories:
                    base[cc] += obj.__get__(self)    
        return base            

    @classmethod
    def all_categories(self):
        return COST_CATEGORIES

cost_type = typing.Union[float,int,CostModel,dict]
def evaluate_slot_cost(slot_item:cost_type,saved:set=None):
    sub_tot = 0
    #log.debug(f'evaluating slot: {slot_item}')
    if isinstance(slot_item,(float,int)):
        sub_tot += numpy.nansum([slot_item,0])
    elif isinstance(slot_item,CostModel):
        sub_tot += numpy.nansum([slot_item.sum_costs(saved),0])
    elif isinstance(slot_item,type) and issubclass(slot_item,CostModel):
        log.warning(f'slot {slot_item} has class CostModel, using its `item_cost` only, create an instance to fully model the cost')
        sub_tot = numpy.nansum([sub_tot,slot_item.cost_per_item ])
    elif isinstance(slot_item,dict):
        sub_tot += numpy.nansum(list(slot_item.values()))
    return sub_tot

def gend(deect:dict):
    for k,v in deect.items():
        if isinstance(v,dict):
            for kk,v in gend(v):
                yield f'{k}.{kk}',v
        else:
            yield k,v
parent_types = typing.Union[Component,'System']
@forge
class Economics(Component): 
    """Economics is a component that summarizes costs and reports the economics of a system and its components in a recursive format"""

    term_length: int = attrs.field(default=0)
    discount_rate: float = attrs.field(default=0.0)
    fixed_output: float = attrs.field(default=numpy.nan)
    output_type: str = attrs.field(default='generic')
    terms_per_year: int = attrs.field(default=1)

    _calc_output: float = None
    _costs: float  = None
    _cost_references: dict = None
    _cost_categories: dict = None
    _comp_categories: dict = None
    _comp_costs: dict = None
    parent:parent_types

    def __on_init__(self):
        self._cost_categories = collections.defaultdict(list)
        self._comp_categories = collections.defaultdict(list)
        self._comp_costs = dict()        

    def update(self,parent:parent_types):
        #self.parent = parent
        
        self.parent = parent

        self._gather_cost_references(parent)
        self._calc_output = self.calculate_production(parent,0)
        self._costs = self.calculate_costs(parent)

        if self._calc_output is None:
            self.warning(f'no economic output!')
        if self._costs is None:
            self.warning(f'no economic costs!')

        # #Update child cost elements with parents
        # for slot,comp in self.internal_components().items():
        #     if isinstance(comp,CostModel):
        #         comp.update(self)

    def calculate_production(self,parent,term)->float:
        """must override this function and set economic_output"""
        return numpy.nansum([0,self.fixed_output])

    def calculate_costs(self,parent)->float:
        """recursively accounts for costs in the parent, its children recursively."""

        return self.sum_cost_references()
    
    #Reference Utilitly Functions   
    def sum_cost_references(self):
        cst = 0
        for k,v in self._cost_references.items():
            if k.endswith('item_cost'):
                cst += v.value()
        return cst

    def sum_references(self,refs):
        return numpy.nansum([r.value() for r in refs])
        
    def get_prop(self,ref):
        if ref.use_dict:
            return ref.key
        elif ref.key in ref.comp.class_cost_properties():
            return ref.comp.class_cost_properties()[ref.key]
        # elif ref.key in ref.comp.classmethod_system_properties():
        #     return ref.comp.classmethod_system_properties()[ref.key]
        # else:
        #     raise KeyError(f'ref key doesnt exist as property: {ref.key}')

    def term_fgen(self,comp,prop):
        if isinstance(comp,dict):
            return lambda term: comp[prop] if term == 0 else 0
        return lambda term: prop.__get__(comp) if prop.apply_at_term(comp,term) else 0

    def sum_term_fgen(self,ref_group):
        term_funs = [self.term_fgen(ref.comp,self.get_prop(ref)) 
                            for ref in ref_group]
        return lambda term: numpy.nansum([t(term) for t in term_funs])        

    #Gather & Set References (the magic!)
    def internal_references(self,recache=True):
        """standard component references are """
        if not recache and hasattr(self,'__cache_refs'):
            return self.__cache_refs
        
        d = self._gather_references()
        self._create_term_eval_functions()
        #Gather all internal economic variables and report costs
        props = d['properties']
        
        #calculate lifecycle costs
        lc_out = self.lifecycle_output

        if self._cost_references:
            props.update(**self._cost_references)

        if self._cost_categories:
            for key,refs in self._cost_categories.items():
                props[key] = Ref(self._cost_categories,key,False,False,eval_f=self.sum_references)

        if self._comp_categories:
            for key,refs in self._comp_categories.items():
                props[key] = Ref(self._comp_categories,key,False,False,eval_f=self.sum_references)    

        for k,v in lc_out.items():
            props[k] = Ref(lc_out,k,False,False)

        self.__cache_refs = d        
    
        return d

    @property
    def lifecycle_output(self)->dict:
        """return lifecycle calculations for lcoe"""
        totals = {}
        totals['category'] = lifecat = {}
        totals['annualized'] = annul = {}
        summary = {}
        out = {'summary':summary,'lifecycle':totals}

        lc = self.lifecycle_dataframe
        for c in lc.columns:
            if 'category' not in c and 'cost' not in c:
                continue
            tot = lc[c].sum()
            if 'category' in c:
                c_ = c.replace('category.','')
                lifecat[c_] = tot
            else:
                totals[c] = tot
            annul[c] = tot * self.terms_per_year / (self.term_length+1)

        summary['total_cost'] = lc.term_cost.sum()
        summary['years'] = lc.year.max()+1
        LC = lc.levalized_cost.sum()
        LO = lc.levalized_output.sum()
        summary['levalized_cost'] =  LC / LO if LO != 0 else numpy.nan
        summary['levalized_output'] = LO / LC if LC != 0 else numpy.nan
        

        out2 = dict(gend(out))
        self._term_output = out2
        return self._term_output


    @property
    def lifecycle_dataframe(self) -> pandas.DataFrame:
        """simulates the economics lifecycle and stores the results in a term based dataframe"""    
        out = []

        if self.term_length == 0:
            rng = [0]
        else:
            rng = list(range(0,self.term_length))
        
        for i in rng:
            t = i
            row = {'term':t,'year':t/self.terms_per_year}
            out.append(row)
            for k,sum_f in self._term_comp_category.items():
                row[k] = sum_f(t)
            for k,sum_f in self._term_cost_category.items():
                row[k] = sum_f(t)
            for k,sum_f in self._term_comp_cost.items():
                row[k] = sum_f(t)
            row['term_cost'] = tc = numpy.nansum([v(t) for v in self._term_comp_cost.values()])
            row['levalized_cost'] = tc * (1+self.discount_rate)**(-1*t)
            row['output'] = output =  self.calculate_production(self.parent,t)
            row['levalized_output'] = output * (1+self.discount_rate)**(-1*t)


        return pandas.DataFrame(out)


    def _create_term_eval_functions(self):
        """uses reference summation grouped by categories & component"""
        self._term_comp_category = {}
        if self._comp_categories:
            for k,vrefs in self._comp_categories.items():
                self._term_comp_category[k] = self.sum_term_fgen(vrefs)

        self._term_cost_category = {}
        if self._cost_categories:
            for k,vrefs in self._cost_categories.items():
                self._term_cost_category[k] = self.sum_term_fgen(vrefs)

        self._term_comp_cost = {}
        if self._comp_costs:
            for k, ref in self._comp_costs.items():
                prop = self.get_prop(ref)
                self._term_comp_cost[k] = self.term_fgen(ref.comp,prop)
        

    def _gather_cost_references(self,parent:'System'):
        """put many tabulation.Ref objects into a dictionary to act as additional references for this economics model.
        
        References are found from a walk through the parent slots through all child slots"""
        self._cost_references = CST = {}
        comps = {}
        comp_set = set()

        #print(f'gather cost refs')

        self._cost_categories = collections.defaultdict(list)
        self._comp_categories = collections.defaultdict(list)
        self._comp_costs = dict()
        for key,level,conf in parent.go_through_configurations(check_config=False):
            if conf is self:
                continue
            bse = f'{key}.' if key else ''
            #prevent duplicates'
            if conf in comp_set:
                continue
            elif isinstance(conf,Configuration):
                comp_set.add(conf)
            else:
                comp_set.add(key)

            
            kbase = '.'.join(key.split('.')[:-1])
            comp_key = key.split('.')[-1]

            #Get Costs Directly From the cost model instance
            if isinstance(conf,CostModel):
                comps[key] = conf
                self.debug(f'adding cost model for {kbase}.{comp_key}')
                self._extract_cost_references(conf,bse)
                
            #Look For Defaults in parent to determine if a 
            elif kbase and kbase in comps:
                self.debug(f'adding cost for {kbase}.{comp_key}')

                child = comps[kbase]
                if isinstance(child,CostModel) and comp_key in parent._slot_costs:
                    compcanidate = child._slot_costs[comp_key]
                    if isinstance(compcanidate,CostModel):
                        self.debug(f'dflt child costmodel {kbase}.{comp_key}')
                        self._extract_cost_references(compcanidate,bse+'cost.')
                    else:                    
                        _key=bse+'cost.item_cost'
                        self.debug(f'dflt child cost for {kbase}.{comp_key}')
                        CST[_key] = ref = Ref(child._slot_costs,comp_key,False,False, eval_f = evaluate_slot_cost)
                        cc = 'unit'
                        self._comp_costs[_key] = ref
                        self._cost_categories['category.'+cc].append(ref)
                        self._comp_categories[bse+'category.'+cc].append(ref)

            elif isinstance(parent,CostModel) and kbase == '' and comp_key in parent._slot_costs:
                
                compcanidate = parent._slot_costs[comp_key]
                if isinstance(compcanidate,CostModel):
                    self.debug(f'dflt parent cost model for {kbase}.{comp_key}')
                    self._extract_cost_references(compcanidate,bse+'cost.')
                else:
                    self.debug(f'dflt parent cost for {kbase}.{comp_key}')
                    _key=bse+'cost.item_cost'
                    CST[_key] = ref = Ref(parent._slot_costs,comp_key,False,False, eval_f = evaluate_slot_cost)
                    cc = 'unit'
                    self._comp_costs[_key] = ref
                    self._cost_categories['category.'+cc].append(ref)
                    self._comp_categories[bse+'category.'+cc].append(ref)

            else:
                self.debug(f'unhandled cost: {key}')
                    
        self._cost_references = CST
        self._anything_changed = True
        return CST
    
    def _extract_cost_references(self,conf:'CostModel',bse:str):
        #Add cost fields
        _key = bse+'item_cost'
        CST = self._cost_references

        for cost_nm,cost_prop in conf.class_cost_properties().items():
            _key=bse+'cost.'+cost_nm
            CST[_key] = ref = Ref(conf,cost_nm,True,False)
            self._comp_costs[_key] = ref

            #If there are categories we'll add references to later sum them
            if cost_prop.cost_categories:
                for cc in cost_prop.cost_categories:
                    self._cost_categories['category.'+cc].append(ref)
                    self._comp_categories[bse+'category.'+cc].append(ref)
            else:
                #we'll reference it as uncategorized
                cc = 'uncategorized'
                self._cost_categories['category.'+cc].append(ref)
                self._comp_categories[bse+'category.'+cc].append(ref)

        #add slot costs with now current items:
        for slot_name, slot_value in conf._slot_costs.items():
            #Skip items that are internal components
            if slot_name in conf.internal_components():
                self.debug(f'skipping slot {slot_name}')
                continue
            else:
                self.debug(f'adding slot {conf}.{slot_name}')
            #Check if current slot isn't occupied
            cur_slot = getattr(conf,slot_name)
            _key = bse+slot_name+'.cost.item_cost'
            if not isinstance(cur_slot,Configuration) and _key not in CST:
                CST[_key] = ref = Ref(conf._slot_costs,slot_name,False,False,eval_f = evaluate_slot_cost)

                cc = 'unit'
                self._comp_costs[_key] = ref
                self._cost_categories['category.'+cc].append(ref)
                self._comp_categories[bse+'category.'+cc].append(ref)
            elif _key in CST:
                self.debug(f'skipping key {_key}')

        #add base class slot values when comp was nonee
        for compnm,comp in conf.internal_configurations(False).items():
            if comp is None:
                
                comp_cls = conf.slots_attributes()[compnm].type.accepted
                for cc in comp_cls:
                    if issubclass(cc,CostModel):
                        if cc._slot_costs:
                            for k,v in cc._slot_costs.items():
                                _key=bse+compnm+'.'+k+'.cost.item_cost'
                                if _key in CST:
                                    break #skip if already added

                                if isinstance(v,CostModel):
                                    self._extract_cost_references(v,bse+compnm+'.'+k+'.')
                                else:
                                    
                                    self.debug(f'adding missing cost for {conf}.{compnm}')
                                    CST[_key] = ref = Ref(cc._slot_costs,k,False,False,eval_f = evaluate_slot_cost)

                                    cc = 'unit'
                                    self._comp_costs[_key] = ref
                                    self._cost_categories['category.'+cc].append(ref)
                                    self._comp_categories[bse+'category.'+cc].append(ref)
                                
                            break #only add once      

    @property
    def cost_references(self):
        return self._cost_references

    @system_property
    def combine_cost(self)->float:
        if self._costs is None:
            return 0
        return self._costs
    
    @system_property
    def output(self)->float:
        if self._calc_output is None:
            return 0
        return self._calc_output


# if isinstance(conf,ComponentIter):
#     conf = conf.current
#     #if isinstance(conf,CostModel):
#     #    sub_tot += conf.item_cost
# if isinstance(conf,ComponentIter):
#     item = conf.current
#     if conf.wide:
#         items = item
#     else:
#         items = [items]
# else:
#     items = [conf]
#for conf in items:


# if isinstance(self,CostModel):
#     sub_tot += self.item_cost

#accomodate ComponentIter in wide mode
# if isinstance(self,ComponentIter):
#     item = self.current
#     if self.wide:
#         items = item
#     else:
#         items = [items]
# else:
#     items = [self]

#accomodate ComponentIter in wide mode
#for item in items: