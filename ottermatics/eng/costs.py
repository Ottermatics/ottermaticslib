"""Defines a CostModel & Economics Component that define & orchestrate cost accounting respectively.

CostModels can have a `cost_per_item` and additionally calculate a `cumulative_cost` from internally defined `CostModel`s.

CostModel's can have cost_property's which detail how and when a cost should be applied & grouped. By default each CostModel has a `cost_per_item` which is reflected in `item_cost` cost_property set on the `initial` term as a `unit` category. Multiple categories of cost are also able to be set on cost_properties as follows

```
@otterize
class Widget(Component,CostModel):

    @cost_property(mode='initial',category='capex,manufacturing')
    def cost_of_XYZ(self):
        return ...
```

Economics models sum CostModel.cost_properties recursively on the parent they are defined. Economics computes the grouped category costs for each item recursively as well as summary properties like annualized values and levalized cost. Economic output is determined by a `fixed_output` or overriding `calculate_production(self,parent)` to dynamically calculate changing economics based on factors in the parent.

The economics term_length applies costs over the term, using the `cost_property.mode` to determine at which terms a cost should be applied.

class Parent(System)

    econ = SLOT.define(Economics)

Parent(econ=Economics(term_length=25,discount_rate=0.05,fixed_output=1000))

#WARNING: this module may not work as intended due to injection fwd of committ stream for compiling purposes from neptunya branch
"""


from ottermatics.components import Component
from ottermatics.configuration import otterize,Configuration
from ottermatics.tabulation import TabulationMixin, system_property, Ref
from ottermatics.properties import instance_cached,solver_cached
from ottermatics.logging import LoggingMixin
from ottermatics.component_collections import ComponentIter
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

@otterize
class CostModel(TabulationMixin): 
    """CostModel is a mixin for components or systems that reports its costs through the `cost` system property, which by default sums the `item_cost` and `sub_items_cost`.

    `item_cost` is determined by `calculate_item_cost()` which by default uses: `cost_per_item` field to return the item cost
    
    `sub_items_cost` system_property summarizes the costs of any component in a SLOT that has a `CostModel` or for SLOTS which CostModel.declare_cost(`slot`,default=numeric|CostModelInst|dict[str,float])
    """
    _slot_costs: dict #TODO: insantiate per class

    cost_per_item: float = attrs.field(default=numpy.nan)

    #TODO: add dictionary & category implementations for economics comp to sum groups of
    #cost_category: str = attrs.field(default=None)

    
    @classmethod
    def subcls_compile(cls):
        assert not issubclass(cls,ComponentIter), 'component iter not supported'
        cls.reset_cls_costs()

    @classmethod
    def reset_cls_costs(cls):
        cls._slot_costs = {}


    @classmethod
    def default_cost(cls,slot_name:str,cost:typing.Union[float,'CostModel']):
        """Provide a default cost for SLOT items that are not CostModel's. Cost is applied class wide, but can be overriden with custom_cost per instance"""
        assert not isinstance(cost,type), f'insantiate classes before adding as a cost!'
        assert slot_name in cls.slots_attributes(), f'slot {slot_name} doesnt exist'
        assert isinstance(cost,(float,int,dict)) or isinstance(cost,CostModel), 'only numeric types or CostModel instances supported'
        cls._slot_costs[slot_name] = cost

        #IDEA: create slot if one doesn't exist, for dictionaries and assign a ComponentDict+CostModel in wide mode?

    def custom_cost(self,slot_name:str,cost:typing.Union[float,'CostModel']):
        """Takes class costs set, and creates a copy of the class costs, then applies the cost numeric or CostMethod in the same way but only for that instance of"""
        assert not isinstance(cost,type), f'insantiate classes before adding as a cost!'
        assert slot_name in self.slots_attributes(), f'slot {slot_name} doesnt exist'
        assert isinstance(cost,(float,int,dict)) or isinstance(cost,CostModel), 'only numeric types or CostModel instances supported'

        if self._slot_costs is self.__class__._slot_costs:
            self._slot_costs =  self.__class__._slot_costs.copy()
        self._slot_costs[slot_name] = cost     


    def calculate_item_cost(self)->float:
        """override this with a parametric model related to this systems attributes and properties"""
        return self.cost_per_item
    
    @system_property
    def sub_items_cost(self)->float:
        """calculates the total cost of all sub-items, using the components CostModel if it is provided, and using the declared_cost as a backup"""
        return self.sub_costs()

    @cost_property
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

    def sum_costs(self,saved:set=None):
        if saved is None:
            saved = set((self,)) #item cost included!
        elif self not in saved:
            saved.add(self)
        csts = [self.sub_costs(saved),self.itemized_costs]
        return numpy.nansum(csts)
         
    
    def sub_costs(self,saved:set=None):
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
                sub = comp.sum_costs(saved)
                log.debug(f'{self} adding: {comp}: {sub}+{sub_tot}')
                cst = [sub_tot,sub]
                sub_tot = numpy.nansum(cst)

            elif slot in self._slot_costs:
                dflt = self._slot_costs[slot]
                sub = evaluate_slot_cost(dflt,saved)
                log.debug(f'{self} adding slot: {comp}.{slot}: {sub}+{sub_tot}')
                cst= [sub_tot,sub]
                sub_tot = numpy.nansum(cst)

        return sub_tot

    #Cost Term & Category Reporting
    def costs_at_term(self,term:int,test_val=True)->dict:
        """returns a dictionary of all costs at term i, with zero if the mode 
        function returns False at that term"""
        ccp = self.class_cost_properties()
        return {k: obj.__get__(self) if obj.apply_at_term(self,term)==test_val else 0 
                  for k,obj in ccp.items()}        

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

@otterize
class Economics(Component): 
    """Economics is a component that summarizes costs and reports the economics of a system and its components in a recursive format"""

    term_length: int = attrs.field(default=0)
    discount_rate: float = attrs.field(default=0.0)
    fixed_output: float = attrs.field(default=numpy.nan)
    output_type: str = attrs.field(default='generic')
    terms_per_year: int = attrs.field(default=1)

    _output: float = None
    _costs: float  = None
    _cost_references: dict = None
    _cost_categories: dict = None
    _comp_categories: dict = None
    _comp_costs: dict = None
    parent:'System'

    def __on_init__(self):
        self._cost_categories = collections.defaultdict(list)
        self._comp_categories = collections.defaultdict(list)
        self._comp_costs = dict()        

    def update(self,parent:typing.Union[Component,'System']):
        #self.parent = parent
        self._gather_cost_references(parent)
        self._output = self.calculate_production(parent)
        self._costs = self.calculate_costs(parent)

        if self._output is None:
            self.warning(f'no economic output!')
        if self._costs is None:
            self.warning(f'no economic costs!')

        # #Update child cost elements with parents
        # for slot,comp in self.internal_components.items():
        #     if isinstance(comp,CostModel):
        #         comp.update(self)

    def calculate_production(self,parent)->float:
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
        summary['levalized_cost'] =  LC / LO
        summary['levalized_output'] = LO / LC
        

        out2 = dict(gend(out))
        self._term_output = out2
        return self._term_output


    @property
    def lifecycle_dataframe(self) -> pandas.DataFrame:
        """simulates the economics lifecycle and stores the results in a term based dataframe"""    
        out = []

        for i in range(self.term_length+1):
            row = {'term':i,'year':i/self.terms_per_year}
            out.append(row)
            for k,sum_f in self._term_comp_category.items():
                row[k] = sum_f(i)
            for k,sum_f in self._term_cost_category.items():
                row[k] = sum_f(i)
            for k,sum_f in self._term_comp_cost.items():
                row[k] = sum_f(i)
            row['term_cost'] = tc = numpy.nansum([v(i) for v in self._term_comp_cost.values()])
            row['levalized_cost'] = tc / (1+self.discount_rate)**i
            row['levalized_output'] = self.output / (1+self.discount_rate)**i


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
        CST = {}
        comps = {}
        comp_set = set()

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

            comps[key] = conf
            kbase = '.'.join(key.split('.')[:-1])
            comp_key = key.split('.')[-1]

            #Get Costs Directly
            if isinstance(conf,CostModel):
                CST[bse+'combine_cost'] = Ref(conf,'combine_cost',True,False)
                _key = bse+'item_cost'
                #CST[_key] = ref = Ref(conf,'item_cost',True,False)
                CST[bse+'sub_cost'] = Ref(conf,'sub_items_cost',True,False)
                # cc = 'unit'
                # self._comp_costs[_key] = ref
                # self._cost_categories['category.'+cc].append(ref)
                # self._comp_categories[bse+'category.'+cc].append(ref)   

                #Add cost fields
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
                    cur_slot = getattr(conf,slot_name)
                    if not isinstance(cur_slot,Configuration):
                        _key = bse+slot_name+'.item_cost'
                        CST[_key] = ref = Ref(conf._slot_costs,slot_name,False,False,eval_f = evaluate_slot_cost)

                        cc = 'unit'
                        self._comp_costs[_key] = ref
                        self._cost_categories['category.'+cc].append(ref)
                        self._comp_categories[bse+'category.'+cc].append(ref)

            #Look For Defaults
            elif kbase and kbase in comps:
                child = comps[kbase]
                if isinstance(parent,CostModel) and comp_key in child._slot_costs:
                    _key=bse+'item_cost'
                    CST[_key] = ref = Ref(child._slot_costs,comp_key,False,False, eval_f = evaluate_slot_cost)
                    cc = 'unit'
                    self._comp_costs[_key] = ref
                    self._cost_categories['category.'+cc].append(ref)
                    self._comp_categories[bse+'category.'+cc].append(ref)

                elif isinstance(parent,CostModel) and kbase == '' and comp_key in parent._slot_costs:
                    _key=bse+'item_cost'
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
        if self._output is None:
            return 0
        return self._output


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