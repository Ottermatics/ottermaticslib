from ottermatics.components import Component
from ottermatics.configuration import otterize,Configuration
from ottermatics.tabulation import TabulationMixin, system_property, Ref
from ottermatics.properties import instance_cached
from ottermatics.logging import LoggingMixin
from ottermatics.component_collections import ComponentIter
import typing
import attrs

class CostLog(LoggingMixin):pass
log = CostLog()

#Cost Term Modes are a quick lookup for cost term support
cost_term_modes = {'initial': lambda term: True if term <= 1 else False,
                   'maintenance': lambda term: True if term > 1 else False,
                   'always': lambda term: True}

class cost_property(system_property):
    """A thin wrapper over `system_property` that will be accounted by `Economics` Components
    
    #todo: add categories & terms
    """
    pass

@otterize
class CostMixin(TabulationMixin): 
    """CostMixin is a mixin for components or systems that reports its costs through the `cost` system property, which by default sums the `item_cost` and `sub_items_cost`.

    `item_cost` is determined by `calculate_item_cost()` which by default uses: `cost_per_item` field to return the item cost
    
    `sub_items_cost` system_property summarizes the costs of any component in a SLOT that has a `CostMixin` or for SLOTS which CostMixin.declare_cost(`slot`,default=numeric|CostMixinInst|dict[str,float])
    """
    _slot_costs: dict #TODO: insantiate per class

    cost_per_item: float = attrs.field(default=0)

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
    def default_cost(cls,slot_name:str,cost:typing.Union[float,'CostMixin']):
        """Provide a default cost for SLOT items that are not CostMixin's. Cost is applied class wide, but can be overriden with custom_cost per instance"""
        assert not isinstance(cost,type), f'insantiate classes before adding as a cost!'
        assert slot_name in cls.slots_attributes(), f'slot {slot_name} doesnt exist'
        assert isinstance(cost,(float,int,dict)) or isinstance(cost,CostMixin), 'only numeric types or CostMixin instances supported'
        cls._slot_costs[slot_name] = cost

        #IDEA: create slot if one doesn't exist, for dictionaries and assign a ComponentDict+CostMixin in wide mode?

    def custom_cost(self,slot_name:str,cost:typing.Union[float,'CostMixin']):
        """Takes class costs set, and creates a copy of the class costs, then applies the cost numeric or CostMethod in the same way but only for that instance of"""
        assert not isinstance(cost,type), f'insantiate classes before adding as a cost!'
        assert slot_name in self.slots_attributes(), f'slot {slot_name} doesnt exist'
        assert isinstance(cost,(float,int,dict)) or isinstance(cost,CostMixin), 'only numeric types or CostMixin instances supported'

        if self._slot_costs is self.__class__._slot_costs:
            self._slot_costs =  self.__class__._slot_costs.copy()
        self._slot_costs[slot_name] = cost     


    def calculate_item_cost(self)->float:
        """override this with a parametric model related to this systems attributes and properties"""
        return self.cost_per_item
    
    @system_property
    def sub_items_cost(self)->float:
        """calculates the total cost of all sub-items, using the components CostMixin if it is provided, and using the declared_cost as a backup"""
        return self.sub_costs()

    @system_property
    def item_cost(self)->float:
        return self.calculate_item_cost()

    @system_property
    def combine_cost(self)->float:
        return self.sum_costs()

    def sum_costs(self,saved:set=None):
        if saved is None:
            saved = set((self,)) #item cost included!
        return self.sub_costs(saved) + self.item_cost
    
    def sub_costs(self,saved:set=None):
        if saved is None:
            saved = set()

        sub_tot = 0

        # if isinstance(self,CostMixin):
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
    
        for slot in self.slots_attributes():
            comp = getattr(self,slot)

            if comp in saved:
                #print(f'skipping {slot}:{comp}')
                continue
            elif isinstance(comp,Configuration):
                saved.add(comp)

            if isinstance(comp,CostMixin):
                sub_tot += comp.sum_costs(saved)

            elif slot in self._slot_costs:
                dflt = self._slot_costs[slot]

                sub_tot += evaluate_slot_cost(dflt,saved)

        return sub_tot        

cost_type = typing.Union[float,int,CostMixin,dict]
def evaluate_slot_cost(slot_item:cost_type,saved:set=None):
    sub_tot = 0
    if isinstance(slot_item,(float,int)):
        sub_tot += slot_item
    elif isinstance(slot_item,CostMixin):
        sub_tot += slot_item.sum_costs(saved)
    elif isinstance(slot_item,type) and issubclass(slot_item,CostMixin):
        log.warning(f'slot {slot_item} has class CostMixin, using its `item_cost` only, create an instance to fully model the cost')
        sub_tot += slot_item.cost_per_item
    elif isinstance(slot_item,dict):
        sub_tot += sum(slot_item.values())
    return sub_tot

@otterize
class Economics(Component): 
    """Economics is a component that summarizes costs and reports the economics of a system and its components in a recursive format"""

    term_length: float = attrs.field(default=10)
    discount_rate: float = attrs.field(default=0.05)
    output_type: str = attrs.field(default='generic')

    _output: float = None
    _costs: float  = None
    _cost_references: dict = None
    parent:'System'


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
        #     if isinstance(comp,CostMixin):
        #         comp.update(self)

    def calculate_production(self,parent)->float:
        """must override this function and set economic_output"""
        return None

    def calculate_costs(self,parent)->float:
        """recursively accounts for costs in the parent, its children recursively. 
        
        #TODO: Add a cost term application rule by function(term) and keyword: term ==1 is initial >1 is maintinance > 0 is always
        #TODO: Add cost categorization"""

        return self.sum_cost_references()

    def sum_cost_references(self):
        cst = 0
        for k,v in self._cost_references.items():
            if k.endswith('item_cost'):
                cst += v.value()
        return cst
        

    def internal_references(self,recache=True):
        """standard component references are """
        d = self._gather_references()
        #Gather all internal economic variables and report costs
        if self._cost_references:
            d['properties'].update(**self._cost_references)
        return d
    
    def _gather_cost_references(self,parent):
        
        CST = {}
        comps = {}
        comp_set = set()
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

            # if isinstance(conf,ComponentIter):
            #     conf = conf.current
            #     #if isinstance(conf,CostMixin):
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

            #Get Costs Directly
            if isinstance(conf,CostMixin):
                CST[bse+'combine_cost'] = Ref(conf,'combine_cost',True,False)
                CST[bse+'item_cost'] = Ref(conf,'item_cost',True,False)
                CST[bse+'sub_cost'] = Ref(conf,'sub_items_cost',True,False)

                #add slot costs with now current items:
                for slot_name, slot_value in conf._slot_costs.items():
                    cur_slot = getattr(conf,slot_name)
                    if not isinstance(cur_slot,Configuration):
                        CST[bse+slot_name+'.item_cost'] = Ref(conf._slot_costs,slot_name,False,False,eval_f=evaluate_slot_cost)

            #Look For Defaults
            elif kbase and kbase in comps:
                child = comps[kbase]
                if isinstance(parent,CostMixin) and comp_key in child._slot_costs:
                    CST[bse+'item_cost'] = Ref(child._slot_costs,comp_key,False,False,eval_f=evaluate_slot_cost)

                elif isinstance(parent,CostMixin) and kbase == '' and comp_key in parent._slot_costs:
                    CST[bse+'item_cost'] = Ref(parent._slot_costs,comp_key,False,False,eval_f=evaluate_slot_cost)

            else:
                self.debug(f'unhandled cost: {key}')
                    
        self._cost_references = CST
        return CST

    @property
    def cost_references(self):
        return self._cost_references

    @system_property
    def combine_cost(self)->float:
        return self._costs
    
    @system_property
    def output(self)->float:
        return self._output     

    @system_property
    def levalized_cost(self)->float:
        cost = self.combine_cost
        output = self.output

        #TODO: calculate levelized cost




