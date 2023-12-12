from ottermatics.components import Component
from ottermatics.configuration import otterize
from ottermatics.tabulation import TabulationMixin, system_property
from ottermatics.properties import instance_cached

import typing
import attrs

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
    def cost(self)->float:
        return self.sum_costs()

    def sum_costs(self,saved=None):
        if saved is None:
            saved = set((self,)) #item cost included!
        return self.sub_costs(saved) + self.item_cost
    
    def sub_costs(self,saved=None):
        if saved is None:
            saved = set()
        sub_tot = 0
        for slot,comp in self.internal_components.items():
            if comp in saved:
                #print(f'skipping {slot}:{comp}')
                continue
            else:
                saved.add(comp)
            if isinstance(comp,CostMixin):
                sub_tot += comp.sum_costs(saved)
            elif slot in self._slot_costs:
                dflt = self._slot_costs[slot]
                if isinstance(dflt,(float,int)):
                    sub_tot += dflt
                elif isinstance(dflt,CostMixin):
                    sub_tot += dflt.sum_costs(saved)
                elif isinstance(dflt,type) and issubclass(dflt,CostMixin):
                    self.warning(f'slot {slot} has class CostMixin, using its `item_cost` only, create an instance to fully model the cost')
                    sub_tot += dflt.cost_per_item
                elif isinstance(dflt,dict):
                    sub_tot += sum(dflt.values())

        return sub_tot        

@otterize
class Economics(Component): 
    """Economics is a component that summarizes costs and reports the economics of a system and its components in a recursive format"""

    term: float = attrs.field(default=10)
    discount_rate: float = attrs.field(default=0.05)

    _output: float = None
    _costs: float  = None

    def update(self,parent:typing.Union[Component,'System']):
        self._output = self.calculate_production(parent)
        self._costs = self.calculate_costs(parent)

        if self._output is None:
            self.warning(f'no economic output!')
        if self._costs is None:
            self.warning(f'no economic costs!')

    def calculate_production(self,parent)->float:
        """must override this function and set economic_output"""
        return None

    def calculate_costs(self,parent)->float:
        """recursively accounts for costs in the parent, its children recursively. Add a cost term application rule by function(term) and keyword: term ==1 is initial >1 is maintinance > 0 is always """
        return None


    #@instance_cached
    def internal_references(self,recache=True):
        """standard component references are """
        d = self._gather_references()

        #TODO: Gather all internal economic variables and report costs
        return d

    @system_property
    def total_cost(self)->float:
        return self._costs
    
    @system_property
    def output(self)->float:
        return self._output     

    @system_property
    def levalized_cost(self)->float:
        cost = self.total_cost
        output = self.output

        #TODO: calculate levelized cost




