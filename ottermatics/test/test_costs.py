"""Testing of costs reporting module
- emphasis on recursive testing w/o double accounting on instance basis (id!)
- emphasis on default / subclass adjustments
"""


import unittest
from ottermatics.configuration import otterize
from ottermatics.eng.costs import CostMixin, Economics
from ottermatics.slots import SLOT
from ottermatics.system import System
from ottermatics.components import Component
from ottermatics.component_collections import ComponentIterator

@otterize
class Norm(Component):
    pass

@otterize
class Comp1(Component,CostMixin):
    norm = SLOT.define(Norm,none_ok=True)

@otterize
class Comp2(Norm,CostMixin):

    comp1 = SLOT.define(Comp1,none_ok=True,default_ok=False)

@otterize
class CompGroup(ComponentIterator,CostMixin):
    pass

@otterize
class EconRecursive(System,CostMixin):
    econ = SLOT.define(Economics)
    comp1 = SLOT.define(Comp1)
    comp2 = SLOT.define(Comp2)

@otterize
class EconWide(System):
    econ = SLOT.define(Economics)
    comp_set = SLOT.define_iterator(CompGroup,wide=True)

@otterize
class EconNarrow(System):
    econ = SLOT.define(Economics)
    comp_set = SLOT.define_iterator(CompGroup)


class TestCostAccounting(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    #def test_



class TestCostMixin(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self) -> None:
        Comp1.reset_cls_costs()
        Comp2.reset_cls_costs()

    def test_comp1(self):
        c0 = Comp1(cost_per_item=0)
        self.assertEqual(c0.cost , 0)

        c1 = Comp1(cost_per_item=5)
        self.assertEqual(c1.cost , 5)

        c2 = Comp1(cost_per_item=10)
        self.assertEqual(c2.cost , 10)

    def test_comp2(self):
        c1 = Comp1(cost_per_item=0)
        c2 = Comp2(cost_per_item = 0,comp1=c1)
        self.assertEqual(c2.cost , 0)

        c1 = Comp1(cost_per_item=5)
        c2 = Comp2(cost_per_item = 5,comp1=c1)
        self.assertEqual(c2.cost , 10)
        
        c1 = Comp1(cost_per_item=10)
        c2 = Comp2(cost_per_item = 10,comp1=c1)
        self.assertEqual(c2.cost , 20)

    def test_default(self):
        c1 = Comp1(cost_per_item=0)
        self.assertEqual(c1.cost , 0)

        Comp1.default_cost('norm',10)
        c1 = Comp1(cost_per_item=0)
        self.assertEqual(c1.cost , 10)

        c1 = Comp1(cost_per_item=0)
        c1.custom_cost('norm',20)
        self.assertEqual(c1.cost , 20)
        self.assertEqual(Comp1._slot_costs['norm'] , 10)
        
    def test_ref_loop(self):
        #Comp1.default_cost('norm',10)
        c2 = Comp2(cost_per_item=5)
        c1 = Comp1(cost_per_item=5,norm=c2)        
        c2.comp1 = c1
        self.assertEqual(c1.cost , 10)
        self.assertEqual(c1.sub_items_cost , 10)

    def test_override(self):
        Comp1.default_cost('norm',10)
        #c2 = Comp2(cost_per_item=5)
        c1 = Comp1(cost_per_item=5)#,norm=c2)
        self.assertEqual(c1.cost , 15)
        self.assertEqual(c1.sub_items_cost , 10)        

        #Comp1.reset_cls_costs()
        c2 = Comp2(cost_per_item=15)
        c1 = Comp1(cost_per_item=15,norm=c2)
        self.assertEqual(c1.cost , 30)
        self.assertEqual(c1.sub_items_cost , 15)        


if __name__ == '__main__':

    unittest.main()


