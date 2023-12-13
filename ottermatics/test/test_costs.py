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
class CompGroup(ComponentIterator):
    pass
# @otterize
# class CostGroup(CompGroup,CostMixin):
#     pass

@otterize
class EconRecursive(System,CostMixin):
    econ = SLOT.define(Economics)
    comp1 = SLOT.define(Comp1,none_ok=True)
    comp2 = SLOT.define(Comp2,none_ok=True)

@otterize
class EconWide(System):
    econ = SLOT.define(Economics)
    comp_set = SLOT.define_iterator(CompGroup,wide=True)

@otterize
class EconNarrow(System):
    econ = SLOT.define(Economics)
    comp_set = SLOT.define_iterator(CompGroup,wide=False)

#TODO: get cost accounting working for ComponentIter components
# class TestEconomicsIter(unittest.TestCase):
# 
#     def tearDown(self):
#         Comp1.reset_cls_costs()
#         Comp2.reset_cls_costs()
#         EconWide.reset_cls_costs()
#         EconNarrow.reset_cls_costs()
#         CostGroup.reset_cls_costs()
# 
#     def test_comp_wide(self):
# 
#     def test_comp_narrow(self):
# 
#     def test_cost_wide(self):
# 
#     def test_comp_narrow(self):
#         cg = CompGroup(component_type=Comp1)
#         total = 0
#         for i in range(11):
#             cg.data.append(Comp1(cost_per_item=i))
#             total += i
# 
#         ew = EconNarrow(comp_set = cg)
#         ew.run()
#         
#         d = ew.data_dict
#         self.assertEqual(d['econ.comp_set.item_cost'], ew.comp_set.current.item_cost)
#         self.assertEqual(ew.econ.total_cost,total)
#         self.assertEqual(ew.econ.total_cost,total)

class TestEconomicsAccounting(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        Comp1.reset_cls_costs()
        Comp2.reset_cls_costs()
        EconRecursive.reset_cls_costs()

    def test_recursive_null(self,ANS=60):

        Comp1.default_cost('norm',5)
        Comp2.default_cost('comp1',10)
        EconRecursive.default_cost('comp1',3)
        EconRecursive.default_cost('comp2',7)
        er = EconRecursive(cost_per_item=50,comp1=None,comp2=None)
        er.run()
        self.assertEqual(er.combine_cost,ANS)
        self.assertEqual(er.econ.combine_cost,ANS)
        self.assertEqual(er.comp1,None)
        self.assertEqual(er.comp2,None)

        d = er.data_dict
        self.assertEqual(ANS,d['econ.combine_cost'])
        self.assertEqual(ANS,d['combine_cost'])
        self.assertEqual(3,d['econ.comp1.item_cost'])
        self.assertEqual(7,d['econ.comp2.item_cost'])



    def test_recursive_comp2(self,ANS=63):

        Comp1.default_cost('norm',5)
        Comp2.default_cost('comp1',10)
        EconRecursive.default_cost('comp1',3)
        EconRecursive.default_cost('comp2',7)
        er = EconRecursive(cost_per_item=50,comp1=None)
        er.run()
        self.assertEqual(er.combine_cost,ANS)
        self.assertEqual(er.econ.combine_cost,ANS)
        self.assertEqual(er.comp1,None)

        d = er.data_dict
        self.assertEqual(ANS,d['econ.combine_cost'])
        self.assertEqual(ANS,d['combine_cost']) 
        self.assertEqual(3,d['econ.comp1.item_cost'])
        self.assertEqual(10,d['econ.comp2.combine_cost'])

    def test_recursive_all(self,ANS=65):

        Comp1.default_cost('norm',5)
        Comp2.default_cost('comp1',10)
        EconRecursive.default_cost('comp1',3)
        EconRecursive.default_cost('comp2',7)
        er = EconRecursive(cost_per_item=50)
        er.run()
        self.assertEqual(er.combine_cost,ANS)
        self.assertEqual(er.econ.combine_cost,ANS)

        d = er.data_dict
        self.assertEqual(ANS,d['econ.combine_cost'])
        self.assertEqual(ANS,d['combine_cost']) 
        self.assertEqual(5,d['econ.comp1.combine_cost'])
        self.assertEqual(10,d['econ.comp2.combine_cost']) 
        self.assertEqual(10,d['econ.comp2.comp1.item_cost'])     
        self.assertEqual(5,d['econ.comp1.norm.item_cost'])                




class TestCostMixin(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self) -> None:
        Comp1.reset_cls_costs()
        Comp2.reset_cls_costs()

    def test_comp1(self):
        c0 = Comp1(cost_per_item=0)
        self.assertEqual(c0.combine_cost, 0)

        c1 = Comp1(cost_per_item=5)
        self.assertEqual(c1.combine_cost, 5)

        c2 = Comp1(cost_per_item=10)
        self.assertEqual(c2.combine_cost, 10)

    def test_comp2(self):
        c1 = Comp1(cost_per_item=0)
        c2 = Comp2(cost_per_item = 0,comp1=c1)
        self.assertEqual(c2.combine_cost, 0)

        c1 = Comp1(cost_per_item=5)
        c2 = Comp2(cost_per_item = 5,comp1=c1)
        self.assertEqual(c2.combine_cost, 10)
        
        c1 = Comp1(cost_per_item=10)
        c2 = Comp2(cost_per_item = 10,comp1=c1)
        self.assertEqual(c2.combine_cost, 20)

    def test_default(self):
        c1 = Comp1(cost_per_item=0)
        self.assertEqual(c1.combine_cost, 0)

        Comp1.default_cost('norm',10)
        c1 = Comp1(cost_per_item=0)
        self.assertEqual(c1.combine_cost, 10)

        c1 = Comp1(cost_per_item=0)
        c1.custom_cost('norm',20)
        self.assertEqual(c1.combine_cost, 20)
        self.assertEqual(Comp1._slot_costs['norm'] , 10)
        
    def test_ref_loop(self):
        #Comp1.default_cost('norm',10)
        c2 = Comp2(cost_per_item=5)
        c1 = Comp1(cost_per_item=5,norm=c2)        
        c2.comp1 = c1
        self.assertEqual(c1.combine_cost, 10)
        self.assertEqual(c1.sub_items_cost , 10)

    def test_override(self):
        Comp1.default_cost('norm',10)
        #c2 = Comp2(cost_per_item=5)
        c1 = Comp1(cost_per_item=5)#,norm=c2)
        self.assertEqual(c1.combine_cost, 15)
        self.assertEqual(c1.sub_items_cost , 10)        

        #Comp1.reset_cls_costs()
        c2 = Comp2(cost_per_item=15)
        c1 = Comp1(cost_per_item=15,norm=c2)
        self.assertEqual(c1.combine_cost, 30)
        self.assertEqual(c1.sub_items_cost , 15)        


if __name__ == '__main__':

    unittest.main()


