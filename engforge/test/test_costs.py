"""Testing of costs reporting module
- emphasis on recursive testing w/o double accounting on instance basis (id!)
- emphasis on default / subclass adjustments
"""


import unittest
from engforge.components import Component
from engforge.test.solver_testing_components import *


class TestEconDefaults(unittest.TestCase):
    def tearDown(self) -> None:
        Comp1.reset_cls_costs()
        EconDefault.reset_cls_costs()

    def test_custom_costs(self):
        ed = EconDefault()
        self.assertNotIn("comp1", ed.internal_components())

        ed.custom_cost("comp1", Comp1())
        self.assertIn("comp1", ed.internal_components())

    def test_non_costmodel_default(self):
        EconDefault.default_cost("comp", Comp1(cost_per_item=10))
        ed = EconDefault(comp=Component())
        ed.run()
        self.assertEqual(ed.combine_cost, 10)
        self.assertEqual(ed.econ.combine_cost, 10)


class TestCategoriesAndTerms(unittest.TestCase):
    def tearDown(self) -> None:
        Comp1.reset_cls_costs()
        Comp2.reset_cls_costs()
        EconRecursive.reset_cls_costs()
        TermCosts.reset_cls_costs()

    def test_itemized_costs(self):
        tc = TermCosts()

        ct = tc.costs_at_term(0)
        self.assertEqual(ct["cost_init"], 100)
        self.assertEqual(ct["cost_maintenance"], 0)
        self.assertEqual(ct["cost_tax"], 1)
        self.assertEqual(ct["cost_wage_tax"], 0)

        ct = tc.costs_at_term(1)
        self.assertEqual(ct["cost_init"], 00)
        self.assertEqual(ct["cost_maintenance"], 10)
        self.assertEqual(ct["cost_tax"], 1)
        self.assertEqual(ct["cost_wage_tax"], 0)

        ct = tc.costs_at_term(2)
        self.assertEqual(ct["cost_init"], 00)
        self.assertEqual(ct["cost_maintenance"], 10)
        self.assertEqual(ct["cost_tax"], 1)
        self.assertEqual(ct["cost_wage_tax"], 15)

    def test_category_costs(self):
        tc = TermCosts()
        cc = tc.cost_categories_at_term(0)
        self.assertEqual(cc["capex"], 100)
        self.assertEqual(cc["opex"], 0)
        self.assertEqual(cc["tax"], 1)

        tc = TermCosts()
        cc = tc.cost_categories_at_term(2)
        self.assertEqual(cc["capex"], 0)
        self.assertEqual(cc["opex"], 25)
        self.assertEqual(cc["tax"], 16)

    def test_econ(self):
        tc = TermCosts()
        Comp2.default_cost("comp1", 50)
        c2 = Comp2(cost_per_item=10)
        er = EconRecursive(comp1=tc, comp2=c2)
        # inkw = {'econ_term_length':[1,5,10,50],'econ_discount_rate':[0,0.01,0.05,0.1]}
        inkw = {}
        er.run(revert_last=False, **inkw)

        d = er.data_dict

        self.assertEqual(d["econ.lifecycle.term_cost"], 161)
        self.assertEqual(d["econ.summary.total_cost"], 161)
        self.assertEqual(d["combine_cost"], 161)
        self.assertEqual(d["sub_items_cost"], 161)

    def test_econ_array(self):
        tc = TermCosts()
        Comp2.default_cost("comp1", 50)
        c2 = Comp2(cost_per_item=10)
        er = EconRecursive(comp1=tc, comp2=c2)
        inkw = {"econ.term_length": [0, 5, 10, 15], "econ.fixed_output": 1}
        # inkw = {}
        er.run(revert_last=False, revert_every=False, **inkw)

        df = er.dataframe
        tc = (
            df["econ_summary_total_cost"]
            == np.array([161.0, 220.0, 305.0, 390.0])
        ).all()
        self.assertTrue(tc)


class TestEconomicsAccounting(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        Comp1.reset_cls_costs()
        Comp2.reset_cls_costs()
        EconRecursive.reset_cls_costs()

    def test_econ_defaults(self):
        Comp1.default_cost("norm", Norm(cost_per_item=10))
        Comp2.default_cost("comp1", Comp1(cost_per_item=5))
        EconRecursive.default_cost("comp2", Comp2(cost_per_item=3))
        er = EconRecursive(cost_per_item=50)
        # c2 = er.comp2
        self.assertEqual(er.combine_cost, 78)
        self.assertEqual(er.sub_items_cost, 28)

        er.run()
        d = er.data_dict
        self.assertEqual(78, d["econ.summary.total_cost"])
        self.assertEqual(78, d["econ.lifecycle.annualized.term_cost"])
        self.assertEqual(78, d["econ.lifecycle.annualized.levalized_cost"])
        self.assertEqual(78, d["econ.lifecycle.term_cost"])

    def test_recursive_null(self, ANS=75):
        Comp1.default_cost("norm", 5)
        Comp2.default_cost("comp1", 10)
        EconRecursive.default_cost("comp1", 3)
        EconRecursive.default_cost("comp2", 7)
        er = EconRecursive(cost_per_item=50, comp1=None, comp2=None)
        er.run()
        self.assertEqual(er.combine_cost, ANS)
        self.assertEqual(er.econ.combine_cost, ANS)
        self.assertEqual(er.comp1, None)
        self.assertEqual(er.comp2, None)

        d = er.data_dict
        self.assertEqual(ANS, d["econ.combine_cost"])
        self.assertEqual(ANS, d["combine_cost"])
        self.assertEqual(3, d["econ.comp1.cost.item_cost"])
        self.assertEqual(7, d["econ.comp2.cost.item_cost"])

    def test_recursive_comp2(self, ANS=80):
        Comp1.default_cost("norm", 5)
        Comp2.default_cost("comp1", 10)
        EconRecursive.default_cost("comp1", Comp1(cost_per_item=3))
        EconRecursive.default_cost("comp2", Comp2(cost_per_item=7))
        er = EconRecursive(cost_per_item=50, comp1=None, comp2=None)
        er.run()
        self.assertEqual(er.combine_cost, ANS)
        self.assertEqual(er.econ.combine_cost, ANS)

        d = er.data_dict
        self.assertEqual(ANS, d["econ.combine_cost"])
        self.assertEqual(ANS, d["combine_cost"])
        self.assertEqual(3, d["econ.comp1.cost.item_cost"])
        # self.assertEqual(10,d['econ_comp2.combine_cost'])

    def test_recursive_all(self, ANS=80):
        # FIXME: missing norm cost of -5

        Comp1.default_cost("norm", 5)
        Comp2.default_cost("comp1", 10)
        EconRecursive.default_cost("comp1", Comp1(cost_per_item=3))
        EconRecursive.default_cost("comp2", Comp2(cost_per_item=7))
        er = EconRecursive(cost_per_item=50)
        er.run()
        self.assertEqual(er.combine_cost, ANS)
        self.assertEqual(er.econ.combine_cost, ANS)

        d = er.data_dict
        self.assertEqual(ANS, d["econ.combine_cost"])
        self.assertEqual(ANS, d["combine_cost"])
        # self.assertEqual(5,d['econ_comp1.combine_cost'])
        # self.assertEqual(10,d['econ_comp2.combine_cost'])
        self.assertEqual(10, d["econ.comp2.comp1.cost.item_cost"])
        self.assertEqual(5, d["econ.comp1.norm.cost.item_cost"])


class TestCostModel(unittest.TestCase):
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
        c2 = Comp2(cost_per_item=0, comp1=c1)
        self.assertEqual(c2.combine_cost, 0)

        c1 = Comp1(cost_per_item=5)
        c2 = Comp2(cost_per_item=5, comp1=c1)
        self.assertEqual(c2.combine_cost, 10)

        c1 = Comp1(cost_per_item=10)
        c2 = Comp2(cost_per_item=10, comp1=c1)
        self.assertEqual(c2.combine_cost, 20)

    def test_default(self):
        c1 = Comp1(cost_per_item=0)
        self.assertEqual(c1.combine_cost, 0)

        Comp1.default_cost("norm", 10)
        c1 = Comp1(cost_per_item=0)
        self.assertEqual(c1.combine_cost, 10)

        c1 = Comp1(cost_per_item=0)
        c1.custom_cost("norm", 20)
        self.assertEqual(c1.combine_cost, 20)
        self.assertEqual(Comp1._slot_costs["norm"], 10)

    def test_ref_loop(self):
        # Comp1.default_cost('norm',10)
        c2 = Comp2(cost_per_item=5)
        c1 = Comp1(cost_per_item=5, norm=c2)
        c2.comp1 = c1
        self.assertEqual(c1.combine_cost, 10)
        self.assertEqual(c1.sub_items_cost, 10)

    def test_override(self):
        Comp1.default_cost("norm", 10)
        # c2 = Comp2(cost_per_item=5)
        c1 = Comp1(cost_per_item=5)  # ,norm=c2)
        self.assertEqual(c1.combine_cost, 15)
        self.assertEqual(c1.sub_items_cost, 10)

        # Comp1.reset_cls_costs()
        c2 = Comp2(cost_per_item=15)
        c1 = Comp1(cost_per_item=15, norm=c2)
        self.assertEqual(c1.combine_cost, 40)
        self.assertEqual(c1.sub_items_cost, 25)

    def test_defaults(self):
        # Comp1.reset_cls_costs()
        Comp1.default_cost("norm", Norm(cost_per_item=10))
        Comp2.default_cost("comp1", Comp1(cost_per_item=5))
        c2 = Comp2(cost_per_item=3)
        self.assertEqual(c2.combine_cost, 18)
        self.assertEqual(c2.sub_items_cost, 15)


class TestFanSystemDataFrame(unittest.TestCase):
    def test_dataframe(self):
        # Create the FanSystem instance
        fs = FanSystem(fan=Fan(), motor=Motor(), base=MetalBase())
        fs.run(
            **{
                "econ.term_length": [1, 10, 20, 50],
                "fan.V": [1, 5, 10],
                "fan.area": [1, 5, 10],
            },
            base=[None, MetalBase()]
        )

        # Get the dataframe
        df_complete = fs.last_context.dataframe

        dfc = df_complete
        match = (
            dfc["fan_blade_cost"] + dfc["motor_motor_cost"]
            == dfc["econ_lifecycle_category_capex"]
        ).all()
        self.assertTrue(match)

        match = (
            df_complete["fan_area"] * df_complete["fan_v"]
            == df_complete["fan_volumetric_flow"]
        ).all()
        self.assertTrue(match)


# TODO: get cost accounting working for ComponentIter components
# @forge
# class CompGroup(ComponentIterator):
#     pass
# # @forge
# # class CostGroup(CompGroup,CostModel):
# #     pass

# @forge
# class EconWide(System):
#     econ = Slot.define(Economics)
#     comp_set = Slot.define_iterator(CompGroup,wide=True)
#
# @forge
# class EconNarrow(System):
#     econ = Slot.define(Economics)
#     comp_set = Slot.define_iterator(CompGroup,wide=False)

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
#         self.assertEqual(d['econ_comp_set.item_cost'], ew.comp_set.current.item_cost)
#         self.assertEqual(ew.econ.total_cost,total)
#         self.assertEqual(ew.econ.total_cost,total)


if __name__ == "__main__":
    unittest.main()
