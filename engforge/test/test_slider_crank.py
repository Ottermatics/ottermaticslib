import unittest
import numpy as np

from engforge.test.solver_testing_components import *


class TestSliderCrank(unittest.TestCase):
    def test_optimization(self):
        sc = SliderCrank()

        out = sc.run(combos="design", slv_vars="*")

    def test_design_multiobj(self):
        sc = SliderCrank()

        out = sc.run(
            combos="design,*goal,*sym",
            slv_vars="*",
            k_spring=10,
            revert_last=False,
            revert_every=False,
        )

        out = out["output"][0]
        Yobj = set(out["Yobj"])
        self.assertEqual(set(("sym_slv",)), Yobj)

        self.assertAlmostEqual(sc.dx_goal, 0.0, places=3)
        self.assertAlmostEqual(sc.ds_goal, 0.0, places=3)


# BUG:
# ======================================================================
# FAIL: test_design_objs (engforge.test.test_slider_crank.TestSliderCrank)
# ----------------------------------------------------------------------
# Traceback (most recent call last):
#   File "/mnt/c/Users/Sup/Ottermatics Dropbox/Ottermatics/engforge/engforge/test/test_slider_crank.py", line 34, in test_design_objs
#     self.assertEqual(set(('sym_slv',)),Yobj)
# AssertionError: Items in the second set but not the first:
# 'cost_slv'
#     def test_design_objs(self):
#         sc = SliderCrank()
#
#         out = sc.run(combos='design,*goal,*sym,*cost',slv_vars='*',k_spring=10,revert_last=False,revert_every=False)
#
#         out = out['output'][0]
#         Yobj = set(out['Yobj'])
#         self.assertEqual(set(('sym_slv',)),Yobj)
#
#         self.assertAlmostEqual(sc.dx_goal,0.0,places=3)
#         self.assertAlmostEqual(sc.ds_goal,0.0,places=3)
