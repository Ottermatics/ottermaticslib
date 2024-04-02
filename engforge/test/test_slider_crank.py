import unittest
import numpy as np

from engforge.test.solver_testing_components import *


class TestSliderCrank(unittest.TestCase):

    def test_optimization(self):
        sc = SliderCrank()

        out = sc.run(combos='design',slv_vars='*')