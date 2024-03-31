import unittest

from engforge.configuration import forge
from engforge.system import System
from engforge.components import Component
from engforge.attr_dynamics import Time
from engforge.attr_solver import Solver
from engforge.attr_signals import SIGNAL
from engforge.attr_slots import Slot
from engforge.properties import *
from engforge.problem_context import ProblemExec

import attrs


@forge
class MockComponent(Component):
    input: float = attrs.field(default=0)
    # use to test input from system
    aux: float = attrs.field(default=0)

    comp = Slot.define(Component, none_ok=True)

    @system_property
    def output(self) -> float:
        return self.input * self.input


def limit_max(system,prb):
    return 4 - system.input


@forge
class MockSystem(System):
    input: float = attrs.field(default=0)
    output: float = attrs.field(default=0)

    in2: float = attrs.field(default=1)

    x: float = attrs.field(default=0)

    # move system input to component
    pre_sig = SIGNAL.define("comp.input", "input", mode="pre")
    # move component output to system
    post_sig = SIGNAL.define("output", "comp.output", mode="both")

    Solver.constraint_equality("in_out_diff")
    Solver.constraint_equality("in2out2")

    var_in =Solver.declare_var("input")
    var_in.add_var_constraint(kind="min", value=0)
    var_in.add_var_constraint(kind="max", value=1)

    
    sol2 =Solver.declare_var("in2")
    sol2.add_var_constraint(kind="max", value=limit_max)
    sol2.add_var_constraint(kind="min", value=0)

    comp = Slot.define(MockComponent)

    @system_property
    def in_out_diff(self) -> float:
        # Should solve= 0 when in2=input^2
        return self.input + 1 - self.output - self.in2 * 0.1

    @system_property
    def in2out2(self) -> float:
        return self.in2 - self.input * 2 + self.comp.aux / 10

    @system_property
    def dxdt(self) -> float:
        # dxdt = (in - in**2)/1000.0
        # Should converge to 1,0,-1
        ans = (self.input - self.in2) / 2
        if ans < 0 and self.input < 0:
            return -1 * ans
        return ans


class TestComposition(unittest.TestCase):
    def setUp(self) -> None:
        self.comp2 = MockComponent()
        self.comp = MockComponent(comp=self.comp2)
        self.system = MockSystem(input=5, comp=self.comp)

    def test_signals(self):
        self.system.post_sig.signal.mode = "post"  # usually its both

        # Pre Signals Test
        sysstart = self.system.input
        compstart = self.system.comp.input
        self.assertNotEqual(sysstart, compstart)  # no signals transfering input

        with ProblemExec(self.system,{},post_exec=True,pre_exec=True) as pbx:
            sysend = self.system.input
            compend = self.system.comp.input
            self.assertEqual(sysend, compend)  # signals should work as defined
            self.assertEqual(sysstart, sysend)  # input should remain the same
            self.assertNotEqual(compstart, compend)  # input on comp changes

            sysstart = self.system.output
            compstart = self.system.comp.output
            self.assertNotEqual(sysstart, compstart)  # they aren't aligned
            
            #preserve changes
            pbx.exit_with_state()

        #now post signals should be applied
        sysend = self.system.output
        compend = self.system.comp.output
        self.assertEqual(sysend, compend)  # signals should work
        self.assertEqual(compstart, compend)  # output should change
        self.assertNotEqual(sysstart, sysend)  # output should change

    def test_input_and_run(self):
        self.system.run(**{"comp.aux": 5, "comp.comp.aux": 6})
        self.assertEqual(len(self.system.TABLE), 1, f"wrong run config")
        self.assertEqual(set(self.system.dataframe["comp_aux"]), set([5]))
        self.assertEqual(set(self.system.dataframe["comp_comp_aux"]), set([6]))

        # internal storage
        # self.assertEqual(set(self.system.comp.dataframe["aux"]), set([5]))
        # self.assertEqual(set(self.system.comp.comp.dataframe["aux"]), set([6]))

#         # solver constraints checking
#         input = self.system.dataframe["input"][0]
#         self.assertGreaterEqual(input, -1e3)  # min protection
#         self.assertLessEqual(input, 1)  # max protection
# 
#         in2 = self.system.dataframe["in2"][0]
#         self.assertGreaterEqual(in2, -1e3)  # min protection
#         self.assertLessEqual(in2, 4 - input)  # max protection
