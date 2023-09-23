import unittest

from ottermatics.configuration import otterize
from ottermatics.system import System
from ottermatics.components import Component
from ottermatics.solver import SOLVER, TRANSIENT
from ottermatics.signals import SIGNAL
from ottermatics.slots import SLOT
from ottermatics.properties import *

import attrs


@otterize
class MockComponent(Component):
    input: float = attrs.field(default=0)
    # use to test input from system
    aux: float = attrs.field(default=0)

    comp = SLOT.define(Component, none_ok=True)

    @system_property
    def output(self) -> float:
        return self.input * self.input


def limit_max(system):
    return 4 - system.input


@otterize
class MockSystem(System):
    input: float = attrs.field(default=0)
    output: float = attrs.field(default=0)

    in2: float = attrs.field(default=1)

    x: float = attrs.field(default=0)

    # move system input to component
    pre_sig = SIGNAL.define("comp.input", "input", mode="pre")
    # move component output to system
    post_sig = SIGNAL.define("output", "comp.output", mode="both")

    solver = SOLVER.define("in_out_diff", "input")
    solver.add_constraint("min", 0)
    solver.add_constraint("max", 1)
    # solver.addConstraint('MAX','limit_max')
    # solver.addConstraint('MIN', 0)
    sol2 = SOLVER.define("in2out2", "in2")
    sol2.add_constraint("max", limit_max)
    sol2.add_constraint("min", 0)

    # trans = TRANSIENT.define("x", "dxdt")

    comp = SLOT.define(MockComponent)

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

        # FIXME: turn off
        # self.comp2.log_level = 1
        # self.comp2.resetLog()
        # self.comp.log_level = 1
        # self.comp.resetLog()
        # self.system.log_level = 1
        # self.system.resetLog()

    def test_signals(self):
        self.system.post_sig.signal.mode = "post"  # usually its both

        # Pre Signals Test
        sysstart = self.system.input
        compstart = self.system.comp.input
        self.assertNotEqual(sysstart, compstart)  # no signals transfering input

        # Use The Signals
        self.system.pre_execute()

        sysend = self.system.input
        compend = self.system.comp.input
        self.assertEqual(sysend, compend)  # signals should work as defined
        self.assertEqual(sysstart, sysend)  # input should remain the same
        self.assertNotEqual(compstart, compend)  # input on comp changes

        # Do POST
        sysstart = self.system.output
        compstart = self.system.comp.output
        self.assertNotEqual(sysstart, compstart)  # they aren't aligned

        # Use The Signals
        self.system.post_execute()

        sysend = self.system.output
        compend = self.system.comp.output
        self.assertEqual(sysend, compend)  # signals should work
        self.assertEqual(compstart, compend)  # output should change
        self.assertNotEqual(sysstart, sysend)  # output should change

    def test_input_and_run(self):
        self.system.run(
            **{"comp.aux": 5, "comp.comp.aux": 6}  # , dt=0.1, endtime=0.1
        )
        self.assertEqual(len(self.system.TABLE), 1, f"wrong run config")
        self.assertEqual(set(self.system.dataframe["comp.aux"]), set([5]))
        self.assertEqual(set(self.system.dataframe["comp.comp.aux"]), set([6]))

        #internal storage
        # self.assertEqual(set(self.system.comp.dataframe["aux"]), set([5]))
        # self.assertEqual(set(self.system.comp.comp.dataframe["aux"]), set([6]))

        # solver constraints checking
        input = self.system.dataframe["input"][0]
        self.assertGreaterEqual(input, -1e3)  # min protection
        self.assertLessEqual(input, 1)  # max protection

        in2 = self.system.dataframe["in2"][0]
        self.assertGreaterEqual(in2, -1e3)  # min protection
        self.assertLessEqual(in2, 4 - input)  # max protection
