from engforge.system import System
from engforge.components import Component, forge
from engforge.attr_slots import Slot
from engforge.attr_dynamics import Time
from engforge.attr_signals import Signal
from engforge.attr_solver import Solver
from engforge.properties import system_property
from engforge.dynamics import DynamicsMixin,GlobalDynamics

import numpy as np
import unittest

@forge(auto_attribs=True)
class DynamicComponent(Component, DynamicsMixin):
    dynamic_state_parms: list = ["x", "v"]
    x: float = 1
    v: float = 0

    b: float = 0.1
    K: float = 10
    M: float = 100

    x0: float = 0.5

    nonlinear: bool = False
    Fext: float = 0

    acc =Solver.declare_var("x0")
    no_load = Solver.equality_constraint("a")

    def create_state_matrix(self, **kwargs) -> np.ndarray:
        """creates the state matrix for the system"""
        return np.array([[0, 1], [-self.K / self.M, -self.b / self.M]])

    def create_state_constants(self, **kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.array([0, self.K * self.x0 / self.M])

    def update_state_constants(self, t, F, X) -> np.ndarray:
        """override"""
        F = F.copy()
        F[-1] = (self.K * self.x0 - self.Fext) / self.M
        return F

    @system_property
    def a(self) -> float:
        return (-self.v * self.b - (self.x - self.x0) * self.K) / self.M

@forge(auto_attribs=True)
class TransientSys(Component):
    x: float = 0
    v: float = 0
    a: float = 0

    speed = Time.integrate("x", "v", mode="euler")
    accel = Time.integrate("v", "a", mode="euler")

@forge(auto_attribs=True)
class DynamicSystem(System, GlobalDynamics):
    dynamic_state_parms: list = ["x", "v"]

    x: float = 0
    v: float = 0
    a: float = 0

    Force: float = 0.0
    Damp: float = 10
    Mass: float = 100.0
    K: float = 20

    Fref: float = 10
    omega: float = 1

    comp = Slot.define(DynamicComponent)
    trns = Slot.define(TransientSys)

    sig = Signal.define("trns.a", "spring_accel")
    fig = Signal.define("comp.Fext", "Force")
    slv =Solver.declare_var( "Force")
    slv = Solver.declare_var("delta_a")

    nonlinear: bool = True

    @system_property
    def spring_accel(self) -> float:
        # print(self.comp.v,self.comp.x,self.comp.a)
        return (
            -self.comp.v * self.comp.b
            - (self.comp.x - self.comp.x0) * self.comp.K
        ) / self.comp.M

    @system_property
    def delta_a(self) -> float:
        return (
            self.Fref * np.cos(self.omega * self.time)
            - self.Force
            + self.v * self.Damp
            - self.K * self.x
        ) / self.Mass - self.spring_accel

    def create_state_matrix(self, *args, **kwargs) -> np.ndarray:
        """creates the state matrix for the system"""
        return np.array(
            [[0, 1.0], [-self.K / self.Mass, -1 * self.Damp / self.Mass]]
        )

    def create_state_constants(self, *args, **kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.array([0, self.Force / self.Mass])

    def update_state_nonlinear(self, *args, **kwargs) -> np.ndarray:
        """creates the state matrix for the system"""
        return np.array(
            [[0, 1.0], [-self.K / self.Mass, -1 * self.Damp / self.Mass]]
        )

    def update_state_constants(self, *args, **kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.array([0, self.Force / self.Mass])

class TestDynamics(unittest.TestCase):

    def test_dynamics(self):
        dc = DynamicComponent()
        dc.create_dynamics()

        ds = DynamicSystem(comp=dc)
        ds.create_dynamics()
        # ds.update_dynamics()
        ds.collect_dynamic_refs()
        # ds2 = ds.copy_config_at_state()
        sr = ds.collect_solver_refs()

        min_kw = {"normalize": np.array([1 / 1000])}
        sim, df = ds.simulate(0.01, 30, run_solver=True, return_system=True)
        ax = df.plot("time", ["x", "comp_x", "trns_x", "comp_x0"])
        # ax.set_ylim(-1,5)
        ax2 = ax.twinx()
        ax2.plot(df.time, df.a, "r--", label="acl")
        ax2.plot(df.time, df["trns_a"], "b--", label="trans_acl")
        ax2.plot(df.time, df["spring_accel"], "c--", label="spring_acl")
        ax2.legend()