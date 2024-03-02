import unittest
from engforge.system import System
from engforge.components import Component, forge
from engforge.attr_slots import Slot
from engforge.attr_dynamics import Time
from engforge.attr_signals import Signal
from engforge.attr_solver import Solver
from engforge.properties import system_property
from engforge.dynamics import DynamicsMixin,GlobalDynamics

import numpy as np

Fun_size = lambda sys,info: sys.volume
Fun_eff = lambda sys: sys.cost / sys.volume
Fun_minz = lambda sys,*a: max(sys.x, sys.y)
Fun_em = lambda sys: sys.max_length - sys.combine_length
Fun_cm = lambda sys,info: sys.budget - sys.cost

@forge(auto_attribs=True)
class SolvComp(System,GlobalDynamics):
    x: float = 1.0
    y: float = 1.0
    z: float = 1.0

    dynamic_state_parms: list = ["x", "y", "z"]
    # this should be energy neutral 🤞
    static_A = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    
    zspeed = Time.integrate("z", "zdot", mode="euler")

    base_cost: float = 10
    cost_x: float = 10
    cost_y: float = 20
    cost_z: float = 5

    budget: float = 100
    max_length: float = 50
    volume_goal: float = 10

    Xvar = Solver.declare_var("x", combos="x")
    Yvar = Solver.declare_var("y", combos="y")
    Zvar = Solver.declare_var("z", combos="z")
    Xvar.add_var_constraint(0.1, kind="min")
    Yvar.add_var_constraint(0.1, kind="min")
    Zvar.add_var_constraint(Fun_minz, kind="min")

    #Constraints by function
    #sym = Solver.equality_constraint("x",'y',combos='z_sym_eq')

    #costA = Solver.ineq_con('budget','cost',combos='ineq_cost')
    costF = Solver.ineq_con(Fun_cm,combos='fun_cost')
    #costP = Solver.ineq_con('cost_margin',combos='prop_cost')

    #Constraints by length
    #lenA = Solver.ineq_con('max_length','combine_length',combos='ineq_length')
    #lenF = Solver.ineq_con(Fun_em,combos='fun_length')
    lenP = Solver.ineq_con('edge_margin',combos='prop_length')

    #Objectives
    size_goal = Solver.objective('goal',combos='prop_goal',kind='min')

    size = Solver.objective('volume',combos='prop_size',kind='max')
    #sizeF = Solver.objective(Fun_size,combos='fun_size',kind='max')

    eff = Solver.objective('cost_to_volume',combos='prop_eff',kind='min')
    #effF = Solver.objective(Fun_eff,combos='fun_eff',kind='min')

    @system_property
    def zdot(self)->float:
        return self.x - self.y

    @system_property
    def combine_length(self) -> float:
        return (self.x + self.y + self.z) * 4

    @system_property
    def cost_margin(self) -> float:
        dv = self.budget - self.cost
        #print(f'cost_margin: {dv}')
        return dv

    @system_property
    def edge_margin(self) -> float:
        dv = self.max_length - self.combine_length
        #print(f'edge_margin: {dv}')
        return dv
    
    @system_property
    def goal(self) -> float:
        dv = self.volume_goal - self.volume
        return (dv)**2


    @system_property
    def volume(self) -> float:
        return self.x * self.y * self.z

    @system_property
    def cost(self) -> float:
        edge_cost = self.x * 4 * self.cost_x
        edge_cost =edge_cost+ self.y * 4 * self.cost_y
        edge_cost =edge_cost+ self.z * 4 * self.cost_z
        return self.base_cost + edge_cost

    @system_property
    def cost_to_volume(self) -> float:
        return self.cost / self.volume
    
class SolverTest(unittest.TestCase):

    def setUp(self) -> None:
        self.sc = SolvComp()

    def test(self):
        self.sc.execute()