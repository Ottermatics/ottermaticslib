import unittest
from engforge.system import System
from engforge.configuration import Configuration
from engforge.components import Component, forge, SolveableInterface
from engforge.solveable import SolveableMixin
from engforge.attr_slots import Slot
from engforge.attr_dynamics import Time
from engforge.attr_signals import Signal
from engforge.attr_solver import Solver
from engforge.properties import system_property
from engforge.dynamics import DynamicsMixin,GlobalDynamics

import attrs
import numpy as np

Fun_size = lambda sys,*a,**kw: sys.volume
Fun_eff = lambda sys,*a,**kw: sys.cost / sys.volume
Fun_minz = lambda sys,*a,**kw: max(sys.x, sys.y)
Fun_em = lambda sys,*a,**kw: sys.max_length - sys.combine_length
Fun_cm = lambda sys,*a,**kw: sys.budget - sys.cost

cm = -0.01

@forge(auto_attribs=True)
class SpaceMixin(SolveableInterface):
    
    x: float = attrs.field(default=1.0)
    y: float = attrs.field(default=1.0)
    z: float = attrs.field(default=1.0)

    base_cost: float = 0
    cost_x: float = 10
    cost_y: float = 10
    cost_z: float = 10

    budget: float = 100
    max_length: float = 50
    volume_goal: float = 10

    dynamic_state_parms: list = ["x", "y", "z"]

    static_A = np.array([[cm, 1, -1], [-1, cm, 1], [1, -1, cm]])

    #size constraints
    Xvar = Solver.declare_var("x", combos="x")
    Yvar = Solver.declare_var("y", combos="y")
    Zvar = Solver.declare_var("z", combos="z")
    Xvar.add_var_constraint(0.1, kind="min",combos=['min_len'])
    Yvar.add_var_constraint(0.1, kind="min",combos=['min_len'])
    Zvar.add_var_constraint(0.1, kind="min",combos=['min_len'],active=True)
    Zvar.add_var_constraint(Fun_minz, kind="min",combos=['len_fun_z'],active=False)

    #Constraints by function
    sym = Solver.equality_constraint("x",'y',combos='z_sym_eq',active=False)

    costA = Solver.ineq_con('budget','cost',combos='ineq_cost')
    costF = Solver.ineq_con(Fun_cm,combos='fun_cost',active=False)
    costP = Solver.ineq_con('cost_margin',combos='prop_cost',active=False)

    #Constraints by length
    lenA = Solver.ineq_con('max_length','combine_length',combos='ineq_length')
    lenF = Solver.ineq_con(Fun_em,combos='fun_length',active=False)
    lenP = Solver.ineq_con('edge_margin',combos='prop_length',active=False)

    #Objectives
    size_goal = Solver.objective('goal',combos='prop_goal',kind='min',active=False)

    size = Solver.objective('volume',combos='prop_size',kind='max',active=True)
    #sizeF = Solver.objective(Fun_size,combos='fun_size',kind='max')

    eff = Solver.objective('cost_to_volume',combos='prop_eff',kind='min',active=True)
    #effF = Solver.objective(Fun_eff,combos='fun_eff',kind='min')

    @system_property
    def combine_length(self) -> float:
        return (abs(self.x) + abs(self.y) + abs(self.z)) * 4

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
        return abs(self.x * self.y * self.z)

    @system_property
    def cost(self) -> float:
        edge_cost = abs(self.x) * 4 * self.cost_x
        edge_cost =edge_cost+ abs(self.y) * 4 * self.cost_y
        edge_cost =edge_cost+ abs(self.z) * 4 * self.cost_z
        return self.base_cost + edge_cost

    @system_property
    def cost_to_volume(self) -> float:
        if self.volume == 0:
            return 0
        return self.cost / self.volume 

@forge(auto_attribs=True)
class SolvComp(Component,SpaceMixin):
    # this should be energy neutral 🤞
    #zspeed = Time.integrate("z", "zdot", mode="euler")

    base_cost: float = attrs.field(default=10)
    cost_x: float = attrs.field(default=10)
    cost_y: float = attrs.field(default=20)
    cost_z: float = attrs.field(default=5)

    budget: float = attrs.field(default=300)
    max_length: float = attrs.field(default=100)
    volume_goal: float = attrs.field(default=100)



    

@forge(auto_attribs=True)
class SolvSys(System,GlobalDynamics,SpaceMixin):
    
    comp = Slot.define(SolvComp)

    x: float = 1
    y: float = 1
    z: float = 1
    

    goal_vol_frac: float = 0.5

    total_budget:float = 400
    total_length:float = 400

    sys_budget = Solver.ineq_con('total_budget','system_cost',combos='total_budget')

    sys_length = Solver.ineq_con('total_length','system_length',combos='total_length')

    volfrac = Solver.eq_con('goal_vol_frac','vol_frac',combos='vol_frac_eq',active=False)

    obj = Solver.objective('total_volume',combos='volume',kind='max')
    hght = Solver.objective('total_height',combos='height',kind='max')

    x_lim = Solver.ineq_con('x','comp.x',combos='tierd_top')
    y_lim = Solver.ineq_con('y','comp.y',combos='tierd_top')

    @system_property
    def vol_frac(self)->float:
        cv = self.comp.volume
        return cv / (self.volume+cv)
    
    @system_property
    def total_volume(self)->float:
        return self.volume + self.comp.volume
    
    @system_property
    def total_height(self)->float:
        return max(self.z,0) + max(self.comp.z,0)

    @system_property
    def system_cost(self) -> float:
        return self.cost + self.comp.cost
    
    @system_property
    def system_length(self) -> float:
        return self.combine_length + self.comp.combine_length




   


    
class SingleCompSolverTest(unittest.TestCase):
    inequality_min = -1E-6
    def setUp(self) -> None:
        self.sc = SolvSys()

    def test_exec_results(self):
        extra = dict()
        o = self.sc.execute(**extra)
        res = self.test_results(o)
        self.test_selection(o,res,extra)

    def test_selective_exec(self):
        extra = dict(combos='*fun*,goal',ign_combos=['z_sym_eq','*size*','*eff*'])
        o = self.sc.execute(**extra)
        res = self.test_results(o)
        self.test_selection(o,res,extra)

    def test_runmtx(self):
        extra = dict(budget=[100,500,1000],max_length=[50,100,200])
        o = self.sc.run(**extra)
        res = self.test_results(o)
        df = self.test_dataframe(o)
        self.test_selection(o,res,extra)  

    def test_selective_runmtx(self):
        extra = dict(budget=[100,500,1000],max_length=[50,100,200],combos='*fun*,goal',ign_combos=['z_sym_eq','*size*','*eff*'])        
        o = self.sc.run(**extra)
        res = self.test_results(o)
        df = self.test_dataframe(o)
        self.test_selection(o,res,extra)

    def test_run_wildcard(self):
        extra = dict(combos='*fun*,goal',ign_combos='*',budget=[100,500,1000],max_length=[50,100,200])
        o = self.sc.run(**extra)
        res = self.test_results(o)
        df = self.test_dataframe(o)
        self.test_selection(o,res,extra)               

    #Checks
    def test_dataframe(self,results)->dict:
        #print(results)
        raise NotImplementedError("#FIXME!")

    def test_selection(self,results,df_res,extra):
        #print(results,df_res,extra)
        raise NotImplementedError("#FIXME!")

    def test_results(self,results):
        Ycon = results['Ycon']
        costmg = {k:v for k,v in Ycon.items() if 'cost' in k}
        lenmg = {k:v for k,v in Ycon.items() if 'len' in k}

        #Test input equivalence of methods
        self.assertEqual(len(set(costmg.values())),1)
        self.assertEqual(len(set(lenmg.values())),1)

        #Test inequality constraints
        yvals = (v >= self.inequality_min for v in Ycon.values())
        self.assertTrue(all(yvals))
