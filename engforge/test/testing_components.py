from engforge.system import System
from engforge.configuration import Configuration
from engforge.components import Component, forge, SolveableInterface
from engforge.solveable import SolveableMixin
from engforge.attr_signals import Signal
from engforge.attr_slots import Slot
from engforge.attr_dynamics import Time
from engforge.attr_signals import Signal
from engforge.attr_solver import Solver
from engforge.properties import system_property
from engforge.dynamics import DynamicsMixin,GlobalDynamics
from engforge.logging import LoggingMixin

import attrs
import numpy as np

class TestCompLog(LoggingMixin): pass
log = TestCompLog()

Fun_size = lambda sys,*a,**kw: sys.volume
Fun_eff = lambda sys,*a,**kw: sys.cost / sys.volume
Fun_minz = lambda sys,*a,**kw: max(sys.x, sys.y)
Fun_em = lambda sys,*a,**kw: sys.max_length - sys.combine_length
Fun_cm = lambda sys,*a,**kw: sys.budget - sys.cost

def wrap_f(function):
    def wrapper(*args, **kwargs):
        #log.info(f'calling {function.__name__}| {args} | {kwargs}')
        return function(*args, **kwargs)
    return wrapper

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

    static_A = np.array([[cm, 1, -1], [-1, cm, 1], [1, -1, cm]])

    #size constraints
    Xvar = Solver.declare_var("x", combos="x",active=True)
    Yvar = Solver.declare_var("y", combos="y",active=True)
    Zvar = Solver.declare_var("z", combos="z",active=True)
    Xvar.add_var_constraint(0.1, kind="min",combos=['min_len'])
    Yvar.add_var_constraint(0.1, kind="min",combos=['min_len'])
    Zvar.add_var_constraint(0.1, kind="min",combos=['min_len'])
    #Zvar.add_var_constraint(Fun_minz, kind="min",combos=['len_fun_z'])

    #Constraints by function
    sym = Solver.equality_constraint("x",'y',combos='sym',active=False)

    costA = Solver.ineq_con('budget','cost',combos='ineq_cost',active=False)
    costF = Solver.ineq_con(wrap_f(Fun_cm),combos='ineq_cost',active=False)
    costP = Solver.ineq_con('cost_margin',combos='ineq_cost',active=False)

    #Constraints by length
    lenA = Solver.ineq_con('max_length','combine_length',combos='ineq_length',active=False)
    lenF = Solver.ineq_con(wrap_f(Fun_em),combos='ineq_length',active=False)
    lenP = Solver.ineq_con('edge_margin',combos='ineq_length',active=False)

    #Objectives
    size_goal = Solver.objective('goal',combos='prop_goal',kind='min',active=False)

    size = Solver.objective('volume',combos='obj_size',kind='max',active=False)
    sizeF = Solver.objective(wrap_f(Fun_size),combos='obj_size',kind='max',active=False)

    eff = Solver.objective('cost_to_volume',combos='obj_eff',kind='min',active=False)
    effF = Solver.objective(wrap_f(Fun_eff),combos='obj_eff',kind='min',active=False)

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
class SolvComp(Component,SpaceMixin,DynamicsMixin):
    # this should be energy neutral 🤞
    
    dynamic_state_parms: list = ["x", "y", "z"]

    base_cost: float = attrs.field(default=10)
    cost_x: float = attrs.field(default=10)
    cost_y: float = attrs.field(default=20)
    cost_z: float = attrs.field(default=5)

    budget: float = attrs.field(default=300)
    max_length: float = attrs.field(default=100)
    volume_goal: float = attrs.field(default=100)



    

@forge(auto_attribs=True)
class SolvSys(System,SpaceMixin,GlobalDynamics):
    
    dynamic_state_parms: list = ["x", "y", "z"]

    comp = Slot.define(SolvComp)

    x: float = attrs.field(default=0.5)
    y: float = attrs.field(default=0.5)
    z: float = attrs.field(default=0.5)

    cost_x: float = attrs.field(default=5)
    cost_y: float = attrs.field(default=15)
    cost_z: float = attrs.field(default=10)

    goal_vol_frac: float = 0.5

    sys_budget = Solver.ineq_con('total_budget','system_cost',combos='total_budget')

    sys_length = Solver.ineq_con('total_length','system_length',combos='total_length')

    volfrac = Solver.eq_con('goal_vol_frac','vol_frac',combos='vol_frac_eq',active=False)

    obj = Solver.objective('total_volume',combos='volume',kind='max')
    hght = Solver.objective('total_height',combos='height',kind='max')

    x_lim = Solver.ineq_con('x','comp.x',combos='tierd_top',active=False)
    y_lim = Solver.ineq_con('y','comp.y',combos='tierd_top',active=False)

    sig_x = Signal.define('comp.cost_x','cost_x',mode='both',combos='mirror_costs',active=False)
    sig_y = Signal.define('comp.cost_y','cost_y',mode='both',combos='mirror_costs',active=False)
    sig_z = Signal.define('comp.cost_z','cost_z',mode='both',combos='mirror_costs',active=False)

    @system_property
    def total_budget(self)->float:
        return self.budget + self.comp.budget

    @system_property
    def total_length(self)->float:
        return self.max_length + self.comp.max_length

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