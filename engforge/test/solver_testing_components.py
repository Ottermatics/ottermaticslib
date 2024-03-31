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

import unittest
from engforge.configuration import forge
from engforge.eng.costs import CostModel, Economics, cost_property
from engforge.attr_slots import Slot
from engforge.system import System
from engforge.components import Component
from engforge.component_collections import ComponentIterator

from engforge.properties import system_property

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
    sym = Solver.constraint_equality("x",'y',combos='sym',active=False)

    costA = Solver.con_ineq('budget','cost',combos='ineq_cost',active=False)
    costF = Solver.con_ineq(wrap_f(Fun_cm),combos='ineq_cost',active=False)
    costP = Solver.con_ineq('cost_margin',combos='ineq_cost',active=False)

    #Constraints by length
    lenA = Solver.con_ineq('max_length','combine_length',combos='ineq_length',active=False)
    lenF = Solver.con_ineq(wrap_f(Fun_em),combos='ineq_length',active=False)
    lenP = Solver.con_ineq('edge_margin',combos='ineq_length',active=False)

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
class CubeComp(Component,SpaceMixin):
    # this should be energy neutral 🤞
    
    dynamic_state_vars: list = ["x", "y", "z"]

    base_cost: float = attrs.field(default=10)
    cost_x: float = attrs.field(default=10)
    cost_y: float = attrs.field(default=20)
    cost_z: float = attrs.field(default=5)

    budget: float = attrs.field(default=300)
    max_length: float = attrs.field(default=100)
    volume_goal: float = attrs.field(default=100)





    

@forge(auto_attribs=True)
class CubeSystem(System,SpaceMixin):
    
    dynamic_state_vars: list = ["x", "y", "z"]

    comp = Slot.define(CubeComp)

    x: float = attrs.field(default=0.5)
    y: float = attrs.field(default=0.5)
    z: float = attrs.field(default=0.5)

    cost_x: float = attrs.field(default=5)
    cost_y: float = attrs.field(default=15)
    cost_z: float = attrs.field(default=10)

    goal_vol_frac: float = 0.5

    sys_budget = Solver.con_ineq('total_budget','system_cost',combos='total_budget')

    sys_length = Solver.con_ineq('total_length','system_length',combos='total_length')

    volfrac = Solver.eq_con('goal_vol_frac','vol_frac',combos='vol_frac_eq',active=False)

    obj = Solver.objective('total_volume',combos='volume',kind='max')
    hght = Solver.objective('total_height',combos='height',kind='max')

    x_lim = Solver.con_ineq('x','comp.x',combos='tierd_top',active=False)
    y_lim = Solver.con_ineq('y','comp.y',combos='tierd_top',active=False)

    sig_x_cst = Signal.define('comp.cost_x','cost_x',mode='both',combos='mirror_costs',active=False)
    sig_y_cst = Signal.define('comp.cost_y','cost_y',mode='both',combos='mirror_costs',active=False)
    sig_z_cst = Signal.define('comp.cost_z','cost_z',mode='both',combos='mirror_costs',active=False)

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
    

@forge(auto_attribs=True)
class DynamicComponent(Component):
    dynamic_state_vars: list = ["x", "v"]
    x: float = 1
    v: float = 0

    b: float = 0.1
    K: float = 10
    M: float = 100

    x0: float = 0.5

    nonlinear: bool = False
    Fext: float = 0

    acc =Solver.declare_var("x0")
    no_load = Solver.constraint_equality("a")

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
class DynamicSystem(System):
    dynamic_state_vars: list = ["x", "v"]

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
    #slv = Solver.declare_var("delta_a")

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

    def update_state(self, *args, **kwargs) -> np.ndarray:
        """creates the state matrix for the system"""
        return np.array(
            [[0, 1.0], [-self.K / self.Mass, -1 * self.Damp / self.Mass]]
        )

    def update_state_constants(self, *args, **kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.array([0, self.Force / self.Mass])    
    

@forge
class SpringMass(System):
    
    k: float = attrs.field(default=50)
    m: float = attrs.field(default=1)
    g: float = attrs.field(default=9.81)
    u: float = attrs.field(default=0.3)

    a: float = attrs.field(default=0)
    x: float = attrs.field(default=0.0)
    v: float = attrs.field(default=0.0)

    wo_f: float = attrs.field(default=1.0)
    Fa: float = attrs.field(default=10.0)

    x_neutral: float = attrs.field(default=0.5)

    res =Solver.constraint_equality("sumF")
    var_a = Solver.declare_var("a",combos='a',active=False)
    
    var_b = Solver.declare_var("u",combos='u',active=False)
    var_b.add_var_constraint(0.0,kind="min")
    var_b.add_var_constraint(1.0,kind="max")

    vtx = Time.integrate("v", "accl")
    xtx = Time.integrate("x", "v")
    xtx.add_var_constraint(0,kind="min")
    #xtx.add_var_constraint(lambda s,p:0,kind="min")

    #FIXME: implement trace testing
    #pos = Trace.define(y="x", y2=["v", "a"])

    @system_property
    def dx(self) -> float:
        return self.x_neutral - self.x

    @system_property
    def Fspring(self) -> float:
        return self.k * self.dx

    @system_property
    def Fgrav(self) -> float:
        return self.g * self.m

    @system_property
    def Faccel(self) -> float:
        return self.a * self.m

    @system_property
    def Ffric(self) -> float:
        return self.u * self.v

    @system_property
    def sumF(self) -> float:
        return self.Fspring - self.Fgrav - self.Faccel - self.Ffric + self.Fext
    
    @system_property
    def Fext(self) -> float:
        return self.Fa * np.cos( self.time * self.wo_f )

    @system_property
    def accl(self) -> float:
        return self.sumF / self.m
    

#COSTS Testing
    
@forge
class Norm(Component,CostModel):
    pass

@forge
class Comp1(Component,CostModel):
    norm = Slot.define(Norm,none_ok=True)
    not_cost = Slot.define(Component)

@forge
class Comp2(Norm,CostModel):
    comp1 = Slot.define(Comp1,none_ok=True,default_ok=False)

quarterly = lambda inst,term: True if (term+1)%3==0 else False
@forge
class TermCosts(Comp1,CostModel):

    @cost_property(category='capex')
    def cost_init(self):
        return 100
    
    @cost_property(mode='maintenance',category='opex')
    def cost_maintenance(self):
        return 10

    @cost_property(mode='always',category='tax')
    def cost_tax(self):
        return 1
    
    @cost_property(mode=quarterly,category='opex,tax',label='quarterly wage tax')
    def cost_wage_tax(self):
        return 5*3
    

@forge
class EconDefault(System,CostModel):
    econ = Slot.define(Economics)
    comp = Slot.define(Component,none_ok=True)
    comp1 = Slot.define(Comp1,none_ok=True)

@forge
class EconRecursive(System,CostModel):
    econ = Slot.define(Economics)
    comp1 = Slot.define(Comp1,none_ok=True)
    comp2 = Slot.define(Comp2,none_ok=True)


#FAN System test
@forge
class Fan(Component,CostModel):
    """a fan component"""
    blade_cost_com: float = attrs.field(default=100.0)
    area:float = attrs.field(default=10.0)
    V:float = attrs.field(default=5.0)

    @system_property
    def volumetric_flow(self) -> float:
        return self.V * self.area
    
    @cost_property(category='capex,mfg,material',mode='initial')
    def blade_cost(self):
        return self.area * self.blade_cost_com
    
    @cost_property(category='labor,opex',mode='maintenance')
    def repair_cost(self):
        return self.volumetric_flow * 0.1     
    
@forge
class Motor(Component,CostModel):
    """a fan component"""
    spc_motor_cost: float = attrs.field(default=100.0)

    @system_property
    def power(self) -> float:
        return self.parent.fan.volumetric_flow * self.parent.fan.V
    
    @cost_property(category='capex,mfg,electrical',mode='initial')
    def motor_cost(self):
        return self.power * self.spc_motor_cost  
    
    @cost_property(category='labor,opex',mode='maintenance')
    def repair_cost(self):
        return self.power * 0.1    

@forge
class MetalBase(Component,CostModel):

    cost_per_item = 1000
        
@forge
class SysEcon(Economics):

    terms_per_year = 12

    def calculate_production(self,parent,term):
        return self.parent.fan.volumetric_flow

@forge
class FanSystem(System,CostModel):

    base = Slot.define(Component)
    fan = Slot.define(Fan)
    motor = Slot.define(Motor)

    econ = Slot.define(SysEcon)

FanSystem.default_cost('base',100)