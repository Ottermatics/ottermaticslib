from engforge.system import System
from engforge.dynamics import GlobalDynamics,DynamicsMixin
from engforge.components import forge, Component
from engforge.attr_solver import Solver
from engforge.properties import system_property

import numpy as np

@forge(auto_attibutes=True)
class SliderCrank(System,GlobalDynamics):

    dynamic_state_vars = ['theta','omega','x']

    dynamic_input_vars = ['Tg']

    #outer gear
    Ro: float = 0.33
    #motor gear
    Tg: float = 0
    Rg: float = 0.03
    Tg_max: float = 100
    omg_g_max: float = 1000
    
    #crank 
    Rc: float = 0.25
    Lo: float = 0.3

    #rotation position
    theta: float = 0.
    omega: float = 0.
    b_rot: float = 0.

    #pushrod
    x: float = 1
    x_offset: float = 0
    y_offset: float = 0

    #Radius Design Variables
    rc_slv = Solver.declare_var('Rc',combos='design')
    rc_slv.add_var_constraint(lambda s,p: s.Ro,'max')
    rc_slv.add_var_constraint(0,'min')

    ro_slv = Solver.declare_var('Ro',combos='design')
    ro_slv.add_var_constraint(0,'min')

    rg_slv = Solver.declare_var('Rg',combos='design')
    rg_slv.add_var_constraint(0,'min')

    rg_slv = Solver.declare_var('Lo',combos='design')
    rg_slv.add_var_constraint(lambda s,p: s.Rc,'min')    

    off_slv = Solver.declare_var('y_offset',combos='design')
    off_slv = Solver.declare_var('x_offset',combos='design')

    l_eq_slv = Solver.equality_constraint('Lo','Lcalc',combos='design')

    @system_property
    def gear_ratio(self) ->  float:
        return self.Ro/self.Rg
    
    @system_property
    def rotational_drag(self) -> float:
        return self.omega * self.b_rot
    
    @system_property
    def Rc_x(self)-> float:
        return self.Rc * np.sin(self.theta)
    
    @system_property
    def Rc_y(self)-> float:
        return self.Rc * np.cos(self.theta)    

    @system_property
    def dXr(self) -> float:
        return self.x + self.x_offset - self.Rc_x
    
    @system_property
    def dYr(self) -> float:
        return self.y_offset - self.Rc_y
    
    @system_property
    def Lcalc(self):
        return (self.dYr**2 + self.dXr**2)**0.5
    
    
