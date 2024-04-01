from engforge.system import System
from engforge.dynamics import GlobalDynamics,DynamicsMixin
from engforge.components import forge, Component
from engforge.attr_solver import Solver
from engforge.properties import system_property
from engforge.eng.costs import cost_property,CostModel

from engforge.eng.solid_materials import *

import numpy as np

theta = np.concatenate((np.linspace(0,2*np.pi,120),np.array([0])))

@forge(auto_attribs=True)
class SliderCrank(System,CostModel):

    success_thresh = 1000
    dynamic_state_vars:list = ['theta','omega']
    dynamic_input_vars:list = ['Tg']

    #outer gear
    Ro: float = 0.33
    gear_thickness: float = 0.0127 #1/2 in
    gear_material: SolidMaterial = ANSI_4130() #try aluminum and mild steel

    #motor gear
    Tg: float = 0
    Rg: float = 0.03
    Tg_max: float = 10 #Nm
    omg_g_max: float = 3000*(2*3.14159/60) #3000 rpm
    
    #crank 
    Rc: float = 0.25
    Lo: float = 0.3
    Lo_factor: float = 2.0001

    #rotation position
    theta: float = 0.
    omega: float = 0.
    b_rot: float = 0.0

    #pushrod & spring
    k_spring: float = 0.0
    bf_rod: float = 0.0
    y_offset: float = 0
    x_offset: float = 0
    X_spring_center: float = 0
    
    #signal targets or input
    F_external_x: float = 0
    F_external_y: float = 0
    dX_actuator: float = 1

    #Radius Design Variables
    rc_slv = Solver.declare_var('Rc',combos='design')
    rc_slv.add_var_constraint(lambda s,p: s.Ro,'max',combos='max_crank')
    rc_slv.add_var_constraint(0,'min')

    ro_slv = Solver.declare_var('Ro',combos='design')
    ro_slv.add_var_constraint(0,'min')

    rg_slv = Solver.declare_var('Rg',combos='design')
    rg_slv.add_var_constraint(0,'min')

    rg_slv = Solver.declare_var('Lo',combos='design')
    rg_slv.add_var_constraint(lambda s,p: s.Rc*s.Lo_factor,'min')    

    offy_slv = Solver.declare_var('y_offset',combos='design')
    offx_slv = Solver.declare_var('x_offset',combos='design')
    spring_slv = Solver.declare_var('X_spring_center',combos='spring_sym')

    #objectives
    goal_dx_range: float = 0.25
    main_gear_speed_min: float = 20*(2*3.14159/60)
    main_gear_speed_max: float = 120*(2*3.14159/60)

    cost_slv = Solver.objective('combine_cost',kind='min',combos='design,cost,goal')

    #constraints
    gear_speed_slv = Solver.eq_con('dx_goal',combos='design,gear')
    range_slv = Solver.eq_con('ds_goal',combos='design,goal')
    sym_slv = Solver.eq_con('end_force_diff',combos='spring_sym')
    

    gear_pos_slv = Solver.con_ineq('final_gear_ratio',combos='design,gear')
    crank_pos_slv = Solver.con_ineq('crank_gear_ratio',combos='design,gear')
    motor_pos_slv = Solver.con_ineq('motor_gear_ratio',combos='design,gear')



    #Dynamics
    nonlinear: bool = True #sinusoidal-isms
    #TODO: v_pos = Transient.time_derivative('x_pos)

    #Forces & Torques
    @system_property
    def input_torque(self)-> float:
        return self.Tg * self.final_gear_ratio
    
    @system_property
    def end_force_diff(self)-> float:
        x = self.motion_curve
        return ((x.max()-self.X_spring_center)-(x.min()-self.X_spring_center))
    
    @system_property
    def reaction_torque(self) -> float:
        return self.Rc_x * self.Freact_y - self.Rc_y * self.Fslide_tot

    @system_property
    def Freact_y(self)-> float:
        return np.sin(self.gamma)*np.cos(self.gamma)*self.Fslide_tot + self.F_external_y

    @system_property
    def Fslide_tot(self) -> float:
        return self.Fslide_fric + self.Fslide_spring + self.F_external_x
    
    @system_property
    def Fslide_fric(self) -> float:
        return -1 * self.bf_rod * self.v_pos
    
    @system_property
    def Fslide_spring(self) -> float:
        return -1 * self.k_spring * (self.x_pos - self.X_spring_center)

    #Positions & Angles
    @system_property
    def crank_angle(self) -> float:
        """angle between the crank and the vertical axis"""
        return np.rad2deg(self.theta)
    
    @system_property
    def alpha(self) -> float:
        '''angle between the crank and the horizontal axis'''
        return self.theta - np.pi/2

    @system_property
    def gamma(self) -> float:
        '''angle between the rod and the horizontal axis'''
        return np.arcsin(np.sin(self.alpha)*(self.Rc/self.Lo))

    @solver_cached
    def motion_curve(self) -> np.ndarray:
        x = self.Rc*np.cos(theta) + (self.Lo**2 - (self.Rc*np.sin(theta)-self.y_offset)**2)**0.5
        return x

    @system_property
    def max_dx_range(self) -> float:
        x = self.motion_curve
        return max(x.max() - x.min(),0)
    
    @system_property
    def max_x_theta(self) -> float:
        x = self.motion_curve
        return theta[np.argmax(x)]

    @system_property
    def min_x_theta(self) -> float:
        x = self.motion_curve
        return theta[np.argmin(x)]

    @system_property
    def dx_goal(self)->float:
        return (self.max_dx_range - self.goal_dx_range)
    
    @system_property
    def Rc_x(self)-> float:
        return self.Rc * np.cos(self.theta)
    
    @system_property
    def Rc_y(self)-> float:
        return self.Rc * np.sin(self.theta)    

    @system_property
    def x_pos(self) -> float:
        return self.Rc_x + self.x_offset
    
    @system_property
    def y_pos(self) -> float:
        return self.Rc_y - self.y_offset
    
    @system_property
    def v_pos(self) -> float:
        #TODO:replace with v_pos = Transient.time_derivative('x_pos)
        omega_rad = self.omega
        th = self.theta
        A = self.Rc*omega_rad
        
        s = (self.Rc*np.sin(th) - self.y_offset)
        N = s*np.cos(th)
        D = (self.Lo**2 - s**2)**0.5
        return A*((N/D) - np.sin(th))
    

    #Gear Properties
    @system_property
    def ds_goal(self)->float:
        dS = self.main_gear_speed_max - self.omg_g_max/self.motor_gear_ratio
        return dS

    @system_property
    def mass_main_gear(self) -> float:
        return  (np.pi * self.gear_material.density * self.Ro**2  * self.gear_thickness)
    
    @system_property
    def Imain_gear(self) -> float:
        return self.mass_main_gear * (self.Ro**2)

    @system_property
    def mass_pwr_gear(self) -> float:
        return  (np.pi*self.gear_material.density*self.Rg**2*self.gear_thickness)
    
    @system_property
    def Ipwr_gear(self) -> float:
        return self.mass_main_gear * (self.Rg**2)    

    @cost_property
    def main_gear_cost(self) -> float:
        return self.gear_material.cost_per_kg * self.mass_main_gear
    
    @cost_property
    def gear_cost(self) -> float:
        return self.gear_material.cost_per_kg * self.mass_pwr_gear
    
    @system_property
    def motor_gear_ratio(self) ->  float:
        return self.Ro/self.Rg
    
    @system_property
    def crank_gear_ratio(self) ->  float:
        return self.Rc/self.Ro
    
    @system_property
    def final_gear_ratio(self) ->  float:
        return self.motor_gear_ratio* self.crank_gear_ratio
    
    @system_property
    def crank_rod_ratio(self) ->  float:
        return self.Lo/self.Rc

    #Dynamics Matricies
    def create_state_matrix(self,*args,**kw):
        #TODO: add rotational-positional parasitics (if any)

        return np.array([[0,1],[0,-self.b_rot/self.Imain_gear]])  
    
    def create_state_input_matrix(self, **kwargs) -> np.ndarray:
        return np.array([[0],[1/self.Imain_gear]])
    
    def create_state_constants(self, **kwargs) -> np.ndarray:
        return np.array([0,0])
    
    def update_state(self,t, A, X):
        return np.array([[0,1],[0,-self.b_rot/self.Imain_gear]])    
    
    def update_state_constants(self, t, F, X) -> np.ndarray:
        #TODO: add torque & force interactions
        accl_input = self.input_torque / self.Imain_gear
        accl_react = self.reaction_torque / self.Imain_gear
        return np.array([0,accl_input+accl_react]) 