
from engforge import *
import attrs
import numpy

@forge(auto_attribs=True)
class FourBar(System):

    #r1
    r1: float = 0.02
    gamma_zero: float = 10*3.1415/180 #radians
    gamma: float = 0
    r1_slv = Solver.declare_var('r1',combos='indep,r1')
    r1_slv.add_var_constraint(0.0,kind='min',combos='lim')
    r1_slv.add_var_constraint(0.05,kind='max',combos='lim')

    gz_slv = Solver.declare_var('gamma_zero',combos='indep,zero')
    gz_slv.add_var_constraint(0.0,kind='min',combos='lim')
    gz_slv.add_var_constraint(15*3.1415/180,kind='max',combos='lim')

    gam_slv = Solver.declare_var('gamma',combos='indep,gamma')
    gam_slv.add_var_constraint(0.0,kind='min',combos='lim') 
    gam_sig = Signal.define('gamma','gamma_zero',combos='gam_sig')
    
    #arm
    theta: float = 0 #radians
    ra: float = 0.03
    theta_slv = Solver.declare_var('theta',combos='indep,theta')
    theta_slv.add_var_constraint(0.0,kind='min',combos='lim')
    theta_slv.add_var_constraint(3.14159,kind='max',combos='lim')    

    #r2
    x2: float = 0.05
    h2: float = 0.
    x2_slv = Solver.declare_var('x2',combos='indep,x2')
    x2_slv.add_var_constraint(0.025,kind='min',combos='lim')
    x2_slv.add_var_constraint(0.05,kind='max',combos='lim')
    
    h2_slv = Solver.declare_var('h2',combos='indep,h2')
    h2_slv.add_var_constraint(-0.01,kind='min',combos='lim')
    h2_slv.add_var_constraint(0.01,kind='max',combos='lim')

    len_slv = Solver.con_eq('l3_zero','l3_gamma',combos='goal,leneq')
    goal_slv = Solver.objective('gamma_goal',kind='max',combos='goal,gamma')
    l3_max = Solver.objective('l3_zero',kind='max',combos='goal,l3max')

    @system_prop
    def gamma_goal(self)-> float:
        return self.gamma
    
    @system_prop
    def r3_x_zero(self)-> float:
        '''length defined by closed condition'''
        return self.x2 + self.ra - self.r1 * numpy.cos(self.gamma_zero)
    
    @system_prop
    def r3_x(self)-> float:
        '''length defined by gamma/theta'''
        return self.x2 + self.ra*numpy.cos(self.theta) - self.r1 * numpy.cos(self.gamma)
    
    @system_prop
    def y_zero(self)-> float:
        '''length defined by closed condition'''
        return self.h2 - self.r1 * numpy.sin(self.gamma_zero)
    
    @system_prop
    def y(self)-> float:
        '''length defined by gamma/theta'''
        return self.x2 + self.ra* numpy.sin(self.theta) - self.r1 * numpy.sin(self.gamma) 
    
    @system_prop
    def l3_zero(self)-> float:
        return (self.r3_x_zero**2 + self.y_zero**2)
    
    @system_prop
    def l3_gamma(self)-> float:
        return (self.r3_x**2 + self.y**2)
    
#%run -i ~/engforge/engforge/test/test_four_bar.py
import numpy as np

fb = FourBar()
fb.run(combos='*',revert_last=False,revert_every=False)

df = fb.last_context.dataframe

fb.run(combos='goal,gamma,theta,lim',theta=np.linspace(0,3.14159),revert_last=False,revert_every=False)