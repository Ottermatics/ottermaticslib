

'''We'll use the QP formulation to develop a fluid analysis system for fluids
start with single phase and move to others


1) Pumps Power = C x Q x P 
2) Compressor = C x (QP) x (PR^C2 - 1)
3) Pipes = dP = C x fXL/D x V^2 / 2g
4) Pipe Fittings / Valves = dP = C x V^2 (fXL/D +K)   |   K is a constant, for valves it can be interpolated between closed and opened
5) Splitters & Joins: Handle fluid mixing
6) Phase Separation (This one is gonna be hard)
7) Heat Exchanger dP = f(V,T) - phase change issues
8) Filtration dP = k x Vxc x (Afrontal/Asurface) "linear"
''' 

from ottermatics.components import Component
from ottermatics.configuration import otterize
from ottermatics.tabulation import table_property, NUMERIC_VALIDATOR, STR_VALIDATOR
from ottermatics.fluid_material import FluidMaterial
from ottermatics.common import G_grav_constant
import attr, attrs

import numpy
import fluids

@otterize(auto_attribs=True)
class Node(Component):
    x: float 
    y: float
    z: float


@otterize(auto_attribs=True)
class PipeFlow(Component):
    D: float
    v: float
    material: FluidMaterial

    def set_flow(self,flow):
        v = flow / self.A
        self.v = v

    #Geometric Propreties
    @table_property
    def A(self):
        ''':returns: the cross sectional area of pipe in [m2]'''
        return 3.1415* (self.D/2.)**2.0

    @table_property
    def C(self):
        ''':returns: the sectional circmerence of pipe'''
        return 3.1415*self.D

    #Fluid Propreties
    @table_property
    def Q(self):
        ''':returns: the volumetric flow through this pipe in [m3/s]'''
        return self.A * self.v

    @table_property
    def Mf(self):
        ''':returns: the massflow through this pipe in [kg/s]'''
        return self.density * self.Q

    @table_property
    def reynoldsNumber(self):
        ''':returns: the flow reynolds number'''
        return self.density * self.v * self.D / self.viscosity

    #Material Properties Exposure
    @table_property
    def density(self):
        return self.material.density

    @table_property
    def viscosity(self):
        return self.material.viscosity

    @table_property
    def enthalpy(self):
        return self.material.enthalpy

    @table_property
    def T(self):
        return self.material.T

    @T.setter
    def T(self,new_T):
        self.material.T = new_T

    @table_property
    def P(self):
        return self.material.P

    @P.setter
    def P(self,new_P):
        self.material.P = new_P

    @property
    def dP_f(self):
        '''The loss of pressure in the pipe due to pressure'''
        raise NotImplemented()

    @property
    def dP_p(self):
        '''The loss of pressure in the pipe due to potential'''
        raise NotImplemented()

    @property
    def dP_tot(self):
        raise NotImplemented()

    @property
    def Fvec(self):
        '''returns the fluidized vector for 1D CFD for continuity:0, and momentum:1'''
        #TODO: add energy eq
        #TODO: 1D CFD Pipe Network Solver :)
        return [ self.density * self.v,
                 self.density * self.v**2.0 + self.P ] #TODO: include Txx




@otterize
class Pipe(PipeFlow,Component):

    node_s = attr.ib( type=Node )
    node_e = attr.ib( type=Node )
    roughness = attr.ib( default = 0.0, type=float)
    bend_radius = attr.ib( default = None , type = float)

    _skip_attr = ['node_s','node_e']

    straight_method = 'Clamond'
    laminar_method='Schmidt laminar'
    turbulent_method = 'Schmidt turbulent'

    @table_property
    def Lx(self):
        return self.node_e.x - self.node_s.x

    @table_property
    def Ly(self):
        return self.node_e.y - self.node_s.y

    @table_property
    def Lz(self):
        return self.node_e.z - self.node_s.z

    @table_property
    def Lhz(self):
        ''':returns: The length of pipe element in the XY plane'''
        return numpy.sqrt( self.Lx**2.0 + self.Ly**2.0 )

    @table_property
    def L(self):
        ''':returns: The absolute length of pipe element'''
        return numpy.sqrt( self.Lx**2.0 + self.Ly**2.0 + self.Lz**2.0 )

    @table_property
    def inclination(self):
        ''':returns: the inclination angle in degrees'''
        return numpy.rad2deg( numpy.arctan2(self.Lz,self.Lhz) )
        
    @table_property
    def friction_factor(self):
        '''The friction factor considering bend radius'''
        if self.bend_radius is None:
            re =  self.reynoldsNumber
            return fluids.friction.friction_factor( re, self.roughness , Method=self.straight_method)
        else:
            re =  self.reynoldsNumber
            dc = self.D * self.bend_radius
            return fluids.friction.friction_factor_curved( re, self.D, dc, self.roughness, laminar_method=self.laminar_method, turbulent_method=self.turbulent_method)

    @table_property
    def Kpipe(self):
        '''The loss coeffient of this pipe section'''
        return self.friction_factor * self.L / self.D

    @table_property
    def dP_f(self):
        '''The loss of pressure in the pipe due to pressure'''
        return self.density * self.v**2.0 * self.Kpipe / 2.0

    @table_property
    def dP_p(self):
        '''The loss of pressure in the pipe due to potential'''
        return self.density * self.Lz * 9.81

    @table_property
    def dP_tot(self):
        return self.dP_f + self.dP_p



@otterize
class PipeFitting(Node,PipeFlow):
    
    Kfitting = attr.ib( default = 0.1, type=float)

    @table_property
    def dP_f(self):
        '''The loss of pressure in the pipe due to pressure'''
        return self.density * self.v**2.0 * self.Kfitting / 2.0  

    @table_property
    def dP_p(self):
        '''The loss of pressure in the pipe due to potential'''
        return 0.0       

    @table_property
    def dP_tot(self):
        return self.dP_f + self.dP_p
         

#TODO: Add in fitting numbers:
'''                          Fitting                      Types       K
0                       45° Elbow         Standard (R/D = 1)    0.35
1                       45° Elbow    Long Radius (R/D = 1.5)    0.20
2                90° Elbow Curved         Standard (R/D = 1)    0.75
3                90° Elbow Curved    Long Radius (R/D = 1.5)    0.45
4      90° Elbow Square or Mitred                        NaN    1.30
5                       180° Bend               Close Return    1.50
6                Tee, Run Through             Branch Blanked    0.40
7                   Tee, as Elbow            Entering in run    1.00
8                   Tee, as Elbow         Entering in branch    1.00
9             Tee, Branching Flow                        NaN    1.00
10                       Coupling                        NaN    0.04
11                          Union                        NaN    0.04
12                     Gate valve                 Fully Open    0.17
13                     Gate valve                   3/4 Open    0.90
14                     Gate valve                   1/2 Open    4.50
15                     Gate valve                   1/4 Open   24.00
16                Diaphragm valve                 Fully Open    2.30
17                Diaphragm valve                   3/4 Open    2.60
18                Diaphragm valve                   1/2 Open    4.30
19                Diaphragm valve                   1/4 Open   21.00
20        Globe valve, Bevel Seat                 Fully Open    6.00
21        Globe valve, Bevel Seat                   1/2 Open    9.50
22  Globe Valve, Composition seat                 Fully Open    6.00
23  Globe Valve, Composition seat                   1/2 Open    8.50
24                      Plug disk                 Fully Open    9.00
25                      Plug disk                   3/4 Open   13.00
26                      Plug disk                   1/2 Open   36.00
27                      Plug disk                   1/4 Open  112.00
28                    Angle valve                 Fully Open    2.00
29       Y valve or blowoff valve                 Fully Open    3.00
30                      Plug cock                \theta = 5°    0.05
31                      Plug cock               \theta = 10°    0.29
32                      Plug cock               \theta = 20°    1.56
33                      Plug cock               \theta = 40°   17.30
34                      Plug cock               \theta = 60°  206.00
35                Butterfly valve                \theta = 5°    0.24
36                Butterfly valve               \theta = 10°    0.52
37                Butterfly valve               \theta = 20°    1.54
38                Butterfly valve               \theta = 40°   10.80
39                Butterfly valve               \theta = 60°  118.00
40                    Check valve                      Swing    2.00
41                    Check valve                       Disk   10.00
42                    Check valve                       Ball   70.00
43                     Foot valve                        NaN   15.00
44                    Water meter                       Disk    7.00
45                    Water meter                     Piston   15.00
46                    Water meter  Rotary (star-shaped disk)   10.00
47                    Water meter              Turbine-wheel    6.00'''


@otterize(auto_attribs=True)
class Pump(Component):
    '''Simulates a pump with power input, max flow, and max pressure by assuming a flow characteristic'''
    max_flow: float #volumetric rate m3/s
    max_pressure: float #Pa
    #throttle: float

    @property
    def design_flow_curve(self):
        ''':returns: a tuple output of flow vector, and pressure vector'''
        flow = numpy.linspace(0,self.max_flow)
        return flow, self.max_pressure * (1 - (flow/self.max_flow)**2.0 )

    def dPressure(self,current_flow):
        '''The pressure the pump generates'''
        flow,dP = self.design_flow_curve
        assert current_flow >= 0, "Flow must be positive"
        assert current_flow <= self.max_flow, "Flow must be less than max flow"
        return numpy.interp(current_flow, flow, dP)

    def power(self,current_flow):
        '''The power used considering in watts'''
        return self.dPressure(current_flow) * current_flow
        

















if __name__ == '__main__':

    N = 10

    rho = 1000.0
    f = 0.015
    L = 1.0
    Po = 1E5


    A = 1.0 * ones(N)
    u = 0.0 * ones(N)
    p = Po * ones(N)

    Pin = 1.1E5
    Uin = lambda Ps: sqrt((Pin - Ps)/(rho*2.0))

    def F(i):
        ui = u[i]
        ps = p[i]    
        F1 = rho * ui
        F2 = F1 * ui + ps    
        return F1,F2
        
    def J(i):
        return 0.0, -rho * f 
        
    def decodeF(F1,F2):
        up = F1 / rho
        pp = F2 - rho * up**2.0
        return up,pp

    plast = p.copy()
    ulast = u.copy()
    it = 0
    started = False
    while not started or (plast - p).mean() > 1E-3 or (ulast - u).mean() > 1E-3:
        started = True

        plast = p.copy()
        ulast = u.copy()

        for i in range(N):
            if i == 0: #inlet
                ps = p[i]
                ui = Uin(ps)
                u[i] = ui
                
            
            #STATUS
            F1,F2 = F(i)
            J1,J2 = J(i)
            
            #PREDICT i+1
            dF1,dF2 = J1, J2
            Fp1,Fp2 = F1 + dF1*L, F2 + dF2*L
            
            up1,pp1 = decodeF(Fp1,Fp2)
            
            #CORRECTOR i+1
            Jp1,Jp2 = 0.0, -rho * f
            dpF1,dpF2 = Jp1, Jp2
            
            #avgGradients
            dF1n,dF2n = 0.5 * (dpF1 + dF1), 0.5 * (dpF2 + dF2)
        
            #Calculate i+1 actual
            Fn1,Fn2 = F1 + dF1n*L, F2 + dF2n*L
            
            un,pn = decodeF(Fn1,Fn2)
            if N-1 > i:
                u[i+1] = un
                p[i+1] = pn
        
        for i in reversed(range(N)):
            #if i == 0: #inlet
            #    ps = p[i]
            #    ui = Uin(ps)
            #    u[i] = ui
                
            if i == N-1:
                p[i] = Po
        
            #STATUS
            F1,F2 = F(i)
            J1,J2 = J(i)
        
            #PREDICT i+1
            dF1,dF2 = J1, J2
            Fp1,Fp2 = F1 + dF1*L, F2 + dF2*L
        
            up1,pp1 = decodeF(Fp1,Fp2)
        
            #CORRECTOR i+1
            Jp1,Jp2 = 0.0, -rho * f
            dpF1,dpF2 = Jp1, Jp2
        
            #avgGradients
            dF1n,dF2n = 0.5 * (dpF1 + dF1), 0.5 * (dpF2 + dF2)
        
            #Calculate i+1 actual
            Fn1,Fn2 = F1 + dF1n*L, F2 + dF2n*L
        
            un,pn = decodeF(Fn1,Fn2)
            if N-1 > i:
                u[i-1] = un
                p[i-1] = pn
        
        print(u)
        print(p)
        it += 1
        if it >= 2:
            break        