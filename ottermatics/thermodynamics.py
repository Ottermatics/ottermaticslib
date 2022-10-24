import attr
#from ottermatics.configuration import Configuration, otterize
from ottermatics.components import Component, table_property, otterize

from ottermatics.fluid_material import Water,Air,Steam
from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP

from numpy import vectorize

STD_TEMP = 273 + 20
STD_PRESSURE = 1.01325E5

#Thermodynamics
@vectorize
def heat_of_vaporization(P,material='Water',measure='Hmolar'):
    '''Returns heat of vaporization for material
    :param P: pressure in P
    '''
    assert measure.startswith('H') #Must be an enthalpy
    if P > 2.2063e+07: #Critical Pressure
        return 0.0
    H_L = PropsSI(measure,'P',P,'Q',0,material)
    H_V = PropsSI(measure,'P',P,'Q',1,material)
    return H_V - H_L

@vectorize
def boiling_point(P,material='Water'):
    '''Returns the boiling point in `K` for the material'''
    return PropsSI('T','P',P,'Q',1,material)

#Heat Exchanger
def dp_he_entrance(sigma,G,rho):
    '''Heat Exchanger Entrance Pressure Loss
    :param sigma: contraction-ratio - ratio of minimum flow area to frontal area
    :param G: mass flux of fluid
    :param rho: density of fluid
    '''
    Kc = 0.42 * (1.0 - sigma**2.0)**2.0
    return (1.0 - sigma**2.0 + Kc) * (G**2.0 / rho) / 2.0

def dp_he_exit(sigma,G,rho):
    '''Heat Exchanger Exit Pressure Loss
    :param sigma: contraction-ratio - ratio of minimum flow area to frontal area
    :param G: mass flux of fluid
    :param rho: density of fluid
    '''
    Ke = (1.0 - sigma)**2.0
    return (1.0 - sigma**2.0 + Ke) * (G**2.0 / rho) / 2.0    

def dp_he_core(G,f,L,rho,Dh):
    '''Losses due to friction
    :param f: fanning friction factor
    :param G: mass flux (massflow / Area)
    :param L: length of heat exchanger
    :param rho: intermediate density
    :param Dh: diameter of heat exchanger
    '''
    top = 4 * f * L * G**2.0
    btm = Dh * 2 * rho
    dp_friciton = top / btm
    return dp_friciton
    
def dp_he_gas_losses(G,rhoe,rhoi):
    '''Measures the pressure loss or gain due to density changes in the HE
    :param G: mass flux
    :param rhoe: exit density
    :param rhoi: entrance density
    '''
    dp_pressure = G**2.0 * ((1.0 / rhoe ) - (1.0/rhoi))
    return dp_pressure

def fanning_friction_factor(Re,method='turbulent'):
    if method == 'turbulent':
        if Re <= 5E4:
            return 0.0791 / (Re**0.25)
        return 0.0014 + 0.125 / (Re**0.32)
    elif method == 'laminar':
        return 16.0 / Re
    else: #Default to turbulent
        return fanning_friction_factor(Re,method='turbulent')

#Simple Elements
@otterize
class SimpleHeatExchanger(Component):

    Thi = attr.ib()
    mdot_h = attr.ib()
    Cp_h = attr.ib()

    Tci = attr.ib()
    mdot_c = attr.ib()
    Cp_c = attr.ib()

    efficiency = attr.ib(default=0.8)
    name = attr.ib(default='HeatExchanger')

    @table_property
    def CmatH(self):
        return self.Cp_h * self.mdot_h

    @table_property
    def CmatC(self):
        return self.Cp_c * self.mdot_c       

    @table_property
    def Tout_ideal(self):
        numerator = self.Thi * self.CmatH + self.Tci * self.CmatC
        denominator = self.CmatC + self.CmatH
        return numerator / denominator

    @table_property
    def Qdot_ideal(self):
        '''Use Tout ideal to determine the heat flow should be the same for both'''
        v1 = self.CmatH * (self.Thi - self.Tout_ideal)
        v2 = self.CmatC * (self.Tout_ideal - self.Tci)
        if abs( (v2 - v1) / float(v1) ) >= 0.1:
            self.warning('Qdot_ideal not matching')
        return (v1+v2)/2.0

    @table_property
    def Qdot(self):
        return self.Qdot_ideal * self.efficiency
    
    @table_property
    def Th_out(self):
        return self.Thi - self.Qdot / self.CmatH

    @table_property
    def Tc_out(self):
        return self.Tci + self.Qdot / self.CmatC  


#Compression
@otterize
class SimpleCompressor(Component):   

    pressure_ratio = attr.ib()

    Tin = attr.ib()
    mdot = attr.ib()

    Cp = attr.ib()
    gamma = attr.ib(default=1.4)

    efficiency = attr.ib(default=0.75)
    name = attr.ib(default='Compressor')

    @table_property
    def temperature_ratio(self):
        return (self.pressure_ratio**((self.gamma-1.0)/self.gamma)  - 1.0)/ self.efficiency

    @table_property
    def Tout(self):
        return self.temperature_ratio * self.Tin

    @table_property
    def power_input(self):
        return self.Cp * self.mdot * (self.Tout - self.Tin)

    def pressure_out(self,pressure_in):
        return self.pressure_ratio * pressure_in

@otterize
class SimpleTurbine(Component):   

    Pout = attr.ib()
    
    Pin = attr.ib()
    Tin = attr.ib()

    mdot = attr.ib()

    Cp = attr.ib()
    gamma = attr.ib(default=1.4)

    efficiency = attr.ib(default=0.80)
    name = attr.ib(default='Turbine')

    @table_property
    def pressure_ratio(self):
        return self.Pin / self.Pout

    @table_property
    def Tout(self):
        return self.Tin * ( 1 - self.efficiency * (1.0 - (1/self.pressure_ratio)**((self.gamma-1.0)/self.gamma))  )
        #return self.Tin * self.pressure_ratio**((self.gamma-1.0)/self.gamma)/ self.efficiency 

    @table_property
    def power_output(self):
        '''calculates power output base on temp diff (where eff applied)'''
        return  self.Cp * self.mdot * ( self.Tin - self.Tout ) 


#Compression
@otterize
class SimplePump(Component):   

    MFin = attr.field() #kg/s
    pressure_ratio = attr.field() #nd
    Tin = attr.field(default=STD_TEMP) #k
    Pin = attr.field(default=STD_PRESSURE) #c

    efficiency = attr.field(default=0.75)
    fluid = attr.field(factory=Water)
    name = attr.field(default='pump')

    def __on_init__(self):
        self.evaluate()

    def evaluate(self):
        self.fluid.T = self.Tin
        self.fluid.P = self.Pin

        Tsat = self.fluid.Tsat
        if self.Tin / Tsat >= 1:
            self.warning('infeasible: pumping vapor!')

        elif (self.Tin / self.fluid.Tsat) > 0.8:
            self.warning('pump near saturated temp')

    @table_property
    def volumetric_flow(self):
        return self.MFin / self.fluid.density

    @table_property
    def temperature_delta(self):
        rho = self.fluid.density
        v = self.MFin / rho
        p = self.Pin
        w = v * p / self.efficiency #work done
        cp = self.fluid.specific_heat
        dt = w * (1-self.efficiency) / (rho*cp*v)
        return dt

    @table_property
    def Pout(self):
        return self.pressure_ratio * self.Pin

    @table_property
    def Tout(self):
        return self.temperature_delta + self.Tin

    @table_property
    def power_input(self):
        return self.volumetric_flow * self.pressure_ratio / self.efficiency
