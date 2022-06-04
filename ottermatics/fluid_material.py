from ottermatics.configuration import Configuration, otterize
from ottermatics.patterns import inst_vectorize

import matplotlib
import random
import attr
import numpy
import inspect
import sys

import CoolProp
from CoolProp.CoolProp import PropsSI
import fluids
import abc


STD_PRESSURE = 1E5 #pa
STD_TEMP = 273+15

@otterize
class FluidMaterial(Configuration):
    '''Placeholder for pressure dependent material'''
    P = attr.ib(default=STD_PRESSURE, type=float)
    T = attr.ib(default=STD_TEMP, type=float)
    

    @abc.abstractproperty
    def density(self,temp:float):
        '''default functionality, assumed gas with eq-state= gas constant'''
        raise NotImplemented()

    @abc.abstractproperty
    def viscosity(self,temp:float):
        '''ideal fluid has no viscosity'''
        raise NotImplemented()


@otterize
class IdealGas(FluidMaterial):
    '''Material Defaults To Gas Properties, so eq_of_state is just Rgas, no viscosity, defaults to air'''
    gas_constant = attr.ib(default=287.,type=float)

    @property
    def density(self):
        '''default functionality, assumed gas with eq-state= gas constant'''
        return self.P / (self.gas_constant * self.T)

    @property
    def viscosity(self):
        '''ideal fluid has no viscosity'''
        return 0.0



IdealAir = type('IdealAir',(IdealGas,),{'gas_constant': 287.})
IdealH2 = type('IdealH2',(IdealGas,),{'gas_constant': 4124.2})
IdealOxygen = type('IdealOxygen',(IdealGas,),{'gas_constant': 259.8})
IdealSteam = type('IdealSteam',(IdealGas,),{'gas_constant': 461.5})

# @otterize
# class PerfectGas(FluidMaterial):
#     '''A Calorically Perfect gas with viscosity'''
#     eq_of_state = attr.ib()
#     P = attr.ib(default=STD_PRESSURE, type=float)
    
#     @property
#     def density(self):
#         '''default functionality, assumed gas with eq-state= gas constant'''
#         return self.eq_of_state.density(T=self.T,P=self.P)

#     @property
#     def viscosity(self):
#         '''ideal fluid has no viscosity'''
#         return self.eq_of_state.viscosity(T=self.T,P=self.P)


@otterize
class CoolPropMaterial(FluidMaterial):
    '''Uses coolprop equation of state'''
    material: str

    @property
    def density(self):
        '''default functionality, assumed gas with eq-state= gas constant'''
        return PropsSI('D','T',self.T,'P',self.P,self.material)

    @property
    def enthalpy(self):
        return PropsSI('H','T',self.T,'P',self.P,self.material)      

    @property
    def viscosity(self):
        '''ideal fluid has no viscosity'''
        return PropsSI('V','T',self.T,'P',self.P,self.material)

Water = type('Water',(CoolPropMaterial,),{'material':'Water'})
Air =  type('Air',(CoolPropMaterial,),{'material':'Air'})
Oxygen =  type('Oxygen',(CoolPropMaterial,),{'material':'Oxygen'})
Hydrogen =  type('Hydrogen',(CoolPropMaterial,),{'material':'Hydrogen'})