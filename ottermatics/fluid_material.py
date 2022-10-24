from ottermatics.configuration import Configuration, otterize
from ottermatics.components import Component,table_property, otterize

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


#TODO: add a exact fluid state (T,P) / (Q,P) in the concept of processes for each thermodynamic operation (isothermal,isobaric,heating...ect)

STD_PRESSURE = 1E5 #pa
STD_TEMP = 273+15

@otterize
class FluidMaterial(Component):
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

    @abc.abstractproperty
    def surface_tension(self):
        return 0.0

    #TODO: enthalpy

@otterize
class IdealGas(FluidMaterial):
    '''Material Defaults To Gas Properties, so eq_of_state is just Rgas, no viscosity, defaults to air'''
    gas_constant = attr.ib(default=287.,type=float)

    @table_property
    def density(self):
        '''default functionality, assumed gas with eq-state= gas constant'''
        return self.P / (self.gas_constant * self.T)

    @table_property
    def viscosity(self):
        '''ideal fluid has no viscosity'''
        return 0.0

    # @table_property
    # def surface_tension(self):
    #     return 0.0



IdealAir = type('IdealAir',(IdealGas,),{'gas_constant': 287.})
IdealH2 = type('IdealH2',(IdealGas,),{'gas_constant': 4124.2})
IdealOxygen = type('IdealOxygen',(IdealGas,),{'gas_constant': 259.8})
IdealSteam = type('IdealSteam',(IdealGas,),{'gas_constant': 461.5})

# @otterize
# class PerfectGas(FluidMaterial):
#     '''A Calorically Perfect gas with viscosity'''
#     eq_of_state = attr.ib()
#     P = attr.ib(default=STD_PRESSURE, type=float)
    
#     @table_property
#     def density(self):
#         '''default functionality, assumed gas with eq-state= gas constant'''
#         return self.eq_of_state.density(T=self.T,P=self.P)

#     @table_property
#     def viscosity(self):
#         '''ideal fluid has no viscosity'''
#         return self.eq_of_state.viscosity(T=self.T,P=self.P)


@otterize
class CoolPropMaterial(FluidMaterial):
    '''Uses coolprop equation of state'''
    material: str

    #TODO: handle phase changes with internal _quality that you can add heat to
    _surf_tension_K = None
    _surf_tension_Nm = None

    @table_property
    def density(self):
        '''default functionality, assumed gas with eq-state= gas constant'''
        return PropsSI('D','T',self.T,'P',self.P,self.material)

    @table_property
    def enthalpy(self):
        return PropsSI('H','T',self.T,'P',self.P,self.material)      

    @table_property
    def viscosity(self):
        return PropsSI('V','T',self.T,'P',self.P,self.material)

    @table_property
    def surface_tension(self):
        """returns liquid surface tension"""
        if self._surf_tension_K and self._surf_tension_Nm:
        
            X = self._surf_tension_K
            Y = self._surf_tension_Nm
            l = Y[0]
            r = Y[-1]
            return numpy.interp(self.T,xp=X,fp=Y,left=l,right=r)
        
        self.warning('no surface tension model! returning 0')
        return 0.0

    @table_property
    def thermal_conductivity(self):
        """returns liquid thermal conductivity"""
        return PropsSI('CONDUCTIVITY','T',self.T,'P',self.P,self.material)

    @table_property
    def specific_heat(self):
        """returns liquid thermal conductivity"""
        return PropsSI('C','P',self.P,'T',self.T,self.material)

    @table_property
    def Tsat(self):
        return PropsSI('T','Q',0,'P',self.P,self.material)

    @table_property
    def Psat(self):
        return PropsSI('P','Q',0,'T',self.T,self.material)  

    def __call__(self,*args,**kwargs):
        """calls coolprop module with args adding the material"""
        args = (*args,self.material)
        return PropsSI(*args)

#TODO: add water suface tenstion
T_K = [273.15, 278.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15,
       343.15, 353.15, 363.15, 373.15, 423.15, 473.15, 523.15, 573.15,
       623.15, 647.25]
ST_NM = [0.0756, 0.0749, 0.0742, 0.0728, 0.0712, 0.0696, 0.0679, 0.0662,
       0.0644, 0.0626, 0.0608, 0.0589, 0.0482, 0.0376, 0.0264, 0.0147,
       0.0037, 0.    ]


Water = type('Water',(CoolPropMaterial,),{'material':'Water','_surf_tension_K':T_K,'_surf_tension_Nm':ST_NM})
Air =  type('Air',(CoolPropMaterial,),{'material':'Air'})
Oxygen =  type('Oxygen',(CoolPropMaterial,),{'material':'Oxygen'})
Hydrogen =  type('Hydrogen',(CoolPropMaterial,),{'material':'Hydrogen'})
Steam = type('Steam',(CoolPropMaterial,),{'material':'IF97:Water','_surf_tension_K':T_K,'_surf_tension_Nm':ST_NM})
SeaWater = type('SeaWater',(CoolPropMaterial,),{'material':'MITSW','_surf_tension_K':T_K,'_surf_tension_Nm':ST_NM})

