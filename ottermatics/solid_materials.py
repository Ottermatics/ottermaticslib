from ottermatics.configuration import Configuration, otterize
from ottermatics.patterns import inst_vectorize

import matplotlib
import random
import attr
import numpy
import inspect
import sys

from sectionproperties.pre.pre import Material

ALL_MATERIALS = []
METALS = []
PLASTICS = []
CERAMICS = []


CMAP = matplotlib.cm.get_cmap('viridis')
def random_color():
    return CMAP(random.randint(0,255))


@otterize
class SolidMaterial(Configuration,Material):
    '''A class to hold physical properties of solid structural materials'''
    name = attr.ib(default='solid material')
    color = attr.ib(factory=random_color)

    #Structural Properties
    density = attr.ib(default=1.0)
    elastic_modulus  = attr.ib(default=1E8) #Pa
    yield_strength  = attr.ib(default=1E6) #Pa
    tensile_strength_ultimate = attr.ib(default=2E6) #Pa
    hardness = attr.ib(default=10) #rockwell
    izod = attr.ib(default=100)
    poissons_ratio = attr.ib(default=0.30)

    #Thermal Properties
    melting_point = attr.ib(default=1000+273) #K
    maxium_service_temp = attr.ib(default=500+273) #K
    thermal_conductivity = attr.ib(default=10) #W/mK
    specific_heat = attr.ib(default=1000) #J/kgK
    thermal_expansion = attr.ib(default = 10E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=1E-8) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=1.0) #dollar per kg

    #Saftey Properties
    factor_of_saftey = attr.ib(default=1.5)

    @property
    def E(self):
        return self.elastic_modulus 

    @property
    def G(self):
        return self.shear_modulus

    @property
    def shear_modulus(self):
        '''Shear Modulus'''
        return self.E / (2.0*(1+self.poissons_ratio))

    @property
    def rho(self):
        return self.density
    @property
    def yield_stress(self):
        return self.yield_strength 
    @property
    def ultimate_stress(self):
        return self.tensile_strength_ultimate

    @inst_vectorize
    def von_mises_stress_max(self,normal_stress,shear_stress):  
        a =  (normal_stress/2.0)
        b = numpy.sqrt(a**2.0 + shear_stress**2.0)
        v1 = a - b
        v2 = a + b
        vout = v1 if abs(v1) > abs(v2) else v2
        return vout

    @property
    def allowable_stress(self):
        return self.yield_stress / self.factor_of_saftey


@otterize
class SS_316(SolidMaterial):
    name = attr.ib(default='stainless steel 316')

    #Structural Properties
    density = attr.ib(default=8000.0) #kg/m3
    elastic_modulus  = attr.ib(default=193E9) #Pa
    yield_strength  = attr.ib(default=240E6) #Pa
    tensile_strength_ultimate = attr.ib(default=550E6) #Pa
    poissons_ratio = attr.ib(default=0.30)

    #Thermal Properties
    melting_point = attr.ib(default=1370+273) #K
    maxium_service_temp = attr.ib(default=870+273) #K
    thermal_conductivity = attr.ib(default=16.3) #W/mK
    specific_heat = attr.ib(default=500) #J/kgK
    thermal_expansion = attr.ib(default = 16E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=7.4E-7) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=3.26) #dollar per kg


@otterize
class ANSI_4130(SolidMaterial):
    name = attr.ib(default='steel 4130')

    #Structural Properties
    density = attr.ib(default=7872.0) #kg/m3
    elastic_modulus  = attr.ib(default=205E9) #Pa
    yield_strength  = attr.ib(default=460E6) #Pa
    tensile_strength_ultimate = attr.ib(default=560E6) #Pa
    poissons_ratio = attr.ib(default=0.28)

    #Thermal Properties
    melting_point = attr.ib(default=1432+273) #K
    maxium_service_temp = attr.ib(default=870+273) #K
    thermal_conductivity = attr.ib(default=42.7) #W/mK
    specific_heat = attr.ib(default=477) #J/kgK
    thermal_expansion = attr.ib(default = 11.2E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=2.23E-7) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=2.92) #dollar per kg   


@otterize
class ANSI_4340(SolidMaterial):
    name = attr.ib(default='steel 4340')

    #Structural Properties
    density = attr.ib(default=7872.0) #kg/m3
    elastic_modulus  = attr.ib(default=192E9) #Pa
    yield_strength  = attr.ib(default=470E6) #Pa
    tensile_strength_ultimate = attr.ib(default=745E6) #Pa
    poissons_ratio = attr.ib(default=0.28)

    #Thermal Properties
    melting_point = attr.ib(default=1427+273) #K
    maxium_service_temp = attr.ib(default=830+273) #K
    thermal_conductivity = attr.ib(default=44.5) #W/mK
    specific_heat = attr.ib(default=475) #J/kgK
    thermal_expansion = attr.ib(default = 13.7E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=2.48E-7) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=2.23) #dollar per kg 



@otterize
class Aluminum(SolidMaterial):
    name = attr.ib(default='aluminum generic')

    #Structural Properties
    density = attr.ib(default=2680.0) #kg/m3
    elastic_modulus  = attr.ib(default=70.3E9) #Pa
    yield_strength  = attr.ib(default=86E6) #Pa
    tensile_strength_ultimate = attr.ib(default=193E6) #Pa
    poissons_ratio = attr.ib(default=0.33)

    #Thermal Properties
    melting_point = attr.ib(default=607+273) #K
    maxium_service_temp = attr.ib(default=343+273) #K
    thermal_conductivity = attr.ib(default=138) #W/mK
    specific_heat = attr.ib(default=880) #J/kgK
    thermal_expansion = attr.ib(default = 22.1E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=4.99E-7) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=1.90) #dollar per kg    


@otterize
class CarbonFiber(SolidMaterial):
    name = attr.ib(default='carbon fiber')

    #Structural Properties
    density = attr.ib(default=1600.0) #kg/m3
    elastic_modulus  = attr.ib(default=140E9) #Pa
    yield_strength  = attr.ib(default=686E6) #Pa
    tensile_strength_ultimate = attr.ib(default=919E6) #Pa
    poissons_ratio = attr.ib(default=0.33)

    #Thermal Properties
    melting_point = attr.ib(default=300+273) #K
    maxium_service_temp = attr.ib(default=150+273) #K
    thermal_conductivity = attr.ib(default=250) #W/mK
    specific_heat = attr.ib(default=1100) #J/kgK
    thermal_expansion = attr.ib(default = 14.1E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=10000) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=1.90) #dollar per kg    

@otterize
class Concrete(SolidMaterial):
    name = attr.ib(default='concrete')

    #Structural Properties
    density = attr.ib(default=2000.0) #kg/m3
    elastic_modulus  = attr.ib(default=2.92E9) #Pa
    yield_strength  = attr.ib(default=57.9E6) #Pa
    tensile_strength_ultimate = attr.ib(default=0.910E6) #Pa
    poissons_ratio = attr.ib(default=0.26)

    #Thermal Properties
    melting_point = attr.ib(default=3000+273) #K
    maxium_service_temp = attr.ib(default=3000+273) #K
    thermal_conductivity = attr.ib(default=0.5) #W/mK
    specific_heat = attr.ib(default=736) #J/kgK
    thermal_expansion = attr.ib(default = 16.41E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=1E6) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=95.44/1000.0) #dollar per kg  

@otterize
class DrySoil(SolidMaterial):
    name = attr.ib(default='dry soil')

    #Structural Properties
    density = attr.ib(default=1600.0) #kg/m3
    elastic_modulus  = attr.ib(default=70.3E9) #Pa
    yield_strength  = attr.ib(default=0.0) #Pa
    tensile_strength_ultimate = attr.ib(default=0.0) #Pa
    poissons_ratio = attr.ib(default=0.33)

    #Thermal Properties
    melting_point = attr.ib(default=1550+273) #K
    maxium_service_temp = attr.ib(default=1450+273) #K
    thermal_conductivity = attr.ib(default=0.25) #W/mK
    specific_heat = attr.ib(default=800) #J/kgK
    thermal_expansion = attr.ib(default = 16.41E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=1E6) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=44.78/1000.0) #dollar per kg        

@otterize
class WetSoil(SolidMaterial):
    name = attr.ib(default='wet soil')

    #Structural Properties
    density = attr.ib(default=2080.0) #kg/m3
    elastic_modulus  = attr.ib(default=70.3E9) #Pa
    yield_strength  = attr.ib(default=0.0) #Pa
    tensile_strength_ultimate = attr.ib(default=0.0) #Pa
    poissons_ratio = attr.ib(default=0.33)

    #Thermal Properties
    melting_point = attr.ib(default=1550+273) #K
    maxium_service_temp = attr.ib(default=1450+273) #K
    thermal_conductivity = attr.ib(default=2.75) #W/mK
    specific_heat = attr.ib(default=1632) #J/kgK
    thermal_expansion = attr.ib(default = 16.41E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=940.0) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=34.44/1000.0) #dollar per kg  

@otterize
class Rock(SolidMaterial):
    name = attr.ib(default='wet soil')

    #Structural Properties
    density = attr.ib(default=2600.0) #kg/m3
    elastic_modulus  = attr.ib(default=67E9) #Pa
    yield_strength  = attr.ib(default=13E6) #Pa
    tensile_strength_ultimate = attr.ib(default=13E6) #Pa
    poissons_ratio = attr.ib(default=0.26)

    #Thermal Properties
    melting_point = attr.ib(default=3000) #K
    maxium_service_temp = attr.ib(default=3000) #K
    thermal_conductivity = attr.ib(default=1.0) #W/mK
    specific_heat = attr.ib(default=2000) #J/kgK
    thermal_expansion = attr.ib(default = 16.41E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=1E6) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=95.44/1000.0) #dollar per kg        


@otterize
class Rubber(SolidMaterial):
    name = attr.ib(default='rubber')

    #Structural Properties
    density = attr.ib(default=1100.0) #kg/m3
    elastic_modulus  = attr.ib(default=0.1E9) #Pa
    yield_strength  = attr.ib(default=0.248E6) #Pa
    tensile_strength_ultimate = attr.ib(default=0.5E6) #Pa
    poissons_ratio = attr.ib(default=0.33)

    #Thermal Properties
    melting_point = attr.ib(default=600+273) #K
    maxium_service_temp = attr.ib(default=300+273) #K
    thermal_conductivity = attr.ib(default=0.108) #W/mK
    specific_heat = attr.ib(default=2005) #J/kgK
    thermal_expansion = attr.ib(default = 100E-6) #m/mK
 
    #Electrical Properties
    electrical_resistitivity = attr.ib(default=1E13) #ohm-m

    #Economic Properties
    cost_per_kg = attr.ib(default=40.0/1000.0) #dollar per kg  


# @attr.s
# class AL_6063(SolidMaterial):
#     name = attr.ib(default='aluminum 6063')

#     #Structural Properties
#     density = attr.ib(default=2700.0) #kg/m3
#     elastic_modulus  = attr.ib(default=192E9) #Pa
#     tensile_strength_yield = attr.ib(default=470E6) #Pa
#     tensile_strength_ultimate = attr.ib(default=745E6) #Pa
#     poissons_ratio = attr.ib(default=0.28)

#     #Thermal Properties
#     melting_point = attr.ib(default=1427+273) #K
#     maxium_service_temp = attr.ib(default=830+273) #K
#     thermal_conductivity = attr.ib(default=44.5) #W/mK
#     specific_heat = attr.ib(default=475) #J/kgK
#     thermal_expansion = attr.ib(default = 13.7E-6) #m/mK
 
#     #Electrical Properties
#     electrical_resistitivity = attr.ib(default=2.48E-7) #ohm-m

#     #Economic Properties
#     cost_per_kg = attr.ib(default=1.90) #dollar per kg


# @attr.s
# class AL_7075(SolidMaterial):
#     name = attr.ib(default='aluminum 7075')

#     #Structural Properties
#     density = attr.ib(default=2700.0) #kg/m3
#     elastic_modulus  = attr.ib(default=192E9) #Pa
#     tensile_strength_yield = attr.ib(default=470E6) #Pa
#     tensile_strength_ultimate = attr.ib(default=745E6) #Pa
#     poissons_ratio = attr.ib(default=0.28)

#     #Thermal Properties
#     melting_point = attr.ib(default=1427+273) #K
#     maxium_service_temp = attr.ib(default=830+273) #K
#     thermal_conductivity = attr.ib(default=44.5) #W/mK
#     specific_heat = attr.ib(default=475) #J/kgK
#     thermal_expansion = attr.ib(default = 13.7E-6) #m/mK
 
#     #Electrical Properties
#     electrical_resistitivity = attr.ib(default=2.48E-7) #ohm-m

#     #Economic Properties
#     cost_per_kg = attr.ib(default=1.90) #dollar per kg            





ALL_MATERIALS = [mat for name,mat in inspect.getmembers(sys.modules[__name__]) if inspect.isclass(mat) and issubclass(mat,SolidMaterial) and mat is not SolidMaterial ]
