from ottermatics import Configuration,inst_vectorize
import attr
import numpy

ALL_MATERIALS = []
METALS = []
PLASTICS = []
CERAMICS = []


@attr.s
class SolidMaterial:
    '''A class to hold physical properties of solid structural materials'''
    name = attr.ib(default='solid material')

    #Structural Properties
    density = attr.ib(default=1.0)
    modulus_of_elasticity = attr.ib(default=1E8) #Pa
    tensile_strength_yield = attr.ib(default=1E6) #Pa
    tensile_strength_ultimate = attr.ib(default=2E6) #Pa
    hardness = attr.ib(default=10) #rockwell
    izod = attr.ib(default=100)
    poisson_ratio = attr.ib(default=0.30)

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

    @property
    def E(self):
        return self.modulus_of_elasticity

    @property
    def G(self):
        '''Shear Modulus'''
        return self.E / (2.0*(1+self.poisson_ratio))

    @property
    def rho(self):
        return self.density
    @property
    def yield_stress(self):
        return self.tensile_strength_yield
    @property
    def ultimate_stress(self):
        return self.tensile_strength_ultimate

    def von_mises_stress_max(self,normal_stress,shear_stress):  
        a =  (normal_stress/2.0)
        b = numpy.sqrt(a**2.0 + shear_stress**2.0)
        v1 = a - b
        v2 = a + b
        vout = v1 if abs(v1) > abs(v2) else v2
        return vout

@attr.s
class SS_316(SolidMaterial):
    name = attr.ib(default='stainless steel 316')

    #Structural Properties
    density = attr.ib(default=8000.0) #kg/m3
    modulus_of_elasticity = attr.ib(default=193E9) #Pa
    tensile_strength_yield = attr.ib(default=240E6) #Pa
    tensile_strength_ultimate = attr.ib(default=550E6) #Pa
    poisson_ratio = attr.ib(default=0.30)

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

@attr.s
class ANSI_4130(SolidMaterial):
    name = attr.ib(default='steel 4130')

    #Structural Properties
    density = attr.ib(default=7872.0) #kg/m3
    modulus_of_elasticity = attr.ib(default=205E9) #Pa
    tensile_strength_yield = attr.ib(default=460E6) #Pa
    tensile_strength_ultimate = attr.ib(default=560E6) #Pa
    poisson_ratio = attr.ib(default=0.28)

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


@attr.s
class ANSI_4340(SolidMaterial):
    name = attr.ib(default='steel 4340')

    #Structural Properties
    density = attr.ib(default=7872.0) #kg/m3
    modulus_of_elasticity = attr.ib(default=192E9) #Pa
    tensile_strength_yield = attr.ib(default=470E6) #Pa
    tensile_strength_ultimate = attr.ib(default=745E6) #Pa
    poisson_ratio = attr.ib(default=0.28)

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



@attr.s
class Aluminum(SolidMaterial):
    name = attr.ib(default='aluminum generic')

    #Structural Properties
    density = attr.ib(default=2680.0) #kg/m3
    modulus_of_elasticity = attr.ib(default=70.3E9) #Pa
    tensile_strength_yield = attr.ib(default=86E6) #Pa
    tensile_strength_ultimate = attr.ib(default=193E6) #Pa
    poisson_ratio = attr.ib(default=0.33)

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

# @attr.s
# class AL_6063(SolidMaterial):
#     name = attr.ib(default='aluminum 6063')

#     #Structural Properties
#     density = attr.ib(default=2700.0) #kg/m3
#     modulus_of_elasticity = attr.ib(default=192E9) #Pa
#     tensile_strength_yield = attr.ib(default=470E6) #Pa
#     tensile_strength_ultimate = attr.ib(default=745E6) #Pa
#     poisson_ratio = attr.ib(default=0.28)

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
#     modulus_of_elasticity = attr.ib(default=192E9) #Pa
#     tensile_strength_yield = attr.ib(default=470E6) #Pa
#     tensile_strength_ultimate = attr.ib(default=745E6) #Pa
#     poisson_ratio = attr.ib(default=0.28)

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