from ottermatics.configuration import Configuration, otterize
from ottermatics.properties import *
from ottermatics.common import *
import matplotlib
import random
import attr
import numpy
import inspect
import sys
import uuid

# One Material To Merge Them All
from PyNite import Material as PyNiteMat
from sectionproperties.pre.pre import Material as SecMat

SectionMaterial = SecMat.mro()[0]

ALL_MATERIALS = []
METALS = []
PLASTICS = []
CERAMICS = []


CMAP = matplotlib.cm.get_cmap("viridis")


def random_color():
    return CMAP(random.randint(0, 255))


@otterize
class SolidMaterial(SectionMaterial, PyNiteMat.Material, Configuration):
    """A class to hold physical properties of solid structural materials and act as both a section property material and a pynite material"""

    __metaclass__ = SecMat

    name: str = attr.ib(default="solid material")
    color: float = attr.ib(default=random_color())

    # Structural Properties
    density: float = attr.ib(default=1.0)
    elastic_modulus: float = attr.ib(default=1e8)  # Pa
    in_shear_modulus: float = attr.ib(default=None)
    yield_strength: float = attr.ib(default=1e6)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=2e6)  # Pa
    hardness: float = attr.ib(default=10)  # rockwell
    izod: float = attr.ib(default=100)
    poissons_ratio: float = attr.ib(default=0.30)

    # Thermal Properties
    melting_point: float = attr.ib(default=1000 + 273)  # K
    maxium_service_temp: float = attr.ib(default=500 + 273)  # K
    thermal_conductivity: float = attr.ib(default=10)  # W/mK
    specific_heat: float = attr.ib(default=1000)  # J/kgK
    thermal_expansion: float = attr.ib(default=10e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=1e-8)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=1.0)  # dollar per kg

    # Saftey Properties
    factor_of_saftey: float = attr.ib(default=1.5)

    _unique_id: str = None

    @classmethod
    def pre_compile(cls):
        cls.color = random_color()

    @property
    def E(self) -> float:
        return self.elastic_modulus

    @property
    def G(self) -> float:
        return self.shear_modulus

    @property
    def nu(self) -> float:
        return self.poissons_ratio

    @property
    def rho(self) -> float:
        return self.density

    @property
    def shear_modulus(self) -> float:
        """Shear Modulus"""
        if self.in_shear_modulus:
            return self.in_shear_modulus
        # hello defaults
        return self.E / (2.0 * (1 + self.poissons_ratio))

    @property
    def yield_stress(self) -> float:
        return self.yield_strength

    @property
    def ultimate_stress(self) -> float:
        return self.tensile_strength_ultimate

    @inst_vectorize
    def von_mises_stress_max(self, normal_stress, shear_stress):
        a = normal_stress / 2.0
        b = numpy.sqrt(a**2.0 + shear_stress**2.0)
        v1 = a - b
        v2 = a + b
        vout = v1 if abs(v1) > abs(v2) else v2
        return vout

    @property
    def allowable_stress(self) -> float:
        return self.yield_stress / self.factor_of_saftey

    @property
    def unique_id(self):
        if self._unique_id is None:
            uid = str(uuid.uuid4())
            self._unique_id = f"{self.name}_{uid}"
        return self._unique_id


@otterize
class SS_316(SolidMaterial):
    name: str = attr.ib(default="stainless steel 316")

    # Structural Properties
    density: float = attr.ib(default=8000.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=193e9)  # Pa
    yield_strength: float = attr.ib(default=240e6)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=550e6)  # Pa
    poissons_ratio: float = attr.ib(default=0.30)

    # Thermal Properties
    melting_point: float = attr.ib(default=1370 + 273)  # K
    maxium_service_temp: float = attr.ib(default=870 + 273)  # K
    thermal_conductivity: float = attr.ib(default=16.3)  # W/mK
    specific_heat: float = attr.ib(default=500)  # J/kgK
    thermal_expansion: float = attr.ib(default=16e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=7.4e-7)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=3.26)  # dollar per kg


@otterize
class ANSI_4130(SolidMaterial):
    name: str = attr.ib(default="steel 4130")

    # Structural Properties
    density: float = attr.ib(default=7872.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=205e9)  # Pa
    yield_strength: float = attr.ib(default=460e6)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=560e6)  # Pa
    poissons_ratio: float = attr.ib(default=0.28)

    # Thermal Properties
    melting_point: float = attr.ib(default=1432 + 273)  # K
    maxium_service_temp: float = attr.ib(default=870 + 273)  # K
    thermal_conductivity: float = attr.ib(default=42.7)  # W/mK
    specific_heat: float = attr.ib(default=477)  # J/kgK
    thermal_expansion: float = attr.ib(default=11.2e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=2.23e-7)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=2.92)  # dollar per kg


@otterize
class ANSI_4340(SolidMaterial):
    name: str = attr.ib(default="steel 4340")

    # Structural Properties
    density: float = attr.ib(default=7872.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=192e9)  # Pa
    yield_strength: float = attr.ib(default=470e6)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=745e6)  # Pa
    poissons_ratio: float = attr.ib(default=0.28)

    # Thermal Properties
    melting_point: float = attr.ib(default=1427 + 273)  # K
    maxium_service_temp: float = attr.ib(default=830 + 273)  # K
    thermal_conductivity: float = attr.ib(default=44.5)  # W/mK
    specific_heat: float = attr.ib(default=475)  # J/kgK
    thermal_expansion: float = attr.ib(default=13.7e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=2.48e-7)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=2.23)  # dollar per kg


@otterize
class Aluminum(SolidMaterial):
    name: str = attr.ib(default="aluminum generic")

    # Structural Properties
    density: float = attr.ib(default=2680.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=70.3e9)  # Pa
    yield_strength: float = attr.ib(default=240e6)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=290e6)  # Pa
    poissons_ratio: float = attr.ib(default=0.33)

    # Thermal Properties
    melting_point: float = attr.ib(default=607 + 273)  # K
    maxium_service_temp: float = attr.ib(default=343 + 273)  # K
    thermal_conductivity: float = attr.ib(default=138)  # W/mK
    specific_heat: float = attr.ib(default=880)  # J/kgK
    thermal_expansion: float = attr.ib(default=22.1e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=4.99e-7)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=1.90)  # dollar per kg


@otterize
class CarbonFiber(SolidMaterial):
    name: str = attr.ib(default="carbon fiber")

    # Structural Properties
    density: float = attr.ib(default=1600.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=140e9)  # Pa
    yield_strength: float = attr.ib(default=686e6)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=919e6)  # Pa
    poissons_ratio: float = attr.ib(default=0.33)

    # Thermal Properties
    melting_point: float = attr.ib(default=300 + 273)  # K
    maxium_service_temp: float = attr.ib(default=150 + 273)  # K
    thermal_conductivity: float = attr.ib(default=250)  # W/mK
    specific_heat: float = attr.ib(default=1100)  # J/kgK
    thermal_expansion: float = attr.ib(default=14.1e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=10000)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=1.90)  # dollar per kg


@otterize
class Concrete(SolidMaterial):
    name: str = attr.ib(default="concrete")

    # Structural Properties
    density: float = attr.ib(default=2000.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=2.92e9)  # Pa
    yield_strength: float = attr.ib(default=57.9e6)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=0.910e6)  # Pa
    poissons_ratio: float = attr.ib(default=0.26)

    # Thermal Properties
    melting_point: float = attr.ib(default=3000 + 273)  # K
    maxium_service_temp: float = attr.ib(default=3000 + 273)  # K
    thermal_conductivity: float = attr.ib(default=0.5)  # W/mK
    specific_heat: float = attr.ib(default=736)  # J/kgK
    thermal_expansion: float = attr.ib(default=16.41e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=1e6)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=95.44 / 1000.0)  # dollar per kg


@otterize
class DrySoil(SolidMaterial):
    name: str = attr.ib(default="dry soil")

    # Structural Properties
    density: float = attr.ib(default=1600.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=70.3e9)  # Pa
    yield_strength: float = attr.ib(default=0.0)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=0.0)  # Pa
    poissons_ratio: float = attr.ib(default=0.33)

    # Thermal Properties
    melting_point: float = attr.ib(default=1550 + 273)  # K
    maxium_service_temp: float = attr.ib(default=1450 + 273)  # K
    thermal_conductivity: float = attr.ib(default=0.25)  # W/mK
    specific_heat: float = attr.ib(default=800)  # J/kgK
    thermal_expansion: float = attr.ib(default=16.41e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=1e6)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=44.78 / 1000.0)  # dollar per kg


@otterize
class WetSoil(SolidMaterial):
    name: str = attr.ib(default="wet soil")

    # Structural Properties
    density: float = attr.ib(default=2080.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=70.3e9)  # Pa
    yield_strength: float = attr.ib(default=0.0)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=0.0)  # Pa
    poissons_ratio: float = attr.ib(default=0.33)

    # Thermal Properties
    melting_point: float = attr.ib(default=1550 + 273)  # K
    maxium_service_temp: float = attr.ib(default=1450 + 273)  # K
    thermal_conductivity: float = attr.ib(default=2.75)  # W/mK
    specific_heat: float = attr.ib(default=1632)  # J/kgK
    thermal_expansion: float = attr.ib(default=16.41e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=940.0)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=34.44 / 1000.0)  # dollar per kg


@otterize
class Rock(SolidMaterial):
    name: str = attr.ib(default="wet soil")

    # Structural Properties
    density: float = attr.ib(default=2600.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=67e9)  # Pa
    yield_strength: float = attr.ib(default=13e6)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=13e6)  # Pa
    poissons_ratio: float = attr.ib(default=0.26)

    # Thermal Properties
    melting_point: float = attr.ib(default=3000)  # K
    maxium_service_temp: float = attr.ib(default=3000)  # K
    thermal_conductivity: float = attr.ib(default=1.0)  # W/mK
    specific_heat: float = attr.ib(default=2000)  # J/kgK
    thermal_expansion: float = attr.ib(default=16.41e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=1e6)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=50.44 / 1000.0)  # dollar per kg


@otterize
class Rubber(SolidMaterial):
    name: str = attr.ib(default="rubber")

    # Structural Properties
    density: float = attr.ib(default=1100.0)  # kg/m3
    elastic_modulus: float = attr.ib(default=0.1e9)  # Pa
    yield_strength: float = attr.ib(default=0.248e6)  # Pa
    tensile_strength_ultimate: float = attr.ib(default=0.5e6)  # Pa
    poissons_ratio: float = attr.ib(default=0.33)

    # Thermal Properties
    melting_point: float = attr.ib(default=600 + 273)  # K
    maxium_service_temp: float = attr.ib(default=300 + 273)  # K
    thermal_conductivity: float = attr.ib(default=0.108)  # W/mK
    specific_heat: float = attr.ib(default=2005)  # J/kgK
    thermal_expansion: float = attr.ib(default=100e-6)  # m/mK

    # Electrical Properties
    electrical_resistitivity: float = attr.ib(default=1e13)  # ohm-m

    # Economic Properties
    cost_per_kg: float = attr.ib(default=40.0 / 1000.0)  # dollar per kg


# @attr.s
# class AL_6063(SolidMaterial):
#     name:str = attr.ib(default='aluminum 6063')

#     #Structural Properties
#     density:float = attr.ib(default=2700.0) #kg/m3
#     elastic_modulus:float  = attr.ib(default=192E9) #Pa
#     tensile_strength_yield = attr.ib(default=470E6) #Pa
#     tensile_strength_ultimate:float = attr.ib(default=745E6) #Pa
#     poissons_ratio:float = attr.ib(default=0.28)

#     #Thermal Properties
#     melting_point:float = attr.ib(default=1427+273) #K
#     maxium_service_temp:float = attr.ib(default=830+273) #K
#     thermal_conductivity:float = attr.ib(default=44.5) #W/mK
#     specific_heat:float = attr.ib(default=475) #J/kgK
#     thermal_expansion:float = attr.ib(default = 13.7E-6) #m/mK

#     #Electrical Properties
#     electrical_resistitivity:float = attr.ib(default=2.48E-7) #ohm-m

#     #Economic Properties
#     cost_per_kg:float = attr.ib(default=1.90) #dollar per kg


# @attr.s
# class AL_7075(SolidMaterial):
#     name:str = attr.ib(default='aluminum 7075')

#     #Structural Properties
#     density:float = attr.ib(default=2700.0) #kg/m3
#     elastic_modulus:float  = attr.ib(default=192E9) #Pa
#     tensile_strength_yield = attr.ib(default=470E6) #Pa
#     tensile_strength_ultimate:float = attr.ib(default=745E6) #Pa
#     poissons_ratio:float = attr.ib(default=0.28)

#     #Thermal Properties
#     melting_point:float = attr.ib(default=1427+273) #K
#     maxium_service_temp:float = attr.ib(default=830+273) #K
#     thermal_conductivity:float = attr.ib(default=44.5) #W/mK
#     specific_heat:float = attr.ib(default=475) #J/kgK
#     thermal_expansion:float = attr.ib(default = 13.7E-6) #m/mK

#     #Electrical Properties
#     electrical_resistitivity:float = attr.ib(default=2.48E-7) #ohm-m

#     #Economic Properties
#     cost_per_kg:float = attr.ib(default=1.90) #dollar per kg


ALL_MATERIALS = [
    mat
    for name, mat in inspect.getmembers(sys.modules[__name__])
    if inspect.isclass(mat)
    and issubclass(mat, SolidMaterial)
    and mat is not SolidMaterial
]
