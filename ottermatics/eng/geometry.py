"""These exist as in interface to sectionproperties from PyNite"""

from ottermatics.configuration import Configuration, otterize
from ottermatics.properties import (
    cached_system_property,
    system_property,
    instance_cached,
)
import numpy
import attr, attrs

from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.pre import Material as sec_material
from sectionproperties.analysis.section import Section
import sectionproperties.pre.library.primitive_sections as sections
import numpy as np
import shapely
import attr, attrs

# generic cross sections from
# https://mechanicalbase.com/area-moment-of-inertia-calculator-of-certain-cross-sectional-shapes/


#TODO: cache SectionProperty sections and develop auto-mesh refinement system.


@otterize
class Profile2D(Configuration):
    name: str = attr.ib(default="generic cross section")

    @property
    def A(self):
        return 0

    @property
    def Ao(self):
        """outside area, over ride for hallow sections"""
        return self.A

    @property
    def Ixx(self):
        return 0

    @property
    def Iyy(self):
        return 0

    @property
    def J(self):
        return 0

    def display_results(self):
        self.info("mock section, no results to display")

    def plot_mesh(self):
        self.info("mock section, no mesh to plot")

    def calculate_stress(self, N, Vx, Vy, Mxx, Myy, M11, M22, Mzz):
        # TODO: Implement stress object, make fake mesh, and mock?
        # sigma_n = N / self.A
        # sigma_bx = self.Myy * self.max_y / self.Ixx
        # sigma_by = self.Mxx * self.max_x / self.Iyy
        # self.warning(f'calculating stress in simple profile!')
        return 0.0


@otterize
class Rectangle(Profile2D):
    """models rectangle with base b, and height h"""

    b: float = attr.ib()
    h: float = attr.ib()
    name: str = attr.ib(default="rectangular section")

    @property
    def A(self):
        return self.h * self.b

    @property
    def Ixx(self):
        return self.b * self.h**3.0 / 12.0

    @property
    def Iyy(self):
        return self.h * self.b**3.0 / 12.0

    @property
    def J(self):
        return (self.h * self.b) * (self.b**2.0 + self.h**2.0) / 12


@otterize
class Triangle(Profile2D):
    """models a triangle with base, b and height h"""

    b: float = attr.ib()
    h: float = attr.ib()
    name: str = attr.ib(default="rectangular section")

    @property
    def A(self):
        return self.h * self.b / 2.0

    @property
    def Ixx(self):
        return self.b * self.h**3.0 / 36.0

    @property
    def Iyy(self):
        return self.h * self.b**3.0 / 36.0

    @property
    def J(self):
        return (self.h * self.b) * (self.b**2.0 + self.h**2.0) / 12


@otterize
class Circle(Profile2D):
    """models a solid circle with diameter d"""

    d: float = attr.ib()
    name: str = attr.ib(default="rectangular section")

    @property
    def A(self):
        return (self.d / 2.0) ** 2.0 * numpy.pi

    @property
    def Ixx(self):
        return numpy.pi * (self.d**4.0 / 64.0)

    @property
    def Iyy(self):
        return numpy.pi * (self.d**4.0 / 64.0)

    @property
    def J(self):
        return numpy.pi * (self.d**4.0 / 32.0)


@otterize
class HollowCircle(Profile2D):
    """models a hollow circle with diameter d and thickness t"""

    d: float = attr.ib()
    t: float = attr.ib()
    name: str = attr.ib(default="rectangular section")

    @property
    def di(self):
        return self.d - self.t * 2

    @property
    def Ao(self):
        """outside area, over ride for hallow sections"""
        return (self.d**2.0) / 4.0 * numpy.pi

    @property
    def A(self):
        return (self.d**2.0 - self.di**2.0) / 4.0 * numpy.pi

    @property
    def Ixx(self):
        return numpy.pi * ((self.d**4.0 - self.di**4.0) / 64.0)

    @property
    def Iyy(self):
        return numpy.pi * ((self.d**4.0 - self.di**4.0) / 64.0)

    @property
    def J(self):
        return numpy.pi * ((self.d**4.0 - self.di**4.0) / 32.0)


# ADVANCED CUSTOM SECTIONS
def get_mesh_size(inst):
    x, y = inst.shape.exterior.coords.xy
    dx = max(x) - min(x)
    dy = max(y) - min(y)
    ddx = min(np.diff(x))
    ddy = min(np.diff(y))
    return min([dx / 100, dy / 100, ddx, ddy]) / 2


@otterize
class ShapelySection(Profile2D):
    """a 2D profile that takes a shapely section to calculate section properties"""

    name: str = attrs.field(default="shapely section")
    shape: shapely.Polygon = attrs.field(
        validator=attr.validators.instance_of(shapely.Polygon)
    )
    mesh_size: float = attrs.field(default=attrs.Factory(get_mesh_size, True))
    material: sec_material = attrs.field(default=None)

    coarse:bool = attrs.field(default=False)

    _geo: Geometry
    _A: float

    def __on_init__(self):
        self.init_with_material(self.material)

    def init_with_material(self, material=None):
        if hasattr(self,'_sec'):
            raise Exception(f'already initalized!')
        self._geo = Geometry(self.shape, self.material)
        self._mesh = self._geo.create_mesh([ self.mesh_size ],coarse=self.coarse)
        self._sec = Section(self._geo)
        self._sec.calculate_geometric_properties()
        self._sec.calculate_warping_properties()
        self._sec.calculate_frame_properties()

        self._A = self._sec.get_area()
        self._Ixx,self._Iyy,self._Ixy = self._sec.get_ic()
        self._J = self._sec.get_j()

    @property
    def A(self):
        return self._A

    @property
    def Ao(self):
        """outside area, over ride for hallow sections"""
        return self.A

    @property
    def Ixx(self):
        return self._Ixx

    @property
    def Iyy(self):
        return self._Iyy

    @property
    def J(self):
        return self._J

    @property
    def Ixy(self):
        return self._Ixy

    def display_results(self):
        self.info("mock section, no results to display")

    def plot_mesh(self):
        self._sec.display_mesh_info()

    def calculate_stress(self, N, Vx, Vy, Mxx, Myy, M11, M22, Mzz):
        # TODO: Implement stress object, make fake mesh, and mock?
        raise NotImplemented(f"implement von mises max picker")

    def plot_mesh(self):
        return self._sec.plot_centroids()





ALL_CROSSSECTIONS = [
    cs for cs in locals() if type(cs) is type and issubclass(cs, Profile2D)
]
