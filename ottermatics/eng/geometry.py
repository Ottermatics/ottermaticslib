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
import functools

# generic cross sections from
# https://mechanicalbase.com/area-moment-of-inertia-calculator-of-certain-cross-sectional-shapes/


def conver_np(inpt):
    if isinstance(inpt,np.ndarray):
        return inpt
    elif isinstance(inpt,(list,tuple)):
        return np.array(inpt)
    elif isinstance(inpt,(int,float)):
        return np.array([inpt])
    else:
        raise ValueError(f'got non numpy array/float: {inpt}')



@attrs.define(slots=False)
class ParametricSpline:
    """a multivariate spline defined by uniform length vector input, with points P1,P2 and their slopes P1ds,P2ds"""

    P1=attrs.field(converter=conver_np)
    P2=attrs.field(converter=conver_np)
    P1ds=attrs.field(converter=conver_np)
    P2ds=attrs.field(converter=conver_np)

    def __attrs_post_init__(self):
        assert len(self.P1) == len(self.P2)
        assert len(self.P1ds) == len(self.P2ds)
        assert len(self.P1) == len(self.P1ds)

    @functools.cached_property
    def a0(self):
        return self.P1

    @functools.cached_property
    def a1(self):
        return self.P1ds

    @functools.cached_property
    def a2(self):
        return 3*(self.P2-self.P1 ) - 2*self.P1ds - self.P2ds
    
    @functools.cached_property
    def a3(self):
        return (self.P2ds - self.P1ds - 2*self.a2)/3.
    
    def coords(self,s:float):
        if isinstance(s,(float,int)):
            assert s <= 1
            assert 0 <= s
            return self._coords(s)
        elif isinstance(s,(list,tuple,numpy.ndarray)):
            assert max(s) <= 1
            assert min(s) >= 0
            ol = [self._coords(si) for si in s]
            return numpy.array(ol)
        else:
            raise ValueError(f'non array/float input {s}')

    def _coords(self,s:float):
        return self.a0 + self.a1*s + self.a2*s**2 +self.a3*s**3



# TODO: cache SectionProperty sections and develop auto-mesh refinement system.

@otterize
class Profile2D(Configuration):
    name: str = attr.ib(default="generic cross section")

    # provide relative interface over
    y_bounds: tuple = None
    x_bounds: tuple = None

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

    def __on_init__(self):
        self.y_bounds = (self.h / 2, -self.h / 2)
        self.x_bounds = (self.b / 2, -self.b / 2)

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

    def __on_init__(self):
        self.y_bounds = (self.h / 2, -self.h / 2)
        self.x_bounds = (self.b / 2, -self.b / 2)

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

    def __on_init__(self):
        self.x_bounds = self.y_bounds = (self.d / 2, -self.d / 2)

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

    def __on_init__(self):
        self.y_bounds = (self.d / 2, -self.d / 2)
        self.x_bounds = (self.d / 2, -self.d / 2)

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

    coarse: bool = attrs.field(default=False)

    _geo: Geometry
    _A: float

    def __on_init__(self):
        self.init_with_material(self.material)

    def init_with_material(self, material=None):
        if hasattr(self, "_sec"):
            raise Exception(f"already initalized!")
        self._geo = Geometry(self.shape, self.material)
        self._mesh = self._geo.create_mesh([self.mesh_size], coarse=self.coarse)
        self._sec = Section(self._geo)
        self._sec.calculate_geometric_properties()
        self._sec.calculate_warping_properties()
        self._sec.calculate_frame_properties()

        self._A = self._sec.get_area()
        self._Ixx, self._Iyy, self._Ixy = self._sec.get_ic()
        self._J = self._sec.get_j()

        self.calculate_bounds()

    def calculate_bounds(self):
        self.info(f"calculating shape bounds!")
        xcg, ycg = self._geo.calculate_centroid()
        minx, maxx, miny, maxy = self._geo.calculate_extents()
        self.y_bounds = (miny - ycg, maxy - ycg)
        self.x_bounds = (minx - xcg, maxx - xcg)

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
