from ottermatics.configuration import Configuration, otterize
import numpy
import attr

#generic cross sections from 
#https://mechanicalbase.com/area-moment-of-inertia-calculator-of-certain-cross-sectional-shapes/

'''These exist as in interface to sectionproperties from PyNite'''

@otterize
class Profile2D(Configuration):
    name = attr.ib(default='generic cross section')
    
    @property
    def A(self):
        return 0

    @property
    def Ixx(self):
        return 0

    @property
    def Iyy(self):
        return 0

    @property
    def J(self):
        return 0   

    def get_ic(self):
        return (self.Ixx, self.Iyy, 0.0)

    def get_j(self):
        return self.J

    def get_area(self):
        return self.A

    def get_c(self):
        return (0,0)

    def display_results(self):
        self.info('mock section, no results to display')

    def plot_mesh(self):
        self.info('mock section, no mesh to plot')

    def calculate_stress(self,N,Vx,Vy,Mxx,Myy,M11,M22,Mzz):
        #TODO: Implement stress object
        return None

    def calculate_geometric_properties(self):
        pass
    
    def calculate_warping_properties(self):
        pass
            

@otterize
class Rectangle(Profile2D):
    b = attr.ib()
    h = attr.ib()
    name = attr.ib(default='rectangular section')

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
        return (self.h * self.b) * (self.b**2.0 + self.h**2.0)/12


@otterize
class Triangle(Profile2D):
    b = attr.ib()
    h = attr.ib()
    name = attr.ib(default='rectangular section')

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
        return (self.h * self.b) * (self.b**2.0 + self.h**2.0)/12


@otterize
class Circle(Profile2D):
    d = attr.ib()
    name = attr.ib(default='rectangular section')

    @property
    def A(self):
        return (self.d/2.0)**2.0 * numpy.pi

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
    d = attr.ib()
    t = attr.ib()
    name = attr.ib(default='rectangular section')

    @property
    def di(self):
        return self.d - self.t * 2

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

ALL_CROSSSECTIONS = [cs for cs in locals() if type(cs) is type and issubclass(cs,Profile2D)]