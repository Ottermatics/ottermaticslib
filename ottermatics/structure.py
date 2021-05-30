import attr
import numpy
import functools
import shapely
import pandas
from shapely.geometry import Polygon, Point

from ottermatics.tabulation import TABLE_TYPES, NUMERIC_VALIDATOR
from ottermatics.configuration import otterize, Configuration
from ottermatics.components import Component
from ottermatics.solid_materials import *

import sectionproperties
import sectionproperties.pre.sections as sections
from sectionproperties.analysis.cross_section import CrossSection

import PyNite as pynite
from PyNite import Visualization

SECTIONS = {k:v for k,v in filter( lambda kv: issubclass(kv[1],sectionproperties.pre.sections.Geometry) if type(kv[1])  is type else False , sections.__dict__.items()) }



nonetype = type(None)

@otterize
class Structure(Component):
    '''A integration between sectionproperties and PyNite, with a focus on ease of use

    Right now we just need an integration between Sections+Materials and Members, to find, CG, and inertial components

    Possible Future Additions:
    1) Structure Motion (free body ie vehicle , multi-strucutre ie robots)
    '''
    frame = None
    _beams = None

    def __on_init__(self):
        self.frame = pynite.FEModel3D()
        self._beams = {} #this is for us!

    @property
    def nodes(self):
        return self.frame.Nodes
    
    @property
    def members(self):
        return self.frame.Members

    @property
    def beams(self):
        return self._beams

    def add_node(self,name,x,y,z):
        self.frame.AddNode(name,x,y,z)

    def add_constraint(self,node,con_DX=False, con_DY=False, con_DZ=False, con_RX=False, con_RY=False, con_RZ=False):
        self.frame.DefineSupport(node,con_DX, con_DY, con_DZ, con_RX, con_RY, con_RZ)

    def add_member(self,name,node1,node2,section,material):
        assert node1 in self.nodes 
        assert node2 in self.nodes
        
        B = beam = Beam(self,name,material,section)
        self._beams[name] = beam

        self.frame.AddMember(name, node1, node2, B.E, B.G, B.Iy, B.Ix, B.J, B.A)

        return beam

    def add_member_with(self,name,node1,node2,E, G, Iy, Ix, J, A):
        assert node1 in self.nodes 
        assert node2 in self.nodes
        
        material = SolidMaterial(density=0,elastic_modulus=E)
        B = beam = Beam(self,name,material,in_Iy=Iy,in_Ix=Ix,in_J=J,in_A=A,section=None)
        self._beams[name] = beam

        self.frame.AddMember(name, node1, node2, E, G, Iy, Ix, J, A)

        return beam        

    def analyze(self,**kwargs):
        return self.frame.Analyze(**kwargs)

    @property
    def cog(self):
        XM = sum([bm.mass * bm.centroid3d for bm in self.beams.values()])
        return XM / self.mass

    @property
    def mass(self):
        return sum([bm.mass for bm in self.beams.values()])

    def visulize(self,**kwargs):
        Visualization.RenderModel(self.frame,**kwargs)

    @property
    def node_dataframes(self):
        out = {}
        for case in self.frame.LoadCombos:
            rows = []
            for node in st.nodes.values():
                row = { 'dx':node.DX[case],'dy':node.DY[case],'dz':node.DZ[case], 'rx': node.RX[case],'ry':node.RY[case],'rz':node.RZ[case],'rxfx':node.RxnFX[case],'rxfy':node.RxnFY[case],'rxfz':node.RxnFZ[case],'rxmx':node.RxnMX[case],'rxmy':node.RxnMY[case],'rxmz':node.RxnMZ[case]}
                rows.append(row)
            
            out[case] = pandas.DataFrame(rows)
        return out
        



@otterize
class Beam(Component):
    '''Beam is a wrapper for emergent useful properties of the structure'''
    structure = attr.ib() #parent structure, will be in its _beams
    name = attr.ib()
    material = attr.ib(validator=attr.validators.instance_of(SolidMaterial))
    section = attr.ib(validator=attr.validators.instance_of((sectionproperties.pre.sections.Geometry,type(None))))

    mesh_size = attr.ib(default=3)

    in_Iy=attr.ib(default=None,validator=attr.validators.instance_of((int,float,nonetype)))
    in_Ix=attr.ib(default=None,validator=attr.validators.instance_of((int,float,nonetype))) 
    in_J=attr.ib(default=None,validator=attr.validators.instance_of((int,float,nonetype))) 
    in_A=attr.ib(default=None,validator=attr.validators.instance_of((int,float,nonetype)))  
    
    _L = None
    _section_properties = None
    _ITensor = None    

    def __on_init__(self):
        self._skip_attr = ['mesh_size']

        self.debug('determining section properties...')
        if self.section is not None:
            self._mesh = self.section.create_mesh([self.mesh_size])
            self._section_properties = CrossSection(self.section, self._mesh) #no material here
            self._section_properties.calculate_geometric_properties()
            self._section_properties.calculate_warping_properties()
        else:
            assert all([val is not None for val in (self.in_Iy,self.in_Ix,self.in_J,self.in_A)])

    @property
    def L(self):
        return self.length

    @property
    def length(self):
        if self._L is None:
            self._L = self.member.L()
        return self._L

    @property
    def member(self):
        return self.structure.members[self.name]

    @functools.cached_property
    def n1(self):
        return self.member.iNode

    @functools.cached_property
    def n2(self):
        return self.member.jNode

    @functools.cached_property
    def P1(self):
        return numpy.array([self.n1.X,self.n1.Y,self.n1.Z])

    @functools.cached_property
    def P2(self):
        return numpy.array([self.n2.X,self.n2.Y,self.n2.Z])

    @functools.cached_property
    def E(self):
        return self.material.E

    @functools.cached_property
    def G(self):
        return self.material.G        

    @functools.cached_property
    def ITensor(self):
        if self._ITensor is None:
            (ixx_c, iyy_c, ixy_c) = self._section_properties.get_ic()
            _ITensor = [[ixx_c, ixy_c],
                        [ixy_c, iyy_c]]
            self._ITensor = numpy.array(_ITensor)
        return self._ITensor
            

    @property
    def Iy(self):
        if self.in_Iy is None:
            self.in_Iy = self.ITensor[1,1]
        return self.in_Iy

    @property
    def Ix(self):
        if self.in_Ix is None:
            self.in_Ix = self.ITensor[0,0]
        return self.in_Ix

    @functools.cached_property
    def Ixy(self):
        return self.ITensor[0,1]

    @property
    def J(self):
        if self.in_J is None:
            self.in_J = self._section_properties.get_j()
        return self.in_J

    @property
    def A(self):
        if self.in_A is None:
            self.in_A = self._section_properties.get_area()
        return self.in_A      

    @functools.cached_property
    def Vol(self):
        return self.A * self.L

    @functools.cached_property
    def mass(self):
        return self.material.density * self.Vol

    @functools.cached_property
    def cost(self):
        return self.mass * self.material.cost_per_kg

    @functools.cached_property
    def centroid2d(self):
        return self._section_properties.get_c()

    @functools.cached_property
    def centroid3d(self):
        return (self.P2 -self.P1) / 2.0 + self.P1

    def section_results(self):
        return self._section_properties.display_results()

    def show_mesh(self):
        return self._section_properties.plot_mesh()


    def von_mises_stress_l(self,reverse_xy=False):
        out = {}
        for combo in self.structure.frame.LoadCombos:
            rows = []
            for i in numpy.linspace(0,1,11):
                inp  = dict(N=self.member.Axial(i,combo),
                            Vx=self.member.Shear('Fz' if not reverse_xy else 'Fy' ,i,combo),
                            Vy=self.member.Shear('Fy' if not reverse_xy else 'Fz' ,i,combo),
                            Mxx=self.member.Moment('Mz' if not reverse_xy else 'My',i,combo), 
                            Myy=self.member.Moment( 'My' if not reverse_xy else 'Mz',i,combo), 
                            M11=0, 
                            M22=0, 
                            Mzz=self.member.Torsion(i,combo) )
                
                sol = self._section_properties.calculate_stress(**inp)
                mat_stresses = sol.get_stress()

                max_vm =  numpy.nanmax([numpy.nanmax(stresses['sig_vm']) for stresses in mat_stresses])
                rows.append(max_vm )

            out[combo] = numpy.array(rows)
        
        return out

    def stress_info(self,reverse_xy=False):
        out = {}
        for combo in self.structure.frame.LoadCombos:
            rows = []
            for i in numpy.linspace(0,1,11):
                inp  = dict(N=self.member.Axial(i,combo),
                            Vx=self.member.Shear('Fz' if not reverse_xy else 'Fy' ,i,combo),
                            Vy=self.member.Shear('Fy' if not reverse_xy else 'Fz' ,i,combo),
                            Mxx=self.member.Moment('Mz' if not reverse_xy else 'My',i,combo), 
                            Myy=self.member.Moment( 'My' if not reverse_xy else 'Mz',i,combo), 
                            M11=0, 
                            M22=0, 
                            Mzz=self.member.Torsion(i,combo) )
                
                sol = self._section_properties.calculate_stress(**inp)
                mat_stresses = sol.get_stress()
                oout = {'x':i}
                for stresses in mat_stresses:
                    vals = {sn+'_'+stresses['Material']:numpy.nanmax(stress) for sn,stress in stresses.items() if 
                    isinstance(stress,numpy.ndarray)}
                    oout.update(vals)

                rows.append( oout )

            out[combo] = pandas.DataFrame(rows)
        
        return out



    def results(self):
        rows = []
        for combo in self.structure.frame.LoadCombos:
            mem = self.member
            row = dict( max_axial = mem.MaxAxial(combo),
                        min_axial = mem.MinAxial(combo),
                        max_my = mem.MaxMoment('My',combo),
                        min_my = mem.MinMoment('My',combo),
                        max_mz = mem.MaxMoment('Mz',combo),
                        min_mz = mem.MinMoment('Mz',combo),
                        max_shear_y = mem.MaxShear('Fy',combo),
                        min_shear_y = mem.MinShear('Fy',combo),
                        max_shear_z = mem.MaxShear('Fz',combo),
                        min_shear_z = mem.MinShear('Fz',combo),
                        max_torsion = mem.MaxTorsion(combo),
                        min_torsion = mem.MinTorsion(combo),
                        max_deflection_y = mem.MaxDeflection('dy',combo),
                        min_deflection_y = mem.MinDeflection('dy',combo),
                        max_deflection_x = mem.MaxDeflection('dx',combo),
                        min_deflection_x = mem.MinDeflection('dx',combo)                     
                        )

            rows.append(row)

        return pandas.DataFrame(rows)

if __name__ == '__main__':


    # tst = Structure(name = 'tetst')
    # tst.add_node('k',0,0,-50)
    # tst.add_node('o',0,0,0)
    # tst.add_node('t',0,0,50)

    # tst.add_member('btm','k','o',material = SS_316(), section = sections.Rhs(1,0.5,0.1,0.1,5))
    # bm = tst.add_member('top','o','t',material = Aluminum(), section = sections.Chs(1.0,0.06,60) )
    
    # tst.add_constraint('k',*(True,)*6)
    # tst.frame.AddNodeLoad('t','FX',10000)
    # tst.frame.AddNodeLoad('t','FY',1)
    # tst.frame.AddNodeLoad('t','FZ',1)

    # tst.analyze(check_statics=True)

    # print(bm.results())
    # #tst.visulize()

    #Result Checked Against
    #https://www.engineeringtoolbox.com/cantilever-beams-d_1848.html

    st = Structure(name = 'cantilever_beam')
    st.add_node('wall',0,0,0)
    st.add_node('free',0,5,0)

    ibeam = sections.ISection(0.3072,0.1243,0.0121,0.008,0.0089,4)
    #ibeam = sections.RectangularSection(1,0.2)
    bm = st.add_member('mem','wall','free',material = Aluminum(), section = ibeam )


    #bm = st.add_member_with('top','wall','free',E=200E9,G=75E9,Ix=8196,Iy=8196,J=8196,A=0.053)

    st.add_constraint('wall',con_DX=True, con_DY=True, con_DZ=True, con_RY=True,con_RX=True,con_RZ=True)
    st.frame.AddNodeLoad('free','FY',3000)

    # st.frame.DefineReleases('mem', False, False, False, False, True, True, \
    #                         False, False, False, False, True, True)

    st.analyze(check_statics=True)

    print(bm.results())
    #st.visulize()






















#This was an attempt to calculate generic section properties using shapely polygon grid masking
# @otterize
# class Profile(Configuration):
#     '''A profile takes an shapely polygon as a profile, possibly with an inner polygon to mask points and then determine section properties'''
#     profile = attr.ib(validator= attr.validators.instance_of( Polygon ) ) #required
#     Npts = attr.ib(default=1000)
    
#     grid_aspect_max = 2.0 #ratio of cells Dxi, will adjust dimensions with max Npts
#     _grid = None

#     @property
#     def box_extents(self):
#         return self.profile.bounds

#     @property
#     def dxdy(self):
#         '''returns dx,dy as constrained by max aspect ratio'''
#         minx, miny, maxx, maxy = self.box_extents
#         dx,dy = (maxx - minx)/Npts, (maxy- miny)/Npts
#         if (dx/dy) >= self.grid_aspect_max: #dx is limiting
#             dy = dx /self.grid_aspect_max
#             return dx,dy

#         if (dy/dx) >= self.grid_aspect_max: #dy is limiting
#             dx = dy / self.grid_aspect_max
#             return dx,dy

#         return dx,dy

#     @property
#     def grid(self):
#         if self._grid is None:
#             dx,dy = self.dxdy
#             minx, miny, maxx, maxy = self.box_extents
#             X = numpy.arange( minx , maxx+dx , dx )
#             Y = numpy.arange( miny , maxy+dy , dy )
#             self._grid = GX,GY = meshgrid(X,Y)
#         return self._grid


#     def shp_mask(self, shp, x, y, m=None):
#         """Use recursive sub-division of space and shapely contains method to create a raster mask on a regular grid.
#         :param shp : shapely's Polygon (or whatever with a "contains" method and intersects method)
#         :param m : mask to fill, optional (will be created otherwise)
#         """
#         rect = self.box_extents

#         if m is None:
#             m = np.zeros((y.size, x.size), dtype=bool)
                
#         if not shp.intersects(rect):
#             m[:] = False
        
#         elif shp.contains(rect):
#             m[:] = True
        
#         else:
#             k, l = m.shape
            
#             if k == 1 and l == 1:
#                 m[:] = shp.contains(Point(x[0], y[0]))
                
#             elif k == 1:
#                 m[:, :l//2] = shp_mask(shp, x[:l//2], y, m[:, :l//2])
#                 m[:, l//2:] = shp_mask(shp, x[l//2:], y, m[:, l//2:])
                
#             elif l == 1:
#                 m[:k//2] = shp_mask(shp, x, y[:k//2], m[:k//2])
#                 m[k//2:] = shp_mask(shp, x, y[k//2:], m[k//2:])
            
#             else:
#                 m[:k//2, :l//2] = shp_mask(shp, x[:l//2], y[:k//2], m[:k//2, :l//2])
#                 m[:k//2, l//2:] = shp_mask(shp, x[l//2:], y[:k//2], m[:k//2, l//2:])
#                 m[k//2:, :l//2] = shp_mask(shp, x[:l//2], y[k//2:], m[k//2:, :l//2])
#                 m[k//2:, l//2:] = shp_mask(shp, x[l//2:], y[k//2:], m[k//2:, l//2:])
            
#         return m