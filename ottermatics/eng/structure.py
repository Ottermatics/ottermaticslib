import attr
import numpy
import functools
import shapely
import pandas
from shapely.geometry import Polygon, Point

from ottermatics.tabulation import (
    TABLE_TYPES,
    NUMERIC_VALIDATOR,
    system_property,
)
from ottermatics.configuration import otterize, Configuration
from ottermatics.components import Component

# from ottermatics.analysis import Analysis
from ottermatics.properties import system_property
from ottermatics.system import System
from ottermatics.eng.solid_materials import *
from ottermatics.common import *

import sectionproperties
import sectionproperties.pre.geometry as geometry
import sectionproperties.pre.library.primitive_sections as sections
import PyNite as pynite
from PyNite import Visualization

from ottermatics.eng.structure_beams import Beam,rotation_matrix_from_vectors


SECTIONS = {
    k: v
    for k, v in filter(
        lambda kv: issubclass(kv[1], geometry.Geometry)
        if type(kv[1]) is type
        else False,
        sections.__dict__.items(),
    )
}


nonetype = type(None)


# TODO: Make analysis, where each load case is a row, but how sytactically?
@otterize
class Structure(System):
    """A integration between sectionproperties and PyNite, with a focus on ease of use

    Right now we just need an integration between Sections+Materials and Members, to find, CG, and inertial components

    Possible Future Additions:
    1) Structure Motion (free body ie vehicle , multi-strucutre ie robots)
    """

    frame: pynite.FEModel3D = None
    _beams: dict = None
    _materials: dict = None

    default_material: SolidMaterial = attrs.field(default=None)

    #Default Loading!
    add_gravity_force: bool = attrs.field(default=True)
    gravity_dir: str = attrs.field(default="FZ")
    gravity_mag: float = attrs.field(default=9.81)
    gravity_scalar: float = attrs.field(default=-1)
    gravity_cases:list = attrs.field(default=['gravity'])
    default_case:str = attrs.field(default='gravity')
    default_combo:str = attrs.field(default='only_gravity')

    #this orchestrates load retrieval
    current_case:str = attrs.field(default='gravity')
    current_combo:str = attrs.field(default='only_gravity')

    _meshes = None

    def __on_init__(self):
        self._materials = {}
        self._meshes = {}        
        self.initalize_structure()
        self.frame.add_load_combo('only_gravity',{'gravity':1.0},'gravity')


    def initalize_structure(self):
        self.frame = pynite.FEModel3D()
        self._beams = {}  # this is for us!
        self._materials = {}
        self.debug("created frame...")

        if self.default_material:
            material = self.default_material
            uid = material.unique_id
            if uid not in self._materials:
                self.debug(f'add material {uid}')
                self.frame.add_material(
                    uid, material.E, material.G, material.nu, material.rho
                )
                self._materials[uid] = material

    @property
    def nodes(self):
        return self.frame.Nodes

    @property
    def members(self):
        return self.frame.Members

    @property
    def beams(self):
        return self._beams
    
    def add_eng_material(self,material:'SolidMaterial'):

        if material.unique_id not in self._materials:
            self.add_material(material.unique_id,material.E,material.G,material.nu,material.rho)
            self._materials[material.unique_id] = material

        
    def add_mesh(self,meshtype,*args,**kwargs):
        """maps to appropriate PyNite mesh generator but also applies loads
        
        Currently these types are supported ['rect','annulus','cylinder','triangle']
        """
        mesh = None
        types = ['rect','annulus','cylinder','triangle']
        if meshtype not in types:
            raise KeyError(f'type {type} not in {types}')
        
        if meshtype == 'rect':
            mesh = self.add_rectangle_mesh(*args,**kwargs)
        elif meshtype == 'annulus':
            mesh = self.add_annulus_mesh(*args,**kwargs)
        elif meshtype == 'cylinder':
            mesh = self.add_cylinder_mesh(*args,**kwargs)
        elif meshtype == 'triangle':
            mesh = self.add_frustrum_mesh(*args,**kwargs)

        assert mesh is not None, f'no mesh made!'

        self._meshes[mesh] = self.frame.Meshes[mesh]

        self.merge_duplicate_nodes()

        if self.add_gravity_force:
            self.apply_gravity_to_mesh(mesh)

    def apply_gravity_to_mesh(self,meshname):
        self.info(f'applying gravity to {meshname}')
        mesh = self._meshes[meshname]

        mat = self._materials[mesh.material]
        rho = mat.rho

        for elid,elmt in mesh.elements.items():
            
            #make node vectors
            n_vecs = {}
            nodes = {}
            for nname in ['i_node','j_node','m_node','n_node']:
                ni = getattr(elmt,nname)
                nodes[nname] = ni
                n_vecs[nname] = numpy.array([ni.X,ni.Y,ni.Z])
            ni = n_vecs['i_node']
            nj = n_vecs['j_node']
            nm = n_vecs['m_node']
            nn = n_vecs['n_node']
            
            #find element area
            im = ni-nm
            jn = nj-nn

            A = numpy.linalg.norm(numpy.cross(im,jn))/2 
             
            #get gravity_forcemultiply area x thickness x density/4 
            mass = A * elmt.t * rho 
            fg = mass * self.gravity_mag * self.gravity_scalar
            fnode = fg / 4
            
            #apply forces to corner nodes / 4
            for case in self.gravity_cases:
                for nname,disnode in nodes.items():
                    self.add_node_load(disnode.name,self.gravity_dir,fnode,case=case)


    def add_member(self, name, node1, node2, section, material=None, **kwargs):
        assert node1 in self.nodes
        assert node2 in self.nodes

        if material is None:
            material = self.default_material

        uid = material.unique_id
        if uid not in self._materials:
            self.debug(f'add material {uid}')
            self.frame.add_material(
                uid, material.E, material.G, material.nu, material.rho
            )
            self._materials[uid] = material

        B = beam = Beam(
            structure=self, name=name, material=material, section=section
        )
        self._beams[name] = beam
        self.frame.add_member(
            name,
            i_node=node1,
            j_node=node2,
            material=uid,
            Iy=B.Iy,
            Iz=B.Ix,
            J=B.J,
            A=B.A,
            **kwargs,
        )
        
        if self.add_gravity_force:
            beam.apply_gravity_force(z_dir=self.gravity_dir,z_mag=self.gravity_scalar)

        return beam

    def add_member_with(
        self, name, node1, node2, E, G, Iy, Ix, J, A, material_rho=0,**kwargs
    ):
        """a way to add specific beam properties to calculate stress,
        This way will currently not caluclate resulatant beam load.
        #TOOD: Add in a mock section_properties for our own stress calcs
        """
        assert node1 in self.nodes
        assert node2 in self.nodes

        material = SolidMaterial(
            density=material_rho, elastic_modulus=E, shear_modulus=G
        )
        uid = material.unique_id
        if uid not in self._materials:
            self.debug(f'add material {uid}')
            self.frame.add_material(
                uid, material.E, material.G, material.nu, material.rho
            )
            self._materials[uid] = material

        B = beam = Beam(
            structure=self,
            name=name,
            material=material,
            in_Iy=Iy,
            in_Ix=Ix,
            in_J=J,
            in_A=A,
            section=None,
        )
        self._beams[name] = beam

        if self.add_gravity_force:
            beam.apply_gravity_force(z_dir=self.gravity_dir,z_mag=self.gravity_scalar)    

        self.frame.add_member(name, i_node=node1, j_node=node2, material=uid, Iy=Iy, Iz=Ix, J=J, A=A,**kwargs)

        return beam

    # def analyze(self, **kwargs):
    #     return self.frame.Analyze(**kwargs)

    #TODO: add mesh COG
    @property
    def cog(self):
        XM = numpy.sum(
            [bm.mass * bm.centroid3d for bm in self.beams.values()], axis=0
        )
        return XM / self.mass

    # TODO: add cost and mass of meshes
    @system_property
    def mass(self) -> float:
        """sum of all beams mass"""
        return sum([bm.mass for bm in self.beams.values()])

    @system_property
    def cost(self) -> float:
        """sum of all beams cost"""
        return sum([bm.cost for bm in self.beams.values()])

    def visulize(self, **kwargs):
        if 'combo_name' not in kwargs:
            kwargs['combo_name'] = 'only_gravity'
        Visualization.RenderModel(self.frame, **kwargs)

    # TODO: add mesh stress / deflection info min/max ect.
    @property
    def node_dataframes(self):
        out = {}
        for case in self.frame.LoadCombos:
            rows = []
            for node in self.nodes.values():
                row = {
                    "name": node.Name,
                    "dx": node.DX[case],
                    "dy": node.DY[case],
                    "dz": node.DZ[case],
                    "rx": node.RX[case],
                    "ry": node.RY[case],
                    "rz": node.RZ[case],
                    "rxfx": node.RxnFX[case],
                    "rxfy": node.RxnFY[case],
                    "rxfz": node.RxnFZ[case],
                    "rxmx": node.RxnMX[case],
                    "rxmy": node.RxnMY[case],
                    "rxmz": node.RxnMZ[case],
                }
                rows.append(row)

            out[case] = pandas.DataFrame(rows)
        return out

    @property
    def INERTIA(self):
        """Combines all the mass moments of inerita from the internal beams (and other future items!) into one for the sturcture with the parallel axis theorm"""
        cg = self.cog
        I = numpy.zeros((3, 3))

        for name, beam in self.beams.items():
            self.debug(f"adding {name} inertia...")
            I += beam.GLOBAL_INERTIA

        return I

    @property
    def Fg(self):
        """force of gravity"""
        return numpy.array([0, 0, -self.mass * g])

    def __getattr__(self, attr):
        return getattr(self.frame, attr)

    def __dir__(self):
        return sorted(set(dir(super(Structure, self))+dir(Structure)+ dir(self.frame)))
