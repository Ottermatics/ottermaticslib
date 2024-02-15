import attr
import numpy
import functools
import shapely
import pandas
from shapely.geometry import Polygon, Point

from engforge.tabulation import (
    TABLE_TYPES,
    NUMERIC_VALIDATOR,
    system_property,
)
from engforge.configuration import forge, Configuration
from engforge.components import Component

# from engforge.analysis import Analysis
from engforge.properties import system_property
from engforge.system import System
from engforge.component_collections import ComponentDict
from engforge.eng.solid_materials import *
from engforge.common import *
from engforge.logging import log, LoggingMixin
from engforge.eng.costs import CostModel,cost_property
from engforge.slots import SLOT
from engforge.eng.prediction import PredictionMixin

import sectionproperties
import sectionproperties.pre.geometry as geometry
import sectionproperties.pre.library.primitive_sections as sections
import PyNite as pynite
import numpy as np
from scipy import optimize as sciopt
import pandas as pd
import ray
import collections
import numpy as np
import itertools as itert
import asyncio
import weakref
from sklearn import svm

from engforge.typing import Options
from engforge.eng.structure_beams import Beam, rotation_matrix_from_vectors


class StructureLog(LoggingMixin):
    pass


log = StructureLog()

SECTIONS = {
    k: v
    for k, v in filter(
        lambda kv: issubclass(kv[1], geometry.Geometry)
        if type(kv[1]) is type
        else False,
        sections.__dict__.items(),
    )
}


class StopAnalysis(Exception):
    pass


nonetype = type(None)

norm = np.linalg.norm


def sort_max_estimated_failures(failures):
    """a generator to iterated through (beam,combo,x) as they return the highest fail_fraction for structures"""

    failures = {
        (beamnm, combo, x): dat
        for beamnm, estfail in failures.items()
        for (combo, x), dat in estfail.items()
    }

    for (beamnm, combo, x), dat in sorted(
        failures.items(), key=lambda kv: kv[1]["fail_frac"], reverse=True
    ):
        yield (beamnm, combo, x), dat


def check_failures(failures: dict) -> bool:
    """check if any failures exist in the nested dictonary and return True if there are"""

    if "mesh_failures" in failures and failures["mesh_failures"]:
        return True

    if "actual" in failures and failures["actual"]:
        for bm, cases in failures["actual"].items():
            for (combo, x), dat in cases.items():
                if "fails" in dat and dat["fails"]:
                    return True
    return False


def check_est_failures(failures: dict, top=True) -> bool:
    """check if any failures exist in the nested dictonary and return True if there are"""
    for bm, cases in failures.items():
        if isinstance(cases, dict):
            if top:
                log.info(f"checking est fail {bm}")
            for k, v in cases.items():
                if isinstance(v, dict):
                    val = check_est_failures(v, top=False)
                    if val:
                        return True
                elif v:
                    return True
        elif cases:
            return True
    return False


# Mesh Utils
def quad_stress_tensor(quad, r=0, s=0, combo="gravity"):
    """determines stresses from a plate or quad"""
    q = quad
    S_xx, S_yy, Txy = q.membrane(r, s, combo)
    Qx, Qy = q.shear(r, s, combo)

    S_xx, S_yy, Txy = float(S_xx), float(S_yy), float(Txy)

    T_zx = float(Qx) / q.t
    T_yz = float(Qy) / q.t

    return np.array([[S_xx, Txy, T_zx], [Txy, S_yy, T_yz], [T_zx, T_yz, 0.0]])


def node_pt(node):
    return np.array([node.X, node.Y, node.Z])


def node_info(quad, material):
    # make point arrays
    ind = node_pt(quad.i_node)
    jnd = node_pt(quad.j_node)
    nnd = node_pt(quad.n_node)
    mnd = node_pt(quad.m_node)

    # area of triangle = 0.5 (V1 x V2)
    C1 = np.cross(jnd - ind, nnd - ind)
    C2 = np.cross(jnd - mnd, nnd - mnd)

    # info!
    A = float((norm(C1) + norm(C2)) * 0.5)
    V = A * quad.t
    mass = V * material.density

    # output
    out = dict(
        A=A,
        cog=(ind + jnd + mnd + mnd) / 4,
        V=V,
        mass=mass,
        cost=mass * material.cost_per_kg,
        allowable_stress = material.allowable_stress,
    )
    return out


def calculate_von_mises_stress(stress_tensor):
    e_val, e_vec = np.linalg.eigh(stress_tensor)
    p3, p2, p1 = np.sort(e_val)
    return 0.707 * ((p1 - p2) ** 2 + (p2 - p3) ** 2 + (p1 - p3) ** 2) ** 0.5


def calculate_quad_vonmises(quad, combo) -> dict:
    """looks at 9 points over a quad to determine its von mises stresses
    :returns: a dictinary of"""
    qvm = {}
    for r, s in itert.product([-1, 0, 1], [-1, 0, 1]):
        S = quad_stress_tensor(q, r, s, combo)
        qvm[(r, s)] = calculate_von_mises_stress(S)
    return qvm

@forge
class BeamDict(ComponentDict):
    component_type:type = attrs.field(default=Beam)

# TODO: Make analysis, where each load case is a row, but how sytactically?
@forge
class Structure(System,CostModel,PredictionMixin):
    """A integration between sectionproperties and PyNite, with a focus on ease of use

    Right now we just need an integration between Sections+Materials and Members, to find, CG, and inertial components

    Possible Future Additions:
    1) Structure Motion (free body ie vehicle , multi-strucutre ie robots)
    """

    frame: pynite.FEModel3D = None
    # _beams: dict = None
    _materials: dict = None

    default_material: SolidMaterial = attrs.field(default=None)

    solve_method = Options('linear','normal','pdelta')

    # Default Loading!
    add_gravity_force: bool = attrs.field(default=True)
    gravity_dir: str = attrs.field(default="FZ")
    gravity_mag: float = attrs.field(default=9.81)
    gravity_scalar: float = attrs.field(default=-1)
    gravity_name: str = attrs.field(default='gravity')
    gravity_cases: list = attrs.field(default=["gravity"])
    default_case: str = attrs.field(default="gravity")
    default_combo: str = attrs.field(default="gravity")

    # this orchestrates load retrieval
    check_statics: bool = attrs.field(default=True)
    current_combo: str = attrs.field(default="gravity")
    #iteration combos iterates over LoadCombos by default
    iteration_combos: list = attrs.field(default=None)

    #merge_duplicate_nodes by default
    merge_nodes: bool = attrs.field(default=True)
    tolerance: float = attrs.field(default=1e-3)
    
    #beams
    beams = SLOT.define_iterator(BeamDict, wide=True)
    
    #per execute failure analysis
    #calculate actual failure will estimate then run failure analysis for bad cases
    #calculate full failure will run failure analysis without stoping at first failure
    calculate_actual_failure: bool =attrs.field(default=True)
    calculate_full_failure: bool =attrs.field(default=False)
    failure_solve_method = Options('bisect','root')
    failure_records: list = attrs.field(default=None)
    
    #prediciton calculation
    prediction: bool = attrs.field(default=False)
    max_records: int = attrs.field(default=10000)
    _prediction_parms: list = attrs.field(default=None)

    min_rec = 10
    near_margin=0.1
    max_margin=1.5
    max_guess = 1E8

    current_failure_summary = None
    _always_save_data = True  # we dont respond to inputs so use this
    _meshes = None
    _any_solved = False

    def __on_init__(self):
        self._materials = weakref.WeakValueDictionary()
        self._meshes = weakref.WeakValueDictionary()
        self.initalize_structure()
        self.frame.add_load_combo(self.default_combo, {self.gravity_name: 1.0}, "gravity" )
        self.create_structure()

        if self.prediction:
            d={'fails':{'mod':svm.SVC(C=1000,gamma=0.1,probability=True),'N':0},
            'fail_frac':{'mod':svm.SVR(C=1,gamma=0.1),'N':0},
            'beam_fail_frac':{'mod':svm.SVR(C=1,gamma=0.1),'N':0},'mesh_fail_frac':{'mod':svm.SVR(C=1,gamma=0.1),'N':0} }
            self._prediction_models = d

    def create_structure(self):
        '''
        Use this method to create a structure on init.

        # Example code for creating a structure
        frame = Structure()
        frame.add_material("default_material", 2E9, 8E8, 0.3, 7850)

        # Add nodes
        node1 = frame.add_node("Node1", 0, 0, 0)
        node2 = frame.add_node("Node2", 0, 0, 10)

        # Add elements
        element = frame.add_member("Element1", "Node1", "Node2", "default_material")

        # Add supports
        frame.def_support("Node1", True, True, True, True, True, True)
        frame.def_support("Node2", True, True, True, True, True, True)

        # Add loads
        self.add_node_load("Node2", "FZ", -100)

        # Solve the structure
        self.run() #calls execute!
        '''
        pass


    # Execution
    def execute(self, combos: list = None,save=True,record=True, *args, **kwargs):
        """wrapper allowing saving of data by load combo"""

        # the string input case, with csv support
        if isinstance(combos, str):
            combos = combos.split(",")  # will be a list!


        
        self.info(f"running with combos: {combos} | {args} | {kwargs} ")
        self._run_id = int(uuid.uuid4())
        iteration_combos = self.iteration_combos
        if iteration_combos is None:
            iteration_combos = list(self.LoadCombos.keys())
                
        for combo in iteration_combos:
            if combos is not None and combo not in combos:
                continue
            #reset cached failures
            self.current_failure_summary = None
            self._current_combo_failure_analysis = None            
            
            #run the analysis
            self.index += 1
            combo_possible = self.struct_pre_execute(combo) #can change combo
            if combo_possible:
                self.current_combo = combo_possible
            else:
                self.current_combo = combo
            combo = self.current_combo
            
            self.info(f"running load combo : {combo} with solver {self.solve_method}")

            if 'check_statics' not in kwargs:
                kwargs['check_statics'] = self.check_statics

            if self.solve_method == 'linear':
                self.analyze_linear(load_combos=[combo], *args, **kwargs)
            elif self.solve_method == 'normal':  
                self.analyze(load_combos=[combo], *args, **kwargs)
            elif self.solve_method == 'pdelta':
                self.analyze_PDelta(load_combos=[combo], *args, **kwargs)

            self._any_solved = True #Flag system properties to save
            self.struct_post_execute(combo)
            
            #backup data saver.
            self.current_failures() #cache failures
            self._anything_changed = True #change costs
            if save:
                self.save_data(index=self.index, force=True)

            #record results for prediction
            if record and self.prediction:
                res = self._prediction_record
                if res: self.record_result(res)

    #Solver Mechanics
    def struct_root_failure(self,x,fail_parm,base_kw,mult=1,target=1,sols=None,save=False,*args,**kw):
        assert target > 0, 'target must be positive'
        bkw = base_kw.copy()
        bkw[fail_parm] = x*mult
        self.setattrs(bkw)
        self.info(f'solving root iter: {bkw}')
        if args or kw:
            self.info(f'adding args to solver: {args} | {kw}')        
        self.execute(save=save,*args,**kw)
        wrong_side = (mult * x)<0
        ff = self.max_est_fail_frac
        bf = self.max_beam_est_fail_frac
        mf = self.max_mesh_est_fail_frac
        if wrong_side:
            #min = 1 with an gradual but exponential value from 0
            l10 = max(np.log10(abs(ff)+1),1)+1
            exp = min(abs(ff),l10)**0.5
            res = max(target*np.e**exp,target)
        else:
            res = (target-ff)

        if sols is not None:
            sols.append({'x':x,'ff':ff,'obj':res,'kw':bkw,'mf':mf,'bf':bf})

        self.info(f'ran: {bkw} -> {ff:5.4f} | {res:5.4f}')
        return res

    def determine_failure_load(self,fail_parm,base_kw,mult=1,target_failure_frac=1.0,guess=None,tol=5E-3,return_sol = False,max_tries=5,*args,**kw):
        
        #TODO: add reversion logic
        if not hasattr(self,'_max_parm_dict'):
            self._max_parm_dict = {}
        tries = 0
        while tries < max_tries:
            try:
                if guess is None:
                    guess = mult*1E6
                    if fail_parm not in self._max_parm_dict:
                        self._max_parm_dict[fail_parm] = self.max_guess
                        maxx = self.max_guess * mult
                    else:
                        self.max_guess = self._max_parm_dict[fail_parm]
                        maxx = self.max_guess * mult

                solutions = []

                #allow passing of args to solver
                if args or kw:
                    self.info(f'adding args to solver: {args} | {kw}')
                func = lambda x,*argsolv: self.struct_root_failure(x,*argsolv,*args,**kw)    

                if self.failure_solve_method == 'bisect':
                    ans = sciopt.root_scalar(func, x0 = 1000, x1 = guess, rtol=tol, xtol = tol,args=(fail_parm,base_kw,mult,target_failure_frac,solutions),bracket=(0,maxx) )
                    self.info(f'{mult}x{fail_parm:<6}| success: {ans.converged} , ans:{ans.root}, base: {base_kw}')

                else: #root
                    ans = sciopt.root_scalar( func, x0 = 100, x1 = guess, rtol=tol, xtol = tol,args=(fail_parm,base_kw,mult,target_failure_frac,solutions) )
                    self.info(f'{mult}x{fail_parm:<6}| success: {ans.converged} , ans:{ans.root}, base: {base_kw}')                        

                if return_sol:
                    return solutions

                if ans.converged:
                    self.setattrs({fail_parm:ans.root})
                    self.execute(save=True,*args,**{k:v for k,v in kw.items() if k != 'save'})
                    return ans.root

                elif solutions:
                    x = np.array([s['x'] for s in solutions])
                    f = np.array([s['ff'] for s in solutions])
                    b = np.array([s['bf'] for s in solutions])
                    m = np.array([s['mf'] for s in solutions])
                    return #TODO: fit the best solution using regression
                return np.nan

            except ValueError as e:
                #increase margin
                self.info(f'increasing {fail_parm} max guess by 100x')
                new_max = self._max_parm_dict[fail_parm]*100
                self._max_parm_dict[fail_parm] = new_max
                tries += 1


            except Exception as e:
                self.error(f'unknown error: {e}')
                raise e

    def _X_prediction_dict(self,base_obj=None) -> dict:
        if not self.prediction or not self._prediction_parms:
            return {}
        if base_obj is None:
            base_obj = self
        d = {k:getattr(base_obj,k,None) for k in self._prediction_parms}
        return d
    
    @property
    def _prediction_record(self)->dict:
        """returns the analysis result for prediction"""
        d = self._X_prediction_dict(self)
        d['fail_frac'] = ff= self.max_est_fail_frac
        d['beam_fail_frac'] = self.max_beam_est_fail_frac
        d['mesh_fail_frac'] = self.max_mesh_est_fail_frac
        d['fails'] = int(ff >= 1-1E-6)
        return d

    def record_result(self,stress_dict):
        """determines if stress record should be added to prediction_records"""
        if not self.prediction:
            return

        if len(self.prediction_records) > self.max_records:
            return

        ff = stress_dict['fail_frac']

        #Add the data to the stress records
        if self.near_margin and abs(ff-1) < self.near_margin:
            self.add_prediction_record(stress_dict)
        elif self.max_margin and ff < self.max_margin*2:
            self.add_prediction_record(stress_dict,False)
        elif not self.near_margin and not self.max_margin:
            self.add_prediction_record(stress_dict,False)
            
     

    def save_data(self,*args,**kw):
        if self._any_solved:
            super().save_data(*args,**kw)
            #self._any_solved = False
        else:
            self.info(f'nothing to save, run() structure first')

    def struct_pre_execute(self, combo=None):
        """yours to override to prep solver or dataframe
        :returns: combo to run, or None if want to continue with current iteration combos
        """
        pass

    def struct_post_execute(self, combo=None):
        """yours to override dataframe"""
        pass

    def initalize_structure(self):
        self.frame = pynite.FEModel3D()
        # self.beams = {}  # this is for us!
        self._materials = {}
        self.debug("created frame...")

        if self.default_material:
            material = self.default_material
            uid = material.unique_id
            if uid not in self._materials:
                self.debug(f"add material {uid}")
                self.frame.add_material(
                    uid, material.E, material.G, material.nu, material.rho
                )
                self._materials[uid] = material

    # Merged Feature Calls
    def add_eng_material(self, material: "SolidMaterial")->str:
        """adds the material to the structure and returns the pynite id"""
        if material.unique_id not in self._materials:
            self.add_material(
                material.unique_id,
                material.E,
                material.G,
                material.nu,
                material.rho,
            )
            self._materials[material.unique_id] = material

        return material.unique_id

    def mesh_extents(self) -> dict:
        mesh_extents = {}
        for mn, m in self.Meshes.items():
            minX, maxX = None, None
            minY, maxY = None, None
            minZ, maxZ = None, None
            for e in m.elements.values():
                for nn in ["i", "j", "n", "m"]:
                    node = getattr(e, f"{nn}_node")
                    X, Y, Z = node.X, node.Y, node.Z
                    if minX is None or X < minX:
                        minX = X
                    if maxX is None or X > maxX:
                        maxX = X
                    if minY is None or Y < minY:
                        minY = Y
                    if maxY is None or Y > maxY:
                        maxY = Y
                    if minZ is None or Z < minZ:
                        minZ = Z
                    if maxZ is None or Z > maxZ:
                        maxZ = Z
            mesh_extents[mn] = dict(
                minX=minX, maxX=maxX, minY=minY, maxY=maxY, minZ=minZ, maxZ=maxZ
            )
        return mesh_extents

    @property
    def quad_info(self) -> dict:
        """return a dictonary of quads with their mass properties like cog, mass, area and volume"""
        if hasattr(self, "_quad_info"):
            return self._quad_info

        self._quad_info = {}
        for mn, mesh in self.Meshes.items():
            for qn, q in mesh.elements.items():
                mat = self._materials[mesh.material]
                self._quad_info[qn] = node_info(q, mat)

        return self._quad_info.copy()

    def add_mesh(self, meshtype, *args, **kwargs):
        """maps to appropriate PyNite mesh generator but also applies loads

        Currently these types are supported ['rect','annulus','cylinder','triangle']
        """
        mesh = None
        types = ["rect", "annulus", "cylinder", "triangle"]
        if meshtype not in types:
            raise KeyError(f"type {type} not in {types}")

        if meshtype == "rect":
            mesh = self.add_rectangle_mesh(*args, **kwargs)
        elif meshtype == "annulus":
            mesh = self.add_annulus_mesh(*args, **kwargs)
        elif meshtype == "cylinder":
            mesh = self.add_cylinder_mesh(*args, **kwargs)
        elif meshtype == "triangle":
            mesh = self.add_frustrum_mesh(*args, **kwargs)
        elif meshtype == "cylinderset":
            mesh = self.add_mesh_cylinders(*args, **kwargs)

        assert mesh is not None, f"no mesh made!"

        self._add_mesh(mesh, self.frame.Meshes[mesh])

    def _add_mesh(self, meshname, mesh):
        self._meshes[meshname] = mesh

        # Ensure custom mesh is in frame meshes
        if meshname not in self.frame.Meshes:
            self.frame.Meshes[meshname] = mesh

        if self.merge_nodes:
            self.merge_duplicate_nodes(tolerance=self.tolerance)
            
        if self.add_gravity_force:
            self.apply_gravity_to_mesh(meshname)

    def apply_gravity_to_mesh(self, meshname):
        self.debug(f"applying gravity to {meshname}")
        mesh = self._meshes[meshname]

        mat = self._materials[mesh.material]
        rho = mat.rho

        for elid, elmt in mesh.elements.items():
            # make node vectors
            n_vecs = {}
            nodes = {}
            for nname in ["i_node", "j_node", "m_node", "n_node"]:
                ni = getattr(elmt, nname)
                nodes[nname] = ni
                n_vecs[nname] = numpy.array([ni.X, ni.Y, ni.Z])

            ni = n_vecs["i_node"]
            nj = n_vecs["j_node"]
            nm = n_vecs["m_node"]
            nn = n_vecs["n_node"]

            # find element area
            im = ni - nm
            jn = nj - nn

            A = numpy.linalg.norm(numpy.cross(im, jn)) / 2

            # get gravity_forcemultiply area x thickness x density/4
            mass = A * elmt.t * rho
            fg = mass * self.gravity_mag * self.gravity_scalar
            fnode = fg / 4

            # apply forces to corner nodes / 4
            for case in self.gravity_cases:
                for nname, disnode in nodes.items():
                    self.add_node_load(
                        disnode.name, self.gravity_dir, fnode, case=self.gravity_name
                    )

    def add_member(self, name, node1, node2, section, material=None, **kwargs):
        assert node1 in self.nodes
        assert node2 in self.nodes

        if material is None and self.default_material:
            material = self.default_material
        elif hasattr(section, "material"):
            material = section.material
        else:
            raise ValueError("material not defined as input or from default sources!")


        uid = material.unique_id
        if uid not in self._materials:
            self.debug(f"add material {uid}")
            self.frame.add_material(
                uid, material.E, material.G, material.nu, material.rho
            )
            self._materials[uid] = material

        beam_attrs = {
            k: v for k, v in kwargs.items() if k in Beam.input_attrs()
        }
        kwargs = {k: v for k, v in kwargs.items() if k not in beam_attrs}

        B = beam = Beam(
            structure=self,
            name=name,
            material=material,
            section=section,
            **beam_attrs,
        )
        self.beams[name] = beam
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
            beam.apply_gravity_force(
                z_dir=self.gravity_dir, z_mag=self.gravity_scalar
            )

        return beam

    def add_member_with(
        self, name, node1, node2, E, G, Iy, Ix, J, A, material_rho=0, **kwargs
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
            self.debug(f"add material {uid}")
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
        self.beams[name] = beam

        if self.add_gravity_force:
            beam.apply_gravity_force(
                z_dir=self.gravity_dir, z_mag=self.gravity_scalar
            )

        self.frame.add_member(
            name,
            i_node=node1,
            j_node=node2,
            material=uid,
            Iy=Iy,
            Iz=Ix,
            J=J,
            A=A,
            **kwargs,
        )

        return beam

    # OUTPUT
    def beam_dataframe(self, univ_parms: list = None,add_columns:list=None):
        """creates a dataframe entry for each beam and combo
        :param univ_parms: keys represent the dataframe parm, and the values represent the lookup value
        """

        df = self.dataframe
        beam_col = set([c for c in df.columns if c.startswith("beams.") and len(c.split(".")) > 2]) 
        beams = set([c.split(".")[1] for c in beam_col])
        parms = set([".".join(c.split(".")[2:]) for c in beam_col])

        # defaults
        if univ_parms is None:
            univ_parms_ = df.columns[df.columns.str.startswith("beams.")].tolist()
            univ_parms = [(c.split(".")[-1],c) for c in univ_parms_]
            uniq_parms = set([c.split(".")[-1] for c in univ_parms_])

        if add_columns is None:
            add_columns = []

        beam_data = []
        for i in range(len(df)):
            row = df.iloc[i]
            add_dat = {k: row[k] for k in add_columns}
            for beam in beams:
                bc = add_dat.copy()  # this is the data entry
                bc["name"] = bc
                for parm in parms:

                    if parm not in uniq_parms:
                        continue
                    #if parm not in row:
                        #continue
                    k = f"beams.{beam}.{parm}"
                    if k in row:
                        v = row[k]
                    #if 'Z1' in k:
                        #print(v)
                        bc[parm] = v

                beam_data.append(bc)  # uno mas

        dfb = pd.DataFrame(beam_data)
        return dfb

    @property
    def cog(self):
        XM_beam = numpy.sum(
            [bm.mass * bm.centroid3d for bm in self.beams.values()], axis=0
        )
        XM_quad = numpy.sum(
            [q["mass"] * q["cog"] for q in self.quad_info.values()], axis=0
        )
        XM = XM_beam + XM_quad
        return XM / self.mass

    @system_property
    def mass(self) -> float:
        """sum of all beams and quad masses"""
        return sum([sum(d.values()) for d in self.masses().values()])

    def masses(self) -> dict:
        """return a dictionary of beam & quad mass"""
        out = {}
        out["beams"] = beams = {}
        out["quads"] = quads = {}
        for beamname, bm in self.beams.items():
            beams[beamname] = bm.mass
        for quadname, q in self.quad_info.items():
            quads[quadname] = q["mass"]
        return out

    @property
    def structure_cost_beams(self) -> float:
        """sum of all beams and quad cost"""
        return sum([sum(self.costs['beams'].values())])

    @property
    def structure_cost_panels(self) -> float:
        """sum of all panels cost"""
        return sum([sum(self.costs['quads'].values())]) 

    @cost_property(category='mfg,material,panels')
    def structure_cost(self):
        return self.structure_cost_beams + self.structure_cost_panels       

    @solver_cached
    def costs(self) -> dict:
        """return a dictionary of beam & quad costs"""
        out = {}
        out["beams"] = beams = {}
        out["quads"] = quads = {}
        for beamname, bm in self.beams.items():
            beams[beamname] = bm.cost
        for quadname, q in self.quad_info.items():
            quads[quadname] = q["cost"]
        return out

    def visulize(self, **kwargs):
        from PyNite import Visualization

        if "combo_name" not in kwargs:
            kwargs["combo_name"] = self.default_combo
        Visualization.render_model(self.frame, **kwargs)

    # TODO: add mesh stress / deflection info min/max ect.
    @property
    def node_dataframe(self):
        #out = {}
        rows = []
        for case in self.frame.LoadCombos:
            for node in self.nodes.values():
                if case not in node.DX:
                    continue
                row = {
                    "case": case,
                    "name": node.name,
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

            #out[case] = pandas.DataFrame(rows)
        df = pandas.DataFrame(rows)
        df.set_index(['case','name'],inplace=True)            
        return df

    @property
    def INERTIA(self):
        """Combines all the mass moments of inerita from the internal beams (and other future items!) into one for the sturcture with the parallel axis theorm"""
        cg = self.cog
        I = numpy.zeros((3, 3))

        for name, beam in self.beams.items():
            self.debug(f"adding {name} inertia...")
            I += beam.GLOBAL_INERTIA

        for quad in self.quad_info.items():
            qcog = quad["cog"] - cg
            qmass = quad["mass"]

            qI = qmass * np.eye(3) * qcog * qcog.T
            I += qI

        return I

    @property
    def Fg(self):
        """force of gravity"""
        return numpy.array([0, 0, -self.mass * g])

    def __getattr__(self, attr):
        # log.info(f'get {attr}')
        if attr in dir(self.frame):
            return getattr(self.frame, attr)
        # if attr in dir(self):
        # return getattr(self,attr)
        # raise AttributeError(f'{attr} not found!')
        return self.__getattribute__(attr)

    def __dir__(self):
        return sorted(
            set(
                dir(super(Structure, self))
                + dir(Structure)
                + dir(self.frame)
                + dir(self.__class__)
            )
        )

    @property
    def nodes(self):
        return self.frame.Nodes

    @property
    def members(self):
        return self.frame.Members

    # Remote Utility Calls (pynite frame integration)
    def merge_displacement_results(self, combo, D):
        """adds a remotely calculated displacement vector and applies to nodes ala pynite convention"""
        self.frame._D[combo] = D

        for node in self.frame.Nodes.values():
            node.DX[combo] = D[node.ID * 6 + 0, 0]
            node.DY[combo] = D[node.ID * 6 + 1, 0]
            node.DZ[combo] = D[node.ID * 6 + 2, 0]
            node.RX[combo] = D[node.ID * 6 + 3, 0]
            node.RY[combo] = D[node.ID * 6 + 4, 0]
            node.RZ[combo] = D[node.ID * 6 + 5, 0]

    def prep_remote_analysis(self):
        """copies PyNite analysis pre-functions"""
        # Generate all meshes
        for mesh in self.frame.Meshes.values():
            if mesh.is_generated == False:
                mesh.generate()

        # Activate all springs and members for all load combinations
        for spring in self.frame.Springs.values():
            for combo_name in self.LoadCombos.keys():
                spring.active[combo_name] = True

        # Activate all physical members for all load combinations
        for phys_member in self.frame.Members.values():
            for combo_name in self.frame.LoadCombos.keys():
                phys_member.active[combo_name] = True

        # Assign an internal ID to all nodes and elements in the model
        self._renumber()

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = self._aux_list()

    # Failure Analysis Interface (Override your for your cases!)
    def failure_load_combos(
        self, report_info: dict, run_full=False, remote_sync=None, **kw
    ):
        """returns or yields a set of load combo names to check failure for.
        You're encouraged to override this for your specific cases
        :param report_info: the report that is generated in `run_failure_analysis` which will dynamically change with
        :yields: the combo name
        :returns: None when done
        """
        failures = report_info["failures"]

        # gravity case first
        o = [self.default_combo]
        yield o[0]
        if report_info["fails"]:
            if not run_full:
                return  # you are weak
        
        iteration_combos = self.iteration_combos
        if iteration_combos is None:
            iteration_combos = list(self.LoadCombos.keys())        

        for combo_name, combo in self.LoadCombos.items():
            if combo_name not in iteration_combos:
                continue
                
            if combo_name != self.default_combo:
                yield combo_name

                # allow time for combo to run
                if remote_sync:
                    d = remote_sync.ensure.remote(combo_name)
                    ray.wait([d])  # ,timeout=remote_timeout)

                cfail = failures[combo_name]
                if check_failures(cfail):
                    if not run_full:
                        return  # you are no good

    def failure_load_exithandler(self, combo, res, report_info, run_full):
        """a callback that is passed the structural combo, the case results

        Override this function to provide custom exit capability
        :param combo: the load combo analyzed
        :param res: the combo failure results
        :param report_info: the report dictionary
        :run_full: option to run everything
        """
        if check_failures(res):
            self.warning(f"combo {combo} failed!")
            report_info["fails"] = True
            if not run_full:
                raise StopAnalysis()

    def run_failure_analysis(self, SF=1.0, run_full=False, **kwargs):
        """
        Failure Determination:
        Beam Stress Estiates are used to determine if a 2D FEA beam/combo analysis should be run. The maximum beam vonmises stress is compared the the beam material allowable stress / saftey factor.

        Override:
        Depending on number of load cases, this analysis could take a while. Its encouraged to use a similar pattern of check and exit to reduce load combo size in an effort to fail early. Override this function if nessicary.

        Case / Beam Ordering:
        runs the gravity case first and checks for failure to exit early,
        then all other combos are run
        :returns: a dictionary with 2D fea failures and estimated failures as well for each beam
        """

        report = {"fails": False}  # innocent till proven guilty
        report["failures"] = failures = {}
        try:
            # Run the rest of the load cases
            for combo in self.failure_load_combos(report, **kwargs):
                # failure_load_combos returns none on exit
                if combo is None:
                    return report

                self.run(combo=combo)
                fail_res = self.run_combo_failure_analysis(combo, run_full, SF)
                failures[combo] = fail_res
                # will call
                self.failure_load_exithandler(combo, fail_res, report, run_full)

        except KeyboardInterrupt:
            return report

        except StopAnalysis:
            return report

        except Exception as e:
            self.error(e, "issue in failur analysis")

        return report  # tada!

    # Custom Failure Analysis Tools
    def check_failures(self, *args, **kwargs) -> bool:
        """exposes check_failures"""
        return check_failures(*args, **kwargs)

    def check_est_failures(self, *args, **kwargs) -> bool:
        """exposes check_est_failures"""
        return check_est_failures(*args, **kwargs)

    @property
    def full_failures(self):
        """return all failure estimates for all combos"""
        return self.estimated_failures_report()

    def check_mesh_failures(self, combo, run_full=False, SF=1.0,return_complete=False)->dict:
        """Returns the von-mises stress / allowable (failure fraction) stresses for each quad in each mesh.
        :param run_full: if false will exit early on first failure
        :returns: a dictinary of quad names and their failure fraction
        """
        self.info(f"checking mesh failure {combo}")
        
        stable = {}
        failures = {}
        summary = {'failures':failures,'stable':stable}
        
        quad_info = self.quad_info
        for meshname, mesh in self._meshes.items():
            matid = mesh.material
            mat = self._materials[matid]
            allowable = mat.allowable_stress
            for quadname, quad in mesh.elements.items():
                for r, s in itert.product([-1, 0, 1], [-1, 0, 1]):
                    S = quad_stress_tensor(quad, r, s, combo)
                    vm = calculate_von_mises_stress(S)
                    fail_frac = vm / allowable
                    if vm * SF > allowable:
                        if quadname not in failures:
                            failures[quadname] = fail_frac
                        elif fail_frac > failures[quadname]:
                            failures[quadname] = fail_frac

                        if not run_full:
                            if return_complete:
                                return summary                            
                            return failures
                    else:
                        if quadname not in stable:
                            stable[quadname] = fail_frac
                        elif fail_frac > stable[quadname]:
                            stable[quadname] = fail_frac

        if return_complete:
            return summary
        return failures

    def mesh_stress_dataframe(self,combo=None):
        """returns a dataframe of each quad, its x,y,z coords, vonmises stress and other mass/cost properties in a dataframe"""
        if combo is None:
            combo = self.current_combo
        max_vm = {}
        for qn, q in self.Quads.items():
            qvm = {}
            for r, s in itert.product([-1, 0, 1], [-1, 0, 1]):
                S = quad_stress_tensor(q, r, s, combo)
                qvm[(r, s)] = calculate_von_mises_stress(S)
            max_vm[qn] = max(qvm.values())

        mesh_data = []

        for qn, maxvm in max_vm.items():
            qinfo = self.quad_info[qn].copy()
            x, y, z = qinfo.pop("cog").tolist()
            mesh_data.append({"x": x, "y": y, "z": z, "vm": maxvm, **qinfo})

        return pd.DataFrame(mesh_data)

    def estimated_failures_report(
        self,
        concentrationFactor=1,
        methodFactor=1,
        fail_frac_criteria=None,
        combos: list = None,
    ) -> dict:
        """uses the beam estimated stresses with specific adders to account for beam 2D stress concentrations and general inaccuracy as well as a fail_fraction to analyze. The estimated loads are very pessimistic with a 250x overestimate"""

        # the string input case, with csv support
        if isinstance(combos, str):
            combos = combos.split(",")  # will be a list!

        # what fraction of allowable stress is generated for this case
        if fail_frac_criteria is None:
            fail_1 = None
        else:
            fail_1 = fail_frac_criteria / (concentrationFactor * methodFactor)

        failures = collections.defaultdict(dict)

        #df = self.dataframe
        run_combos = df.current_combo.to_list()

        failures_count = 0
        cases = 0

        iteration_combos = self.iteration_combos
        if iteration_combos is None:
            iteration_combos = list(self.LoadCombos.keys())  
        
        # gather forces for each beam
        for nm, beam in self.beams.items():
            self.debug(f"estimate beam stress: {beam.name}")

            # try each load combo
            for combo in iteration_combos:
                if combo not in run_combos:
                    continue  # theres, no point

                if combos and combo not in combos:
                    continue  # you are not the chose one(s)
                
                #MUST Loop over beams pos and combos to get the correct data for general case of all combos
                # combo_df = df[df.current_combo == combo]
                # combo_dict = combo_df.to_dict("records")
                # combo_dict = combo_dict[-1]  # last one is the combo in the case of repeats
                # loop positions start,mid,end
                for x in [0, 0.5, 1.0]:
                    cases += 1
                    f = beam.get_forces_at(x, combo=combo)
                    beam_fail_frac = beam.estimate_stress(**f,force_calc=self.calculate_actual_failure)
                    st = beam_fail_frac * beam.material.allowable_stress

                    fails = False
                    if fail_1 is None:
                        deltadf = abs(1-beam_fail_frac)
                        fails = deltadf <= beam.section.fail_frac_criteria()
                    elif beam_fail_frac >= fail_1:
                        fails = True

                    # record!
                    if fails:
                        self.debug(
                            f"estimated beam {beam.name} failure at L={x*100:3.2f}% | {combo}"
                        )
                        failures_count += 1

                        failures[nm][(combo, x)] = _d = {
                            "beam": nm,
                            "est_stress": st,
                            "combo": combo,
                            #"x": x,
                            "allowable_stress": beam.material.allowable_stress,
                            "fail_frac": beam_fail_frac,
                            "loads": f,
                        }

                        # failures[nm][(combo, x)].update(
                        #     **{
                        #         k.split(nm + ".")[-1] if nm in k else k: v
                        #         for k, v in combo_dict.items()
                        #         if ("beams" not in k or nm in k) and k not in _d
                        #     }
                        # )

        if failures_count:
            self.warning(
                f"{combos}| {failures_count} @ {failures_count*100/cases:3.2f}% estimated failures: {set(failures.keys())}"
            )

        return failures
            
    def current_failures(
        self,
        concentrationFactor=1,
        methodFactor=1,
        fail_frac_criteria=0.999,
    ) -> dict:
        """uses the beam estimated stresses with specific adders to account for beam 2D stress concentrations and general inaccuracy as well as a fail_fraction to analyze. The estimated loads are very pessimistic with a 250x overestimate by default for beams. mesh stresses are more accurate and failure is determined by fail_fac (von-mises stress / allowable stress)/fail_frac_criteria
        
        """

        # what fraction of allowable stress is generated for this case
        # what fraction of allowable stress is generated for this case

        fail_1 = fail_frac_criteria / (concentrationFactor * methodFactor)

        #FIXME: only run current combo (remove combo storage)
        failures = collections.defaultdict(dict)
        stable = collections.defaultdict(dict)
        summary = {'beam_failures':failures,'beams_stable':stable}

        failures_count = 0
        checks = 0
        beams = len(self.beams)

        combo = self.current_combo
        self.info(f'determining failures for {self.current_combo}')

        #self._lock_est_failures = True
        #data_dict = self.data_dict
        self._lock_est_failures = False

            
        # gather forces for each beam
        for nm, beam in self.beams.items():
            self.debug(f"estimate beam stress: {beam.name}")

            checks += 1
            beam_fail_frac = beam.fail_factor_estimate

            fails = False
            if beam_fail_frac >= fail_1:              
                fails = True

            if fails:
                self.debug(
                    f"estimated beam {beam.name} failure"
                )
                failures_count += 1

                failures[nm][combo] = _d = {
                    "beam": nm,
                    #"est_stress": st,
                    "combo": combo,
                    #"x": x,
                    #"allowable_stress": beam.material.allowable_stress,
                    "fail_frac": beam_fail_frac,
                    "fail_critera": fail_frac_criteria,
                    #"loads": f,
                }

                # failures[nm][combo].update(
                #     **{
                #         k.split(nm + ".")[-1] if nm in k else k: v
                #         for k, v in data_dict.items()
                #         if ("beams" not in k or nm in k) and k not in _d
                #     }
                # )
            else:
                stable[nm][combo] = _d = {
                    "beam": nm,
                    #"est_stress": st,
                    "combo": combo,
                    #"x": x,
                    #"allowable_stress": beam.material.allowable_stress,
                    "fail_frac": beam_fail_frac,
                    "fail_critera": fail_frac_criteria,
                    #"loads": f,
                }
        
        #add mesh failures
        mesh_summary = self.check_mesh_failures(combo, run_full=self.calculate_full_failure, SF=1.0, return_complete=True)
        mesh_failures = mesh_summary['failures']
        mesh_success = mesh_summary['stable']
        summary['mesh_failures'] = mesh_failures
        summary['mesh_stable'] = mesh_success
        
        #update failures
        mesh_failures_count = len([v for k,v in mesh_failures.items() if v > 0.99])

        max_fail_frac = max([0]+[v['fail_frac'] for beam,fails in failures.items() for k,v in fails.items()])
        max_stable_frac = max([0]+[v['fail_frac'] for beam,stables in stable.items() for k,v in stables.items()])
        max_mesh_fail_frac = max([0]+[v for k,v in mesh_failures.items()])
        max_mesh_stable_frac = max([0]+[v for k,v in mesh_success.items()])
        #Max mesh failure fracs
        max_beam_fail_frac = max(max_fail_frac,max_stable_frac)
        max_mesh_frac = max(max_mesh_fail_frac,max_mesh_stable_frac)
        max_fail_frac = max(max_beam_fail_frac,max_mesh_frac)  

        if failures_count:
            self.warning(
                f"{combo}| {failures_count} @ {failures_count*100/checks:3.2f}% estimated failures: {set(failures.keys())}"
            )
        summary['mesh_fail_count'] = mesh_failures_count
        summary['beam_fail_count'] = failures_count
        summary['failures_count'] = failures_count + mesh_failures_count
        summary['beams'] = beams
        summary['meshs'] = len(mesh_failures)
        summary['checks'] = checks + len(self.quad_info)
        summary['max_mesh_fail_frac'] = max_mesh_frac
        summary['max_beam_fail_frac'] = max_beam_fail_frac
        summary['max_fail_frac'] = max_fail_frac
    
        self.current_failure_summary = summary
        
        return summary
    
    #Failure Analysis Properties
    @system_property
    def max_est_fail_frac(self)->float:
        """estimated failure fraction for the current result"""
        fc = self._current_failures
        if isinstance(fc,dict):
            return fc['max_fail_frac']
        self.warning(f'could not get estimated beam failure frac')
        return fc
    
    @system_property
    def max_beam_est_fail_frac(self)->float:
        fc = self._current_failures
        if isinstance(fc,dict):
            return fc['max_beam_fail_frac']
        self.warning(f'could not get estimated beam failure frac')
        return fc
    
    @system_property
    def max_mesh_est_fail_frac(self)->float:
        fc = self._current_failures
        if isinstance(fc,dict):
            return fc['max_mesh_fail_frac']
        self.warning(f'could not get estimated failure frac')
        return fc
        
    @system_property
    def estimated_failure_count(self)->float:
        fc = self._current_failures
        if isinstance(fc,dict):
            return fc['failures_count']
        self.warning('could not get estimated failure count')
        return fc
    
    @system_property
    def estimated_beam_failure_count(self)->float:
        fc = self._current_failures
        if isinstance(fc,dict):
            return fc['beam_fail_count']    
        self.warning(f'could not get estimated beam failure count')
        return fc
            
    @system_property
    def estimated_mesh_failure_count(self)->float:
        fc = self._current_failures
        if isinstance(fc,dict):
            return fc['mesh_fail_count']
        self.warning(f'could not get estimated mesh failure count')
        return fc 
    
    @system_property
    def actual_failure_count(self)->float:
        if not self.calculate_actual_failure:
            return None

        afc = self._actual_failures
        if isinstance(afc,dict):
            return afc['failure_count']
        self.warning(f'could not get actual failure count')            
        return afc

    @system_property
    def actual_beam_failure_count(self)->float:
        if not self.calculate_actual_failure:
            return None

        afc = self._actual_failures
        if isinstance(afc,dict):
            return afc['beam_failures_count']
        self.warning(f'could not get actual beam failure count')
        return afc

    @system_property
    def actual_mesh_failure_count(self)->float:
        if not self.calculate_actual_failure:
            return None

        afc = self._actual_failures
        if isinstance(afc,dict):
            return afc['mesh_failure_count']
        self.warning(f'could not get actual mesh failure count')
        return afc

    @property
    def _current_failures(self):
        if hasattr(self,'_lock_est_failures') and self._lock_est_failures:
            self.info(f'current failures locked')
            return np.nan
        if self.current_failure_summary is None:
            fc = self.current_failures()
        else:
            fc = self.current_failure_summary
        return fc
    
    @property
    def _actual_failures(self):
        if not self.calculate_actual_failure:
            return None

        if hasattr(self,'_lock_est_failures') and self._lock_est_failures:
            return np.nan
        
        if self._current_combo_failure_analysis is None:
            if self.current_failure_summary is None:
                fc = self.current_failures()
            else:
                fc = self.current_failure_summary
            afc = self.run_combo_failure_analysis(combo=self.current_combo,run_full=self.calculate_full_failure, SF=1.0,fail_fast=not self.calculate_full_failure,failures=fc['beam_failures'])
        else:
            afc = self._current_combo_failure_analysis
        return afc                       

    def run_combo_failure_analysis(
        self,
        combo,
        run_full: bool = False,
        SF: float = 1.0,
        run_section_failure=None,
        fail_fast = True,
        failures = None
    ):
        """runs a single load combo and adds 2d section failures"""
        self.info(f"determine combo {combo} failures...")

        if failures is None:
            failures = self.estimated_failures_report(combos=combo)

        if run_section_failure is None:
            run_section_failure = self.run_failure_sections
            args = (failures, run_full, SF,fail_fast)
        else:
            args = (self, failures, run_full, SF,fail_fast)

        out = {}
        if combo == self.current_combo:
            self._current_combo_failure_analysis = out

        out['beam_failures_count'] = 0
        out['mesh_failure_count'] = 0
        out['failure_count'] = 0

        out["estimate"] = failures
        out["actual"] = check_fail = {}
        out["mesh_failures"] = mesh_failures = self.check_mesh_failures(
            combo, run_full, SF
        )
        # perform 2d analysis if any estimated
        if mesh_failures and not run_full:
            return out

        elif self.calculate_actual_failure and check_est_failures(failures):
            self.info(f"testing actual failures...")
            out["actual"] = dfail = run_section_failure(*args)
            act_fails = {x:f for beam,cfail in dfail.items() for (c,x),f in cfail.items() if f['fails']}
            out['beam_failures_count'] = len(act_fails)
            out['mesh_failure_count'] = len(mesh_failures) 
            out['failure_count'] = len(act_fails)+len(mesh_failures)
            if dfail and not run_full:
                self.info(f"found actual failure...")
                return out

        else:
            self.info(f"no estimated failures found")

        return out

    def run_failure_sections(
        self, failures, run_full: bool = False, SF: float = 1.0, fail_fast=True
    ):
        """takes estimated failures and runs 2D section FEA on those cases"""
        secton_results = {}

        # random beam analysis
        # for beamnm,est_fail_cases in failpairs:
        failed_beams = set()
        failed_pos = set()
        fail_count = 0
        for (beamnm, combo, x), dat in sort_max_estimated_failures(failures):
            c = combo
            self.debug(
                f'running beam sections: {beamnm},{combo} @ {x} {dat["fail_frac"]*100.}%'
            )
            beam = self.beams[beamnm]

            # prepare for output
            if beamnm not in secton_results:
                secton_results[beamnm] = {}

            f = dat["loads"]
            s = beam.get_stress_with_forces(**f)

            secton_results[beamnm][(combo, x)] = d_ = dat.copy()
            d_["combo"] = combo
            d_["x"] = x
            d_["beam"] = beamnm
            d_["stress_analysis"] = s
            d_["stress_results"] = sss = s.get_stress()
            d_["stress_vm_max"] = max([max(ss["sig_vm"]) for ss in sss])
            d_["fail_frac"] = ff = max(
                [
                    max(ss["sig_vm"] / beam.material.allowable_stress)
                    for ss in sss
                ]
            )
            d_["fails"] = fail = ff > 1 / SF

            if fail:
                fail_count += 1
                failed_beams.add(beamnm)
                failed_pos.add(x)
                self.debug(f"beam {beamnm} failed @ {x*100:3.0f}%| {c}")
                if fail_fast:
                    return secton_results
                if not run_full:
                    break  # next beam!
            else:
                self.debug(f"beam {beamnm} ok @ {x*100:3.0f}%| {c}")
    
        if failed_beams or failed_pos:
            self.warning(f"{fail_count:3.0f} across {failed_beams} @ {[f'{p*100:3.0f}' for p in failed_pos]}%| {c}")

        return secton_results





















    

#TODO: make a multiprocessing version of section failure analysis using shared memory or something
    
# Remote Sync Util (locally run with section)
def run_combo_failure_analysis(
    inst, combo, run_full: bool = False, SF: float = 1.0
):
    """runs a single load combo and adds 2d section failures"""
    inst.resetSystemLogs()
    # use parallel failure section
    return inst.run_combo_failure_analysis(
        combo, run_full, SF, run_section_failure=run_failure_sections
    )


def run_failure_sections(
    struct,
    failures,
    run_full: bool = False,
    SF: float = 1.0,
    fail_fast=True,
    group_size=12,
):
    """takes estimated failures and runs 2D section FEA on those cases"""

    struct.resetSystemLogs()

    secton_results = {}
    skip_beams = set()
    if not hasattr(struct, "_putbeam"):
        struct._putbeam = put_beams = {}
    else:
        put_beams = struct._putbeam

    cur = []

    for (beamnm, c, x), dat in sort_max_estimated_failures(failures):
        struct.info(
            f'run remote beam sections: {beamnm},{c} @ {x} {dat["fail_frac"]*100.}%'
        )
        if beamnm in skip_beams:
            continue

        # prep for output
        if beamnm not in secton_results:
            secton_results[beamnm] = {}
        beam = struct.beams[beamnm]

        forces = beam.get_forces_at(x, c)

        # lazy beam puts
        if beamnm not in put_beams:
            beam_ref = ray.put(beam._section_properties)
            put_beams[beamnm] = beam_ref
        else:
            beam_ref = put_beams[beamnm]

        cur.append(
            remote_section.remote(
                beam_ref,
                beam.name,
                forces,
                beam.material.allowable_stress,
                c,
                x,
                SF=SF,
            )
        )

        # run the damn thing
        if len(cur) >= group_size:
            struct.info(f"waiting on {group_size}")
            ray.wait(cur)

            for res in ray.get(cur):
                fail = res["fails"]
                beamnm = res["beam"]
                combo = res["combo"]
                x = res["x"]
                secton_results[res["beam"]][(combo, x)] = res

                if fail:
                    struct.warning(f"beam {beamnm} failed @ {x*100:3.0f}%| {c}")
                    if fail_fast:
                        return secton_results
                    if not run_full:
                        skip_beams.add(beamnm)
                else:
                    struct.debug(f"beam {beamnm} ok @ {x*100:3.0f}%| {c}")

            cur = []
            struct.info(f"done waiting, continue...")

    # finish them!
    if cur:
        struct.info(f"waiting on {len(cur)}")
        ray.wait(cur)
        struct.info(f"done, continue...")

        for res in ray.get(cur):
            fail = res["fails"]
            beamnm = res["beam"]
            combo = res["combo"]
            x = res["x"]
            secton_results[res["beam"]][(combo, x)] = res

            if fail:
                struct.warning(f"beam {beamnm} failed @ {x*100:3.0f}%| {c}")
                if fail_fast:
                    return secton_results
                if not run_full:
                    skip_beams.add(beamnm)
            else:
                struct.debug(f"beam {beamnm} ok @ {x*100:3.0f}%| {c}")

    return secton_results


# Remote Capabilities
@ray.remote(max_calls=5)
def remote_run_combo(ref_structure, combo):
    log = StructureLog()
    try:
        sync_ref = ref_structure["actor_ref"]
        struct_ref = ref_structure["struct_ref"]
        log.info(f"got {ref_structure}")
        struct = ray.get(struct_ref)
        # sync = ray.get(sync_ref)
        sync = sync_ref
        # try:
        struct.resetSystemLogs()
        log.info(f"start combo {combo}")
        # log.info(struct.logger.__dict__)
        struct.run(combos=combo)
        log.info(f"combo done {combo}")
        # assert len(struct.table) == 1, 'bad! sync only at beginning and end'
        row = struct.table[1]
        out = {
            "row": row,
            "D": struct._D[combo],
            "combo": combo,
            "st_id": struct.run_id,
        }
        log.info(f"putting results for {combo}")
        d = sync.put_combo.remote(out)
        ray.wait([d])
        return out

    except Exception as e:
        log.error(e, "issue with remote run")
        raise e


@ray.remote
def remote_section(
    beamsection,
    beamname,
    forces,
    allowable_stress,
    combo,
    x,
    SF=1.0,
    return_section=False,
):
    """optimized method to run 2d fea and return a failure determination"""
    # TODO: allow post processing ops
    s = beamsection.calculate_stress(**forces)

    sss = s.get_stress()
    d_ = {"loads": forces, "beam": beamname, "combo": combo, "x": x}
    if return_section:
        d_["stress_analysis"] = s
        d_["stress_results"] = sss
    d_["stress_vm_max"] = max([max(ss["sig_vm"]) for ss in sss])
    d_["fail_frac"] = ff = max(
        [max(ss["sig_vm"] / allowable_stress) for ss in sss]
    )
    d_["fails"] = fail = ff > 1 / SF

    return d_


# Remote Analysis
def parallel_run_failure_analysis(struct, SF=1.0, run_full=False, **kw):
    """
    Failure Determination:
    Beam Stress Estiates are used to determine if a 2D FEA beam/combo analysis should be run. The maximum beam vonmises stress is compared the the beam material allowable stress / saftey factor.

    Override:
    Depending on number of load cases, this analysis could take a while. Its encouraged to use a similar pattern of check and exit to reduce load combo size in an effort to fail early. Override this function if nessicary.

    Case / Beam Ordering:
    runs the gravity case first and checks for failure to exit early,
    then all other combos are run
    :returns: a dictionary with 2D fea failures and esimated failures as well for each beam
    """

    report = {"fails": False}  # innocent till proven guilty
    report["failures"] = failures = {}

    try:
        # Run the rest of the load cases
        for combo in struct.failure_load_combos(report, run_full, **kw):
            # TODO: handle running list of combos to do simultanious running

            # run and store data
            struct.run(combos=combo)
            # struct_ref = ray.put(struct)

            # check failures
            fail = run_combo_failure_analysis(struct, combo, run_full, SF)

            failures[combo] = fail
            struct.failure_load_exithandler(combo, fail, report, run_full)

    except KeyboardInterrupt:
        return report

    except Exception as e:
        struct.error(e, "issue in failur analysis")

    return report  # tada!


@ray.remote
def parallel_combo_run(combo, ref_structure, run_full=False, SF=1.0):
    # run and store data
    # self.struct.run(combos=combo)

    log = StructureLog()

    sync = ref_structure["actor_ref"]
    struct_ref = ref_structure["struct_ref"]
    log.info(f"running structure {combo}")
    d = remote_run_combo.remote(ref_structure, combo)
    log.info(f"waiting on combo {combo}")
    ray.wait([d])

    # check failures
    log.info(f"failure analysis {combo}")
    fail = remote_combo_failure_analysis(ref_structure, combo, run_full, SF)
    d = sync.put_failure.remote(combo, fail, run_full)
    log.info(f"failure analysis done {combo}")
    return ray.get(d)

# 
# @ray.remote
# class RemoteStructuralAnalysis:
#     """represents a stateful object in a ray cloud contetxt"""
# 
#     def __init__(self, structure, *kw):
#         self.struct = structure
#         self.struct.resetSystemLogs()
#         self.struct.prep_remote_analysis()
# 
#         # reference tracking
#         self.combos_run = {}
#         self.failed = False
#         self.report = {"fails": self.failed}
#         self.report["failures"] = failures = {}
#         self.failures = failures
#         self.put_beams = {}
# 
#         # can run all combos from base
#         self.struct_ref = ray.put(self.struct)
# 
#         # put beams sections for quick analysis
#         for beamnm, beam in self.struct.beams.items():
#             beam_ref = ray.put(beam._section_properties)
#             self.put_beams[beamnm] = beam_ref
# 
#     def get(self, run_id=None):
#         self.struct.info(f"getting {len(self.table)}")
#         if run_id is None:
#             return self.struct._table
#         else:
#             return [v for v in self.table if v["st_id"] == run_id]
# 
#     def put_combo(self, inv):
#         """add the data from remotely running the load combo"""
# 
#         self.struct.info(f"adding {len(self.struct.table)+1}")
#         self.struct._table[len(self.struct.table) + 1] = inv["row"]
#         self.struct.merge_displacement_results(inv["combo"], inv["D"])
#         self.combos_run[inv["combo"]] = inv
#         self.struct.info(f'finished analysis of {inv["combo"]}')
# 
#     def put_failure(self, combo, fail, run_full=False):
#         self.struct.info(f"finished failure calc of {combo}")
#         self.failures[combo] = fail
#         self.struct.failure_load_exithandler(combo, fail, self.report, run_full)
# 
#     async def wait_and_get_failure(self, combo, wait=1, timeout=None):
#         if combo in self.failures:
#             return self.failures[combo]
# 
#         timeelapsed = 0
# 
#         # wait loop
#         while combo not in self.failures:
#             self.struct.info(f"waiting on {combo}")
#             if timeout is not None and timeelapsed > timeout:
#                 raise TimeoutError(f"combo {combo} not found within timeout")
#             await asyncio.sleep(wait)
#             timeelapsed += wait
# 
#         return self.failures[combo]
# 
#     def get_beam_forces(self, beam_name, combo, x):
#         beam = self.struct.beams[beam_name]
#         return beam.get_forces_at(x, combo)
# 
#     def get_beam_info(self, beam_name):
#         r = self.put_beams[beam_name]
#         beam = self.struct.beams[beam_name]
#         out = {"ref": r, "allowable": beam.material.allowable_stress}
#         return out
# 
#     def estimated_failures(self, combos):
#         return self.struct.estimated_failures_report(combos=combos)
# 
#     def return_structure(self):
#         return self.struct
# 
#     async def ensure(self, combo, wait=1, timeout=None):
#         if combo in self.combos_run:
#             return  # good to go
# 
#         timeelapsed = 0
# 
#         # wait loop
#         while combo not in self.combos_run:
#             self.struct.info(f"waiting on {combo}")
#             if timeout is not None and timeelapsed > timeout:
#                 raise TimeoutError(f"combo {combo} not found within timeout")
#             await asyncio.sleep(wait)
#             timeelapsed += wait
# 
#         return  # good to go
# 
#     def run_failure_analysis(self, SF=1.0, run_full=False, **kw):
#         """
#         Failure Determination:
#         Beam Stress Estiates are used to determine if a 2D FEA beam/combo analysis should be run. The maximum beam vonmises stress is compared the the beam material allowable stress / saftey factor.
# 
#         Override:
#         Depending on number of load cases, this analysis could take a while. Its encouraged to use a similar pattern of check and exit to reduce load combo size in an effort to fail early. Override this function if nessicary.
# 
#         Case / Beam Ordering:
#         runs the gravity case first and checks for failure to exit early,
#         then all other combos are run
#         :returns: a dictionary with 2D fea failures and esimated failures as well for each beam
#         """
#         ctx = ray.get_runtime_context()
#         actor_handle = ctx.current_actor
#         try:
#             cases = []
#             ref = {"struct_ref": self.struct_ref, "actor_ref": actor_handle}
#             # Run the rest of the load cases
#             for combo in self.struct.failure_load_combos(
#                 self.report, run_full, remote_sync=actor_handle, **kw
#             ):
#                 d = parallel_combo_run.remote(
#                     combo, ref, run_full=run_full, SF=SF
#                 )
#                 cases.append(d)
# 
#             for coro in asyncio.as_completed(cases):
#                 if self.failed and not run_full:
#                     self.struct.warning(f"structure failed!")
#                     return self.report
# 
#         except KeyboardInterrupt:
#             return self.report
# 
#         except Exception as e:
#             self.struct.error(e, "issue in failur analysis")
# 
#         return self.report  # tada!


# Remote Sync Util
def remote_combo_failure_analysis(
    ref_structure, combo, run_full: bool = False, SF: float = 1.0
):
    """runs a single load combo and adds 2d section failures"""

    sync = ref_structure["actor_ref"]
    struct_ref = ref_structure["struct_ref"]

    log = StructureLog()

    log.info(f"determine combo {combo} failures...")

    out = {}
    d = sync.estimated_failures_report.remote(combos=combo)
    failures = ray.get(d)

    out["estimate"] = failures
    out["actual"] = check_fail = {}
    # perform 2d analysis if any estimated
    if check_est_failures(failures):
        log.info(f"testing estimated failures...")

        dfail = remote_failure_sections(sync, failures, run_full, SF)
        out["actual"] = dfail

        if dfail and not run_full:
            log.info(f"found actual failure...")
            return out
    else:
        log.info(f"no estimated failures found")

    return out


def remote_failure_sections(
    sync,
    failures,
    run_full: bool = False,
    SF: float = 1.0,
    fail_fast=True,
    group_size=12,
):
    """takes estimated failures and runs 2D section FEA on those cases"""
    log = StructureLog()

    secton_results = {}
    skip_beams = set()
    cur = []

    for (beamnm, c, x), dat in sort_max_estimated_failures(failures):
        log.info(
            f'run remote beam sections: {beamnm},{c} @ {x} {dat["fail_frac"]*100.}%'
        )
        if beamnm in skip_beams:
            continue

        # prep for output
        if beamnm not in secton_results:
            secton_results[beamnm] = {}

        # get from sync
        ray.get(sync.ensure.remote(c))
        d = sync.get_beam_forces.remote(beamnm, c, x)
        r = sync.get_beam_info.remote(beamnm)
        forces, beam_ref = ray.get([d, r])
        beam_ref = r["ref"]
        allowable = r["allowable"]

        cur.append(
            remote_section.remote(
                beam_ref, beamnm, forces, allowable, c, x, SF=SF
            )
        )

        # run the damn thing
        if len(cur) >= group_size:
            log.info(f"waiting on {group_size}")
            ray.wait(cur)

            for res in ray.get(cur):
                fail = res["fails"]
                beamnm = res["beam"]
                combo = res["combo"]
                x = res["x"]
                secton_results[res["beam"]][(combo, x)] = res

                if fail:
                    log.warning(f"beam {beamnm} failed @ {x*100:3.0f}%| {c}")
                    if fail_fast:
                        return secton_results
                    if not run_full:
                        skip_beams.add(beamnm)
                # else:
                # log.info(f'beam {beamnm} ok @ {x*100:3.0f}%| {c}')

            cur = []
            log.info(f"done waiting, continue...")

    # finish them!
    if cur:
        log.info(f"waiting on {len(cur)}")
        ray.wait(cur)
        log.info(f"done, continue...")

        for res in ray.get(cur):
            fail = res["fails"]
            beamnm = res["beam"]
            combo = res["combo"]
            x = res["x"]
            secton_results[res["beam"]][(combo, x)] = res

            if fail:
                log.warning(f"beam {beamnm} failed @ {x*100:3.0f}%| {c}")
                if fail_fast:
                    return secton_results
                if not run_full:
                    skip_beams.add(beamnm)
            # else:
            # log.info(f'beam {beamnm} ok @ {x*100:3.0f}%| {c}')

    return secton_results
