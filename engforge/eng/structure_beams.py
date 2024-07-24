from collections.abc import Iterable
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
from engforge.eng.solid_materials import *
from engforge.common import *
import engforge.eng.geometry as ottgeo
from engforge.eng.costs import CostModel, cost_property

import sectionproperties
import sectionproperties.pre.geometry as geometry
import sectionproperties.pre.library.primitive_sections as sections
import sectionproperties.analysis.section as cross_section
import PyNite as pynite

# from PyNite import Visualization
import copy

nonetype = type(None)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / numpy.linalg.norm(vec1)).reshape(3), (
        vec2 / numpy.linalg.norm(vec2)
    ).reshape(3)
    v = numpy.cross(a, b)
    if any(v):  # if not all zeros then
        c = numpy.dot(a, b)
        s = numpy.linalg.norm(v)
        kmat = numpy.array(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
        )
        return numpy.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

    else:
        return numpy.eye(
            3
        )  # cross of all zeros only occurs on identical directions


@forge
class Beam(Component, CostModel):
    """Beam is a wrapper for emergent useful properties of the structure"""

    # parent structure, will be in its _beams
    structure: "Structure" = attr.ib()

    name: str = attr.ib()
    material: "SolidMaterial" = attr.ib(
        validator=attr.validators.instance_of(SolidMaterial),
    )
    section: "Profile2D" = attr.ib(
        validator=attr.validators.instance_of(
            (
                geometry.Geometry,
                ottgeo.Profile2D,
                type(None),
            )
        )
    )

    # Section Overrides
    in_Iy: float = attr.ib(
        default=None,
        validator=attr.validators.instance_of((int, float, nonetype)),
    )
    in_Ix: float = attr.ib(
        default=None,
        validator=attr.validators.instance_of((int, float, nonetype)),
    )
    in_J: float = attr.ib(
        default=None,
        validator=attr.validators.instance_of((int, float, nonetype)),
    )
    in_A: float = attr.ib(
        default=None,
        validator=attr.validators.instance_of((int, float, nonetype)),
    )
    # by default assume symmetric
    in_Ixy: float = attr.ib(
        default=0.0,
        validator=attr.validators.instance_of((int, float, nonetype)),
    )

    min_mesh_size: float = attrs.field(default=0.01)
    analysis_intervals: int = attrs.field(default=3)

    _L = None
    _ITensor = None

    min_stress_xy = None  # set to true or false

    def __on_init__(self):
        self.debug("initalizing...")
        # update material
        if isinstance(self.section, ottgeo.ShapelySection):
            if self.section.material is None:
                raise Exception("No Section Material")
            else:
                self.material = self.section.material
        self.update_section(self.section)

    def update_section(self, section):
        self.section = section

        self.debug(f"determining {section} properties...")
        if isinstance(self.section, geometry.Geometry):
            self.debug(f"converting to shapelysection {section}")
            if not self.section.material:
                material = self.material
            else:
                material = self.section.material
            self.section = ottgeo.ShapelySection(
                shape=self.section, material=self.material
            )

        if isinstance(self.section, ottgeo.ShapelySection):
            self.debug(f"determining profile {section} properties...")
            self.in_Ix = self.section.Ixx
            self.in_Iy = self.section.Iyy
            self.in_J = self.section.J
            self.in_A = self.section.A
            self.in_Ixy = self.section.Ixy

        elif isinstance(self.section, ottgeo.Profile2D):
            self.warning(f"use shapely section instead {section} properties...")
            # raise Exception("use shapely section instead")
            self.in_Ix = self.section.Ixx
            self.in_Iy = self.section.Iyy
            self.in_J = self.section.J
            self.in_A = self.section.A
        else:
            raise Exception(f"unhanelded input {self.section}")

        self.debug(f"checking input values")
        assert all(
            [
                val is not None
                for val in (self.in_Iy, self.in_Ix, self.in_J, self.in_A)
            ]
        )

        # Assemble tensor
        T = [[self.in_Ix, self.in_Ixy], [self.in_Ixy, self.in_Iy]]
        self._ITensor = numpy.array(T)

    @system_property
    def current_combo(self) -> str:
        return self.structure.current_combo

    @system_property
    def L(self) -> float:
        return self.length

    @system_property
    def length(self) -> float:
        return self.member.L()

    @property
    def member(self):
        return self.structure.members[self.name]

    @property
    def n1(self):
        return self.member.i_node

    @property
    def n2(self):
        return self.member.j_node

    @property
    def P1(self):
        return numpy.array([self.n1.X, self.n1.Y, self.n1.Z])

    @property
    def P2(self):
        return numpy.array([self.n2.X, self.n2.Y, self.n2.Z])

    @system_property
    def E(self) -> float:
        return self.material.E

    @system_property
    def G(self) -> float:
        return self.material.G

    @property
    def ITensor(self):
        return self._ITensor

    @system_property
    def Iy(self) -> float:
        return self.in_Iy

    @system_property
    def Ix(self) -> float:
        return self.in_Ix

    @system_property
    def J(self) -> float:
        return self.in_J

    @system_property
    def A(self) -> float:
        return self.in_A

    @property
    def Ao(self):
        """outside area, over ride for hallow sections"""
        if isinstance(self.section, ottgeo.Profile2D):
            return self.section.Ao
        return self.A

    @system_property
    def Ixy(self) -> float:
        return self.ITensor[0, 1]

    @system_property
    def Imx(self) -> float:
        return self.material.density * self.Ix

    @system_property
    def Imy(self) -> float:
        return self.material.density * self.Iy

    @system_property
    def Imxy(self) -> float:
        return self.material.density * self.Ixy

    @system_property
    def Jm(self) -> float:
        return self.material.density * self.J

    @system_property
    def Imz(self) -> float:
        return self.mass * self.L**2.0 / 12.0

    @system_property
    def Iz(self) -> float:
        """Outside area inertia on z"""
        return self.A * self.L**2.0 / 12.0

    @system_property
    def Izo(self) -> float:
        """Outside area inertia on z"""
        return self.Ao * self.L**2.0 / 12.0

    @property
    def LOCAL_INERTIA(self):
        """the mass inertia tensor in local frame"""
        return numpy.array(
            [
                [self.Imz + self.Imx, self.Imxy, 0.0],
                [self.Imxy, self.Imz + self.Imy, 0.0],
                [0.0, 0.0, self.Jm],
            ]
        )

    @property
    def INERTIA(self):
        """the mass inertia tensor in global structure frame"""
        # TODO: include rotation from beam uv vector frame? is it ok already?
        return self.RotationMatrix.dot(self.LOCAL_INERTIA)

    @property
    def GLOBAL_INERTIA(self):
        """Returns the rotated inertia matrix with structure relative parallal axis contribution"""
        return self.INERTIA + self.CG_RELATIVE_INERTIA

    @property
    def CG_RELATIVE_INERTIA(self):
        """the mass inertia tensor in global frame"""
        dcg = self.cog - self.structure.cog
        return self.mass * numpy.eye(3) * dcg * dcg.T

    @system_property
    def Vol(self) -> float:
        return self.A * self.L

    @system_property
    def Vol_outside(self) -> float:
        return self.Ao * self.L

    @system_property
    def section_mass(self) -> float:
        return self.material.density * self.A

    @system_property
    def mass(self) -> float:
        return self.material.density * self.Vol

    @cost_property(category="mfg,material,beams")
    def cost(self) -> float:
        return self.mass * self.material.cost_per_kg

    @property
    def centroid2d(self):
        return self.section._sec.get_c()

    @instance_cached
    def centroid3d(self):
        return self.L_vec / 2.0 + self.P1

    @property
    def n_vec(self):
        return self.L_vec / self.L

    @property
    def L_vec(self):
        return self.P2 - self.P1

    @property
    def cog(self):
        return self.centroid3d

    # Geometry Tabulation
    # p1
    @system_property
    def X1(self) -> float:
        return self.P1[0]

    @system_property
    def Y1(self) -> float:
        return self.P1[1]

    @system_property
    def Z1(self) -> float:
        return self.P1[2]

    # p2
    @system_property
    def X2(self) -> float:
        return self.P2[0]

    @system_property
    def Y2(self) -> float:
        return self.P2[1]

    @system_property
    def Z2(self) -> float:
        return self.P2[2]

    # cg
    @system_property
    def Xcg(self) -> float:
        return self.centroid3d[0]

    @system_property
    def Ycg(self) -> float:
        return self.centroid3d[1]

    @system_property
    def Zcg(self) -> float:
        return self.centroid3d[2]

    @property
    def RotationMatrix(self):
        n_o = [
            1,
            0,
            0,
        ]
        # n_vec is along Z, so we must tranlate from the along axis which is z
        return rotation_matrix_from_vectors(n_o, self.n_vec)

    @property
    def ReverseRotationMatrix(self):
        # FIXME: Ensure that this is the correct orientation
        return self.RotationMatrix.T

    def section_results(self):
        return self.section._sec.display_results()

    def show_mesh(self):
        return self.section._sec.plot_mesh()

    def estimate_stress(self, force_calc=True, **forces):
        """uses the best available method to determine max stress in the beam, for ShapelySections this is done through a learning process, for other sections it is done through a simple calculation aimed at providing a conservative estimate"""
        if isinstance(self.section, ottgeo.ShapelySection):
            return self.section.estimate_stress(**forces, force_calc=force_calc)
        else:
            return self._fallback_estimate_stress(**forces)

    def _fallback_estimate_stress(
        self, n, vx, vy, mxx, myy, mzz, SM=5, **extra
    ):
        """sum the absolute value of each stress component. This isn't accurate but each value here should represent the worst sections, and take the 1-norm to max for each type of stress"""

        if self.section.x_bounds is None:
            self.warning(f"bad section bounds")
            self.section.calculate_bounds()

        m_y = max([abs(v) for v in self.section.y_bounds])
        m_x = max([abs(v) for v in self.section.x_bounds])

        sigma_n = n / self.in_A

        sigma_bx = mxx * m_y / self.in_Ix
        sigma_by = myy * m_x / self.in_Iy
        sigma_tw = mzz * max(m_y, m_x) / self.in_J

        # A2 = self.in_A/2.
        # experimental
        # sigma_vx = vx * (A2*m_y/2) / (self.in_Iy*m_x)
        # sigma_vy = vy * (A2*m_x/2) / (self.in_Ix*m_y)

        # assume circle (worst case)
        sigma_vx = vx * 0.75 / self.in_A
        sigma_vy = vy * 0.75 / self.in_A

        max_bend = abs(sigma_bx) + abs(sigma_by)
        max_shear = abs(sigma_vx) + abs(sigma_vy)

        return (abs(sigma_n) + max_bend + abs(sigma_tw) + max_shear) * SM

    def get_forces_at(self, x, combo=None, lower=True, skip_principle=True):
        """outputs pynite results in section_properties.calculate_stress() input"""
        if combo is None:
            combo = self.structure.current_combo

        x = x * self.L  # frac of L
        principles = ["m11", "m22", "m33"]

        inp = dict(
            N=self.member.axial(x, combo),
            Vx=self.member.shear("Fz", x, combo),
            Vy=self.member.shear("Fy", x, combo),
            Mxx=self.member.moment("Mz", x, combo),
            Myy=self.member.moment("My", x, combo),
            M11=0,
            M22=0,
            Mzz=self.member.torque(x, combo),
        )
        inp = {
            k.lower(): v
            for k, v in inp.items()
            if any([k.lower() not in principles]) or not skip_principle
        }
        return inp

    def get_stress_at(self, x, combo=None, **kw):
        """takes force input and runs stress calculation as a fraction of material properties"""
        if combo is None:
            combo = self.structure.current_combo

        inp = self.get_forces_at(x, combo)

        # preferr real stresses to fail fraction
        if "value" not in kw:
            kw["value"] = True

        return self.calculate_stress(**inp, **kw)

    def calculate_stress(self, **forces):
        """takes force input and runs stress calculation as a fraction of material properties, see ShapelySection.calculate_stress() for more details"""
        # preferr real stresses to fail fraction
        if "value" not in forces:
            forces["value"] = True

        return self.section.calculate_stress(**forces)

    @property
    def Fg(self):
        """force of gravity"""
        return numpy.array([0, 0, -self.mass * g])

    # RESULTS:
    @system_property
    def max_stress_estimate(self) -> float:
        """estimates these are proportional to stress but 2D FEA is "truth" since we lack cross section specifics"""
        if not self.structure._any_solved:
            return numpy.nan

        vals = [
            self.estimate_stress(**self.get_forces_at(x), value=True)
            for x in [0, 0.5, 1]
        ]

        try:
            return max(vals)

        except Exception as e:
            self.warning(f"failed to get max stress estimate {e}| {vals}")
            return numpy.nan

    @system_property
    def fail_factor_estimate(self) -> float:
        """the ratio of max estimated stress to the material's allowable stress"""
        if not self.structure._any_solved:
            return numpy.nan
        return self.max_stress_estimate / self.material.allowable_stress

    # axial
    @system_property
    def min_axial(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.min_axial(self.structure.current_combo)

    @system_property
    def max_axial(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.max_axial(self.structure.current_combo)

    # deflection
    @system_property
    def min_deflection_x(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.min_deflection("dx", self.structure.current_combo)

    @system_property
    def max_deflection_x(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.max_deflection("dx", self.structure.current_combo)

    @system_property
    def min_deflection_y(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.min_deflection("dy", self.structure.current_combo)

    @system_property
    def max_deflection_y(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.max_deflection("dy", self.structure.current_combo)

    # torsion
    @system_property
    def min_torsion(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.min_torque(self.structure.current_combo)

    @system_property
    def max_torsion(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.max_torque(self.structure.current_combo)

    # shear
    @system_property
    def min_shear_z(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.min_shear("Fz", self.structure.current_combo)

    @system_property
    def max_shear_z(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.max_shear("Fz", self.structure.current_combo)

    @system_property
    def min_shear_y(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.min_shear("Fy", self.structure.current_combo)

    @system_property
    def max_shear_y(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.max_shear("Fy", self.structure.current_combo)

    # moment
    @system_property
    def min_moment_z(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.min_moment("Mz", self.structure.current_combo)

    @system_property
    def max_moment_z(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.max_moment("Mz", self.structure.current_combo)

    @system_property
    def min_moment_y(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.min_moment("My", self.structure.current_combo)

    @system_property
    def max_moment_y(self) -> float:
        if not self.structure._any_solved:
            return numpy.nan
        return self.member.max_moment("My", self.structure.current_combo)

    # Load Application
    def get_valid_force_choices(
        only_local=False, only_global=False, use_moment=True
    ):
        if only_local or only_global:
            assert only_global != only_local, "choose local or global"

        floc = set(["Fx", "Fy", "Fz"])
        fglb = set(["FX", "FY", "FZ"])
        mloc = set(["Mx", "My", "Mz"])
        mglb = set(["MX", "MY", "MZ"])

        if only_global:
            if use_moment:
                return list(set.union(*(fglb, mglb)))
            return list(fglb)

        elif only_local:
            if use_moment:
                return list(set.union(*(floc, mloc)))
            return list(floc)

        else:
            if use_moment:
                return list(set.union(*(floc, mloc, fglb, mglb)))
            return list(set.union(*(floc, fglb)))

    # FORCE APPLICATION (TODO: update for global input 0.0.78)
    def apply_pt_load(self, x_frac, case=None, **kwargs):
        """add a force in a global orientation"""
        if case is None:
            case = self.structure.default_case

        # adjust x for relative input
        x = x_frac * self.L

        valid = self.get_valid_force_choices(use_moment=True, only_global=True)
        fin = [(v, kwargs[v]) for v in valid if v in kwargs]
        for Fkey, Fval in fin:
            if Fval:
                self.debug(f"adding {Fkey}={Fval}")
                self.structure.frame.add_member_pt_load(
                    self.member.name, Fkey, Fval, x, case=case
                )

    def apply_distributed_load(
        self, start_factor=1, end_factor=1, case=None, **kwargs
    ):
        """add forces in global vector"""
        if case is None:
            case = self.structure.default_case
        valid = self.get_valid_force_choices(use_moment=False, only_global=True)
        fin = [(v, kwargs[v]) for v in valid if v in kwargs]
        for Fkey, Fval in fin:
            if Fval:
                self.debug(f"adding dist {Fkey}={Fval}")
                self.structure.frame.add_member_dist_load(
                    self.member.name,
                    Fkey,
                    Fval * start_factor,
                    Fval * end_factor,
                    case=case,
                )

    def apply_local_pt_load(self, x, case=None, **kwargs):
        """add a force in a global orientation"""
        if case is None:
            case = self.structure.default_case
        valid = self.get_valid_force_choices(only_local=True, use_moment=True)
        fin = [(v, kwargs[v]) for v in valid if v in kwargs]
        for Fkey, Fval in fin:
            if Fval:
                self.debug(f"adding {Fkey}={Fval}")
                self.structure.frame.add_member_pt_load(
                    self.member.name, Fkey, Fval, x, case=case
                )

    def apply_local_distributed_load(
        self, start_factor=1, end_factor=1, case=None, **kwargs
    ):
        """add forces in global vector"""
        if case is None:
            case = self.structure.default_case
        valid = self.get_valid_force_choices(only_local=True, use_moment=False)
        fin = [(v, kwargs[v]) for v in valid if v in kwargs]
        for Fkey, Fval in fin:
            if Fval:
                self.debug(f"adding dist {Fkey}={Fval}")
                self.structure.frame.add_member_dist_load(
                    self.member.name,
                    Fkey,
                    Fval * start_factor,
                    Fval * end_factor,
                    case=case,
                )

    def apply_gravity_force_distribution(
        self, sv=1, ev=1, z_dir="FZ", z_mag=-1
    ):
        # TODO: ensure that integral of sv, ev is 1, and all positive
        self.debug(f"applying gravity distribution to {self.name}")
        for case in self.structure.gravity_cases:
            total_weight = self.mass * self.structure.gravity_mag
            d = {z_dir: z_mag * total_weight}
            self.apply_distributed_load(case=self.structure.gravity_name, **d)

    def apply_gravity_force(self, x_frac=0.5, z_dir="FZ", z_mag=-1):
        self.debug(f"applying gravity to {self.name}")
        for case in self.structure.gravity_cases:
            total_weight = self.mass * self.structure.gravity_mag
            d = {z_dir: z_mag * total_weight}
            self.apply_pt_load(x_frac, case=self.structure.gravity_name, **d)

    def __dir__(self) -> Iterable[str]:
        d = set(super().__dir__())
        return list(d.union(dir(Beam)))

    def max_von_mises(self) -> float:
        """The worst of the worst cases, after adjusting the beem orientation for best loading"""
        # TODO: make faster system property
        return numpy.nanmax([self.max_von_mises_by_case()])

    def max_von_mises_by_case(self, combos=None):
        """Gathers max vonmises stress info per case"""

        cmprv = {}
        for rxy in [True, False]:
            new = []
            out = self.von_mises_stress_l
            for cmbo, vm_stress_vec in out.items():
                if combos and cmbo not in combos:
                    continue
                new.append(numpy.nanmax(vm_stress_vec))
            cmprv[rxy] = numpy.array(new)

        if cmprv[True] and cmprv[False]:
            vt = numpy.nanmax(cmprv[True])
            vf = numpy.nanmax(cmprv[False])

            # We choose the case with the
            if vf < vt:
                self.min_stress_xy = False
                return vf

            self.min_stress_xy = True
            return vt

    # TODO: Breakout other stress vectors
    @solver_cached
    def von_mises_stress_l(self):
        """Max von-mises stress"""
        out = {}
        sect_stresses = self.section_stresses()
        for combo in self.structure.frame.LoadCombos:
            rows = []
            for i in numpy.linspace(0, 1, self.analysis_intervals):
                max_vm = sect_stresses[combo][i]
                rows.append(max_vm)
            out[combo] = numpy.array(rows)
        return out

    def section_stresses(self, **kwargs):
        # FIXME: enable: assert self.structure.solved, f'must be solved first!'
        combos = {}
        for combo in self.structure.frame.LoadCombos:
            combos[combo] = spans = {}
            for i in numpy.linspace(0, 1, self.analysis_intervals):
                self.info(f"evaluating stresses for {combo} @ {i} w {kwargs}")
                sol = self.get_stress_at(i, combo, **kwargs)
                spans[i] = sol
        return combos

    # #TODO: remove or refactor
    # @solver_cached
    # def stress_info(self):
    #     """Max profile stress info along beam for each type"""
    #     rows = []
    #     for combo in self.structure.frame.LoadCombos:
    #         for i in numpy.linspace(0, 1, self.analysis_intervals):
    #             mat_stresses = self.section_stresses[combo][i].get_stress()
    #             oout = {"x": i, "combo": combo}
    #             for stresses in mat_stresses:
    #                 max_vals = {
    #                     sn
    #                     + "_max_"
    #                     + stresses["Material"]: numpy.nanmax(stress)
    #                     for sn, stress in stresses.items()
    #                     if isinstance(stress, numpy.ndarray)
    #                 }
    #                 min_vals = {
    #                     sn
    #                     + "_min_"
    #                     + stresses["Material"]: numpy.nanmin(stress)
    #                     for sn, stress in stresses.items()
    #                     if isinstance(stress, numpy.ndarray)
    #                 }
    #                 avg_vals = {
    #                     sn
    #                     + "_avg_"
    #                     + stresses["Material"]: numpy.nanmean(stress)
    #                     for sn, stress in stresses.items()
    #                     if isinstance(stress, numpy.ndarray)
    #                 }
    #                 # Make some simple to determine dataframe failure prediction
    #                 factor_of_saftey = (
    #                     self.material.yield_strength
    #                     / numpy.nanmax(stresses["sig_vm"])
    #                 )
    #                 fail_frac = (
    #                     numpy.nanmax(stresses["sig_vm"])
    #                     / self.material.allowable_stress
    #                 )
    #                 fsnm = stresses["Material"] + "_saftey_factor"
    #                 fsff = stresses["Material"] + "_fail_frac"
    #                 allowable = {fsnm: factor_of_saftey, fsff: fail_frac}
    #                 oout.update(allowable)
    #                 oout.update(max_vals)
    #                 oout.update(min_vals)
    #                 oout.update(avg_vals)
    #             rows.append(oout)
    #
    #     return pandas.DataFrame(rows)
