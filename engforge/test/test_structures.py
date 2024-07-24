import unittest
from matplotlib import pylab
from engforge.eng.structure import *
from engforge.eng.geometry import ShapelySection
from sectionproperties.pre.library.steel_sections import i_section

from numpy import *


class test_cantilever(unittest.TestCase):
    # tests the first example here
    # https://www.engineeringtoolbox.com/cantilever-beams-d_1848.html

    def setUp(self):
        self.st = Structure(name="cantilever_beam", add_gravity_force=False)
        self.st.add_node("wall", 0, 0, 0)
        self.st.add_node("free", 0, 5, 0)

        self.ibeam = i_section(0.3072, 0.1243, 0.0121, 0.008, 0.0089, 4)
        beam = ShapelySection(shape=self.ibeam, material=ANSI_4130())
        self.bm = self.st.add_member("mem", "wall", "free", section=beam)

        self.st.def_support(
            "wall",
            support_DX=True,
            support_DY=True,
            support_DZ=True,
            support_RY=True,
            support_RX=True,
            support_RZ=True,
        )
        self.st.frame.add_node_load("free", "FX", 3000, case="gravity")

        self.st.run(check_statics=True)

    def test_beam(self):
        self.subtest_assert_near(self.bm.A, 53.4 / (100**2))
        self.subtest_assert_near(self.bm.Ix, 8196 / (100**4))
        self.subtest_assert_near(self.bm.Iy, 388.8 / (100**4))
        self.subtest_assert_near(self.bm.section_mass, 41.9)

        self.subtest_assert_near(self.bm.max_von_mises(), 27.4e6)
        self.subtest_assert_near(
            float(self.bm.data_dict["min_deflection_y"]), -0.0076
        )
        # self.subtest_assert_near(float(self.bm.data_dict["max_shear_y"]), 3000)
        self.subtest_assert_near(float(self.bm.data_dict["max_shear_y"]), 3000)

        df = self.st.node_dataframe.loc["gravity"]

        dfw = df.loc["wall"]
        dff = df.loc["free"]

        self.subtest_assert_near(float(dfw["rxfx"]), -3000, msg="wall rxfx")
        self.subtest_assert_near(float(dfw["rxmz"]), 15000, msg="wall rxmz")
        self.subtest_assert_near(float(dfw["dx"]), 0, msg="wall dx")

        self.subtest_assert_near(float(dff["dx"]), 0.0076, msg="dx")
        self.subtest_assert_near(float(dff["rxfx"]), 0, msg="rxfx")
        self.subtest_assert_near(float(dff["rxmz"]), 0, msg="rxmz")

        stress_obj = self.bm.get_stress_at(0, "gravity")
        # stress_obj.plot_stress_vm()

    def subtest_assert_near(self, value, truth, pct=0.025, **kw):
        with self.subTest(**kw):
            self.assertAlmostEqual(
                value, truth, delta=max(abs(truth * pct), abs(pct))
            )


class test_truss(unittest.TestCase):
    # Match this example, no beam stresses
    # https://engineeringlibrary.org/reference/trusses-air-force-stress-manual

    def setUp(self):
        self.st = Structure(name="truss", add_gravity_force=False)
        self.st.add_node("A", 0, 0, 0)
        self.st.add_node("B", 15, 30 * sqrt(3) / 2, 0)
        self.st.add_node("C", 45, 30 * sqrt(3) / 2, 0)
        self.st.add_node("D", 75, 30 * sqrt(3) / 2, 0)
        self.st.add_node("E", 90, 0, 0)
        self.st.add_node("F", 60, 0, 0)
        self.st.add_node("G", 30, 0, 0)

        pairs = set()
        Lmin = 30 * sqrt(3) / 2
        Lmax = 30.1

        for n1 in self.st.nodes.values():
            for n2 in self.st.nodes.values():
                L = numpy.sqrt(
                    (n1.X - n2.X) ** 2.0
                    + (n1.Y - n2.Y) ** 2.0
                    + (n1.Z - n2.Z) ** 2.0
                )

                if (
                    L >= Lmin
                    and L <= Lmax
                    and (n1.name, n2.name) not in pairs
                    and (n2.name, n1.name) not in pairs
                ):
                    pairs.add((n1.name, n2.name))

                elif (n1.name, n2.name) in pairs or (
                    n2.name,
                    n1.name,
                ) in pairs:
                    pass

        self.beam = sections.rectangular_section(1.5, 1.5)
        material = ANSI_4130()
        self.section = ShapelySection(shape=self.beam, material=material)

        constrained = ("A", "E")
        for n1, n2 in pairs:
            bkey = f"{n1}_{n2}"
            self.bm = self.st.add_member(
                bkey,
                n1,
                n2,
                # material=ANSI_4130(),
                section=self.section,
                # min_mesh_size=0.01,
            )

            # if n1 not in constrained:
            #     print(f'releasing {bkey}')
            #     self.st.frame.DefineReleases(bkey, Rzi=True)

            # if n2 not in constrained:
            #     print(f'releasing {bkey} J')
            #     self.st.frame.DefineReleases(bkey, Rzj=True)

        self.st.def_support(
            "A",
            support_DX=True,
            support_DY=True,
            support_DZ=True,
            support_RY=True,
            support_RX=True,
            support_RZ=True,
        )
        self.st.def_support(
            "E",
            support_DX=False,
            support_DY=True,
            support_DZ=True,
            support_RY=False,
            support_RX=False,
            support_RZ=True,
        )
        # for node in self.st.nodes:
        #     self.st.frame.Definesupport_(node,support_DZ=True,support_RZ=True)

        self.st.add_node_load("F", "FY", -1000, case="gravity")
        self.st.add_node_load("G", "FY", -2000, case="gravity")

        self.st.run(check_statics=True)
        # self.st.visulize()

    def test_reactions(self):
        df = self.st.node_dataframe.loc["gravity"]
        # print(df)

        dfa = df.loc["A"]
        dfe = df.loc["E"]

        # print(dfa)
        # print(dfe)

        self.subtest_assert_near(float(dfa["rxfy"]), 1667)
        self.subtest_assert_near(float(dfe["rxfy"]), 1333)

        self.subtest_member("A", "B", "max_axial", 1925)
        self.subtest_member("A", "G", "max_axial", -949)  # -926 textbook
        self.subtest_member("B", "C", "max_axial", 1925)
        self.subtest_member("B", "G", "max_axial", -1925)
        self.subtest_member("C", "D", "max_axial", 1541)
        self.subtest_member("F", "G", "max_axial", -1734)
        self.subtest_member("C", "F", "max_axial", 382)
        self.subtest_member("C", "G", "max_axial", -382)
        self.subtest_member("C", "G", "max_axial", -378)  # 1541 textbook
        self.subtest_member("D", "F", "max_axial", -1541)
        self.subtest_member("E", "F", "max_axial", -770)

        # Visualization.RenderModel( self.st.frame )

    def subtest_member(self, nodea, nodeb, result_key, truth, pct=0.025):
        with self.subTest(msg=f"test {nodea}->{nodeb}"):
            key_1 = f"{nodea}_{nodeb}"
            key_2 = f"{nodeb}_{nodea}"

            if key_1 in self.st.beams:
                key = key_1
            elif key_2 in self.st.beams:
                key = key_2
            else:
                raise

            value = self.get_member_result(nodea, nodeb, result_key)
            dopasst = abs(value - truth) <= abs(truth) * pct

            if not dopasst:
                print(
                    f"fails {key} {result_key}| {value:3.5f} == {truth:3.5f}?"
                )
            self.subtest_assert_near(value, truth, pct=pct)

    def subtest_assert_near(self, value, truth, pct=0.025):
        with self.subTest():
            self.assertAlmostEqual(value, truth, delta=abs(truth * pct))

    def get_member_result(self, nodea, nodeb, result_key):
        key_1 = f"{nodea}_{nodeb}"
        key_2 = f"{nodeb}_{nodea}"

        if key_1 in self.st.beams:
            mem = self.st.beams[key_1]
            if result_key in mem.data_dict:
                return float(mem.data_dict[result_key])

        elif key_2 in self.st.beams:
            mem = self.st.beams[key_2]
            if result_key in mem.data_dict:
                return float(mem.data_dict[result_key])

        return numpy.nan  # shouod fail, nan is not comparable
