import unittest
from matplotlib import pylab
from ottermatics.eng.structure import *


class test_cantilever(unittest.TestCase):
    # tests the first example here
    # https://www.engineeringtoolbox.com/cantilever-beams-d_1848.html

    def setUp(self):
        self.st = Structure(name="cantilever_beam")
        self.st.add_node("wall", 0, 0, 0)
        self.st.add_node("free", 0, 5, 0)

        self.ibeam = sections.ISection(0.3072, 0.1243, 0.0121, 0.008, 0.0089, 4)
        self.bm = self.st.add_member(
            "mem", "wall", "free", material=ANSI_4130(), section=self.ibeam
        )

        self.st.add_constraint(
            "wall",
            SupportDX=True,
            SupportDY=True,
            SupportDZ=True,
            SupportRY=True,
            SupportRX=True,
            SupportRZ=True,
        )
        self.st.frame.AddNodeLoad("free", "FX", 3000)

        self.st.analyze(check_statics=True)

    def test_beam(self):
        self.subtest_assert_near(self.bm.A, 53.4 / (100**2))
        self.subtest_assert_near(self.bm.Ix, 8196 / (100**4))
        self.subtest_assert_near(self.bm.Iy, 388.8 / (100**4))
        self.subtest_assert_near(self.bm.section_mass, 41.9)

        self.subtest_assert_near(self.bm.max_von_mises, 27.4e6)
        self.subtest_assert_near(
            float(self.bm.results["min_deflection_y"]), -0.0076
        )
        self.subtest_assert_near(float(self.bm.results["max_shear_y"]), 3000)
        self.subtest_assert_near(float(self.bm.results["max_shear_y"]), 3000)

        df = self.st.node_dataframes["Combo 1"]

        dfw = df[df["name"] == "wall"]
        dff = df[df["name"] == "free"]

        self.subtest_assert_near(float(dfw["rxfx"]), -3000)
        self.subtest_assert_near(float(dfw["rxmz"]), 15000)
        self.subtest_assert_near(float(dfw["dx"]), 0)

        self.subtest_assert_near(float(dff["dx"]), 0.0076)
        self.subtest_assert_near(float(dff["rxfx"]), 0)
        self.subtest_assert_near(float(dff["rxmz"]), 0)

        stress_obj = self.bm.get_stress_at(0, "Combo 1")
        stress_obj.plot_stress_vm()

    def subtest_assert_near(self, value, truth, pct=0.025):
        with self.subTest():
            self.assertAlmostEqual(value, truth, delta=abs(truth * pct))


class test_truss(unittest.TestCase):
    # Match this example, no beam stresses
    # https://engineeringlibrary.org/reference/trusses-air-force-stress-manual

    def setUp(self):
        self.st = Structure(name="truss")
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
                    and (n1.Name, n2.Name) not in pairs
                    and (n2.Name, n1.Name) not in pairs
                ):
                    # print(f'adding {(n1.Name,n2.Name)}')
                    pairs.add((n1.Name, n2.Name))

                elif (n1.Name, n2.Name) in pairs or (
                    n2.Name,
                    n1.Name,
                ) in pairs:
                    pass
                    # print(f'skipping {(n1.Name,n2.Name)}, already in pairs')

        self.beam = sections.RectangularSection(0.5, 0.5)

        constrained = ("A", "E")
        for n1, n2 in pairs:
            bkey = f"{n1}_{n2}"
            self.bm = self.st.add_member(
                bkey, n1, n2, material=ANSI_4130(), section=self.beam
            )

            # if n1 not in constrained:
            #     print(f'releasing {bkey}')
            #     self.st.frame.DefineReleases(bkey, Rzi=True)

            # if n2 not in constrained:
            #     print(f'releasing {bkey} J')
            #     self.st.frame.DefineReleases(bkey, Rzj=True)

        self.st.add_constraint(
            "A",
            SupportDX=True,
            SupportDY=True,
            SupportDZ=True,
            SupportRY=False,
            SupportRX=False,
            SupportRZ=True,
        )
        self.st.add_constraint(
            "E",
            SupportDX=False,
            SupportDY=True,
            SupportDZ=True,
            SupportRY=False,
            SupportRX=False,
            SupportRZ=False,
        )
        # for node in self.st.nodes:
        #     self.st.frame.DefineSupport(node,SupportDZ=True,SupportRZ=True)

        self.st.frame.AddNodeLoad("F", "FY", -1000)
        self.st.frame.AddNodeLoad("G", "FY", -2000)

        self.st.analyze(check_statics=True)

        print(self.st.node_dataframes)

    def test_reactions(self):
        df = self.st.node_dataframes["Combo 1"]

        dfa = df[df["name"] == "A"]
        dfe = df[df["name"] == "E"]

        self.subtest_assert_near(float(dfa["rxfy"]), 1667)
        self.subtest_assert_near(float(dfe["rxfy"]), 1333)

        self.subtest_member("A", "B", "max_axial", 1925)
        self.subtest_member("A", "G", "max_axial", -926)
        self.subtest_member("B", "C", "max_axial", 1925)
        self.subtest_member("B", "G", "max_axial", -1925)
        self.subtest_member("C", "D", "max_axial", 1541)
        self.subtest_member("F", "G", "max_axial", -1734)
        self.subtest_member("C", "F", "max_axial", 382)
        self.subtest_member("C", "G", "max_axial", -382)
        self.subtest_member("C", "G", "max_axial", 1541)
        self.subtest_member("D", "F", "max_axial", -1541)
        self.subtest_member("E", "F", "max_axial", -770)

        # Visualization.RenderModel( self.st.frame )

    def subtest_member(self, nodea, nodeb, result_key, truth, pct=0.025):
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
            print(f"fails {key} {result_key}| {value:3.5f} == {truth:3.5f}?")
        self.subtest_assert_near(value, truth, pct=pct)

    def subtest_assert_near(self, value, truth, pct=0.025):
        with self.subTest():
            self.assertAlmostEqual(value, truth, delta=abs(truth * pct))

    def get_member_result(self, nodea, nodeb, result_key):
        key_1 = f"{nodea}_{nodeb}"
        key_2 = f"{nodeb}_{nodea}"

        if key_1 in self.st.beams:
            mem = self.st.beams[key_1]
            if result_key in mem.results:
                return float(mem.results[result_key])

        elif key_2 in self.st.beams:
            mem = self.st.beams[key_2]
            if result_key in mem.results:
                return float(mem.results[result_key])

        return numpy.nan  # shouod fail, nan is not comparable
