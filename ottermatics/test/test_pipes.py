from ottermatics.eng.pipes import *
from ottermatics.eng.fluid_material import *

import unittest


class TestPipes(unittest.TestCase):
    def test_pipe_analysis(self):
        n1 = FlowInput(x=0, y=0, z=0, flow_in=-0.1)
        n2 = PipeNode(x=10, y=0, z=0)
        n3 = PipeNode(x=0, y=10, z=0)
        n4 = PipeNode(x=10, y=10, z=0)
        n5 = FlowInput(x=20, y=10, z=0, flow_in=0.1)

        n2z = PipeNode(x=10, y=0, z=5, name="n2z")
        n3z = PipeNode(x=0, y=10, z=5, name="n3z")
        n4z = PipeNode(x=10, y=10, z=5, name="n4z")
        n4zo = FlowInput(x=0, y=0, z=0, flow_in=-0.5)

        n2za = PipeNode(x=10, y=0, z=10)
        n3za = PipeNode(x=0, y=10, z=10)
        n4za = PipeNode(x=10, y=10, z=10)
        n2zoa = FlowInput(x=10, y=5, z=15, flow_in=0.6)

        pipe1 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n1, node_e=n2
        )
        pipe2 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n1, node_e=n3
        )
        pipe3 = Pipe(v=random.random(), D=0.1, node_s=n2, node_e=n4)
        pipe4 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n3, node_e=n4
        )
        pipe5 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n4, node_e=n5
        )

        pipe5 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n2z, node_e=n2
        )
        pipe6 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n3, node_e=n3z
        )
        pipe7 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n4, node_e=n4z
        )
        pipe8 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n2, node_e=n4z
        )
        pipe9 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n4zo, node_e=n4z
        )
        pipe10 = Pipe(
            v=random.random(), D=0.1 * random.random(), node_s=n3z, node_e=n2z
        )

        # pipe10 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n3za,node_e=n4)
        # pipe11 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n2za,node_e=n2zoa)
        # pipe12 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n4za,node_e=n3z)
        # pipe13 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n3,node_e=n2za)
        # pipe13 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n1,node_e=n4z)

        ps = PipeSystem(in_node=n1)
        ps.draw()
