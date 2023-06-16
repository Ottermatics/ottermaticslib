from ottermatics.eng.pipes import *
from ottermatics.eng.fluid_material import *


n1 = FlowInput(0,0,0,flow_in=-0.1)
n2 = PipeNode(10,0,0)
n3 = PipeNode(0,10,0)
n4 = PipeNode(10,10,0)
n5 = FlowInput(20,10,0,flow_in=0.1)

n2z = PipeNode(10,0,5,name='n2z')
n3z = PipeNode(0,10,5,name='n3z')
n4z = PipeNode(10,10,5,name='n4z')
n4zo = FlowInput(0,0,0,flow_in=-0.5)

n2za = PipeNode(10,0,10)
n3za = PipeNode(0,10,10)
n4za = PipeNode(10,10,10)
n2zoa = FlowInput(10,5,15,flow_in=0.6)

pipe1 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n1,node_e=n2)
pipe2 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n1,node_e=n3)
pipe3 = Pipe(v=random.random(),D=0.1,node_s=n2,node_e=n4)
pipe4 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n3,node_e=n4)
pipe5 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n4,node_e=n5)

pipe5 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n2z,node_e=n2)
pipe6 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n3,node_e=n3z)
pipe7 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n4,node_e=n4z)
pipe8 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n2,node_e=n4z)
pipe9 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n4zo,node_e=n4z)
pipe10 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n3z,node_e=n2z)

# pipe10 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n3za,node_e=n4)
# pipe11 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n2za,node_e=n2zoa)
# pipe12 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n4za,node_e=n3z)
# pipe13 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n3,node_e=n2za)
# pipe13 = Pipe(v=random.random(),D=0.1*random.random(),node_s=n1,node_e=n4z)


ps = PipeSystem(in_node=n1)
ps.draw()

m