"""We'll use the QP formulation to develop a fluid analysis system for fluids
start with single phase and move to others


1) Pumps Power = C x Q x P 
2) Compressor = C x (QP) x (PR^C2 - 1)
3) Pipes = dP = C x fXL/D x V^2 / 2g
4) Pipe Fittings / Valves = dP = C x V^2 (fXL/D +K)   |   K is a constant, for valves it can be interpolated between closed and opened
5) Splitters & Joins: Handle fluid mixing
6) Phase Separation (This one is gonna be hard)
7) Heat Exchanger dP = f(V,T) - phase change issues
8) Filtration dP = k x Vxc x (Afrontal/Asurface) "linear"
"""

from ottermatics.components import Component
from ottermatics.configuration import otterize
from ottermatics.tabulation import (
    system_property,
    NUMERIC_VALIDATOR,
    STR_VALIDATOR,
    Ref
)
from ottermatics.eng.fluid_material import FluidMaterial
from ottermatics.common import G_grav_constant
from ottermatics.slots import *
from ottermatics.signals import *
from ottermatics.logging import LoggingMixin
from ottermatics.system import System
from ottermatics.properties import *


import networkx as nx

import attr, attrs

import numpy
import fluids

import attrs

class PipeLog(LoggingMixin):
    pass
log = PipeLog()


@otterize
class PipeNode(Component):
    x: float = attrs.field()
    y: float = attrs.field()
    z: float = attrs.field()

    _segments: list #created on init

    def __on_init__(self):
        self._segments = []

    def add_segment(self,pipe:'PipeFlow'):
        if pipe not in self.segments:
            self._segments.append(pipe)
        else:
            self.warning(f'pipe already added: {pipe}')

    @property
    def segments(self):
        return self._segments

    @system_property
    def sum_of_flows(self) -> float:
        out = 0
        for pipe_seg in self._segments:
            if self is pipe_seg.node_s:
                out -= pipe_seg.Q
            elif self is pipe_seg.node_e:
                out += pipe_seg.Q
        return out





@otterize
class PipeFlow(Component):
    D: float = attrs.field()
    v: float = attrs.field(default=0)
    material= SLOT.define(FluidMaterial)

    def set_flow(self, flow):
        v = flow / self.A
        self.v = v

    # Geometric Propreties
    @system_property
    def A(self) -> float:
        """:returns: the cross sectional area of pipe in [m2]"""
        return 3.1415 * (self.D / 2.0) ** 2.0

    @system_property
    def C(self) -> float:
        """:returns: the sectional circmerence of pipe"""
        return 3.1415 * self.D

    # Fluid Propreties
    @system_property
    def Q(self) -> float:
        """:returns: the volumetric flow through this pipe in [m3/s]"""
        return self.A * self.v

    @system_property
    def Mf(self) -> float:
        """:returns: the massflow through this pipe in [kg/s]"""
        return self.density * self.Q

    @system_property
    def reynoldsNumber(self) -> float:
        """:returns: the flow reynolds number"""
        o = abs(self.density * self.v * self.D / self.viscosity)
        return max(o,1)

    # Material Properties Exposure
    @system_property
    def density(self) -> float:
        return self.material.density

    @system_property
    def viscosity(self) -> float:
        return self.material.viscosity

    @system_property
    def enthalpy(self) -> float:
        return self.material.enthalpy

    @system_property
    def T(self) -> float:
        return self.material.T

    @T.setter
    def T(self, new_T):
        self.material.T = new_T

    @system_property
    def P(self) -> float:
        return self.material.P

    @P.setter
    def P(self, new_P):
        self.material.P = new_P

    @property
    def dP_f(self):
        """The loss of pressure in the pipe due to pressure"""
        raise NotImplemented()

    @property
    def dP_p(self):
        """The loss of pressure in the pipe due to potential"""
        raise NotImplemented()

    @property
    def dP_tot(self):
        raise NotImplemented()

    @property
    def Fvec(self):
        """returns the fluidized vector for 1D CFD for continuity:0, and momentum:1"""
        # TODO: add energy eq
        # TODO: 1D CFD Pipe Network Solver :)
        return [
            self.density * self.v,
            self.density * self.v**2.0 + self.P,
        ]  # TODO: include Txx


@otterize
class Pipe(PipeFlow, Component):
    node_s = SLOT.define(PipeNode,default_ok=False)
    node_e = SLOT.define(PipeNode,default_ok=False)
    roughness:float = attrs.field(default=0.0)
    bend_radius:float = attrs.field(default=None)

    straight_method = "Clamond"
    laminar_method = "Schmidt laminar"
    turbulent_method = "Schmidt turbulent"

    def __on_init__(self):
        self.node_s.add_segment(self)
        self.node_e.add_segment(self)

    @system_property
    def Lx(self) -> float:
        return self.node_e.x - self.node_s.x

    @system_property
    def Ly(self) -> float:
        return self.node_e.y - self.node_s.y

    @system_property
    def Lz(self) -> float:
        return self.node_e.z - self.node_s.z

    @system_property
    def Lhz(self) -> float:
        """:returns: The length of pipe element in the XY plane"""
        return numpy.sqrt(self.Lx**2.0 + self.Ly**2.0)

    @system_property
    def L(self) -> float:
        """:returns: The absolute length of pipe element"""
        return numpy.sqrt(self.Lx**2.0 + self.Ly**2.0 + self.Lz**2.0)

    @system_property
    def inclination(self) -> float:
        """:returns: the inclination angle in degrees"""
        return numpy.rad2deg(numpy.arctan2(self.Lz, self.Lhz))

    @system_property
    def friction_factor(self) -> float:
        """The friction factor considering bend radius"""
        if self.bend_radius is None:
            re = self.reynoldsNumber
            return fluids.friction.friction_factor(
                re, self.roughness, Method=self.straight_method
            )
        else:
            re = self.reynoldsNumber
            dc = self.D * self.bend_radius
            return fluids.friction.friction_factor_curved(
                re,
                self.D,
                dc,
                self.roughness,
                laminar_method=self.laminar_method,
                turbulent_method=self.turbulent_method,
            )

    @system_property
    def Kpipe(self) -> float:
        """The loss coeffient of this pipe section"""
        return self.friction_factor * self.L / self.D

    @system_property
    def dP_f(self) -> float:
        """The loss of pressure in the pipe due to pressure"""
        return self.sign * self.density * self.v**2.0 * self.Kpipe / 2.0

    @system_property
    def dP_p(self) -> float:
        """The loss of pressure in the pipe due to potential"""
        return self.density * self.Lz * 9.81

    @system_property
    def dP_tot(self) -> float:
        return self.dP_f + self.dP_p
    
    @system_property
    def sign(self) -> int:
        return numpy.sign(self.v)


#Specalized Nodes
class FlowNode(PipeNode):
    """Base For Boundary Condition Nodes of """

    @system_property
    def dP_f(self) -> float:
        return 0.0

    @system_property
    def dP_p(self) -> float:
        return 0.0
    
    @system_property
    def dP_tot(self) -> float:
        return 0.0 
    
@otterize
class PipeFitting(FlowNode,PipeFlow):
    Kfitting = attr.ib(default=0.1, type=float)

    @system_property
    def dP_f(self) -> float:
        """The loss of pressure in the pipe due to pressure"""
        return self.density * self.v**2.0 * self.Kfitting / 2.0

    @system_property
    def dP_p(self) -> float:
        """The loss of pressure in the pipe due to potential"""
        return 0.0

    @system_property
    def dP_tot(self) -> float:
        return self.dP_f + self.dP_p



# TODO: Add in fitting numbers:
"""                          Fitting                      Types       K
0                       45° Elbow         Standard (R/D = 1)    0.35
1                       45° Elbow    Long Radius (R/D = 1.5)    0.20
2                90° Elbow Curved         Standard (R/D = 1)    0.75
3                90° Elbow Curved    Long Radius (R/D = 1.5)    0.45
4      90° Elbow Square or Mitred                        NaN    1.30
5                       180° Bend               Close Return    1.50
6                Tee, Run Through             Branch Blanked    0.40
7                   Tee, as Elbow            Entering in run    1.00
8                   Tee, as Elbow         Entering in branch    1.00
9             Tee, Branching Flow                        NaN    1.00
10                       Coupling                        NaN    0.04
11                          Union                        NaN    0.04
12                     Gate valve                 Fully Open    0.17
13                     Gate valve                   3/4 Open    0.90
14                     Gate valve                   1/2 Open    4.50
15                     Gate valve                   1/4 Open   24.00
16                Diaphragm valve                 Fully Open    2.30
17                Diaphragm valve                   3/4 Open    2.60
18                Diaphragm valve                   1/2 Open    4.30
19                Diaphragm valve                   1/4 Open   21.00
20        Globe valve, Bevel Seat                 Fully Open    6.00
21        Globe valve, Bevel Seat                   1/2 Open    9.50
22  Globe Valve, Composition seat                 Fully Open    6.00
23  Globe Valve, Composition seat                   1/2 Open    8.50
24                      Plug disk                 Fully Open    9.00
25                      Plug disk                   3/4 Open   13.00
26                      Plug disk                   1/2 Open   36.00
27                      Plug disk                   1/4 Open  112.00
28                    Angle valve                 Fully Open    2.00
29       Y valve or blowoff valve                 Fully Open    3.00
30                      Plug cock                \theta = 5°    0.05
31                      Plug cock               \theta = 10°    0.29
32                      Plug cock               \theta = 20°    1.56
33                      Plug cock               \theta = 40°   17.30
34                      Plug cock               \theta = 60°  206.00
35                Butterfly valve                \theta = 5°    0.24
36                Butterfly valve               \theta = 10°    0.52
37                Butterfly valve               \theta = 20°    1.54
38                Butterfly valve               \theta = 40°   10.80
39                Butterfly valve               \theta = 60°  118.00
40                    Check valve                      Swing    2.00
41                    Check valve                       Disk   10.00
42                    Check valve                       Ball   70.00
43                     Foot valve                        NaN   15.00
44                    Water meter                       Disk    7.00
45                    Water meter                     Piston   15.00
46                    Water meter  Rotary (star-shaped disk)   10.00
47                    Water meter              Turbine-wheel    6.00""" 

# 
@otterize
class FlowInput(FlowNode):
    flow_in: float = attrs.field(default=0.0)

    @system_property
    def sum_of_flows(self) -> float:
        out = self.flow_in
        for pipe_seg in self._segments:
            if self is pipe_seg.node_s:
                out -= pipe_seg.Q
            elif self is pipe_seg.node_e:
                out += pipe_seg.Q
        return out
# 
# class PressureInput(FlowNode):
#     pressure_in: float = attrs.field()
# 
# class PressureOut(FlowNode):
#     pressure_out: float = attrs.field()









@otterize
class Pump(Component):
    """Simulates a pump with power input, max flow, and max pressure by assuming a flow characteristic"""

    max_flow: float  = attrs.field()
    max_pressure: float = attrs.field()
    # throttle: float

    @property
    def design_flow_curve(self):
        """:returns: a tuple output of flow vector, and pressure vector"""
        flow = numpy.linspace(0, self.max_flow)
        return flow, self.max_pressure * (1 - (flow / self.max_flow) ** 2.0)

    def dPressure(self, current_flow):
        """The pressure the pump generates"""
        flow, dP = self.design_flow_curve
        assert current_flow >= 0, "Flow must be positive"
        assert current_flow <= self.max_flow, "Flow must be less than max flow"
        return numpy.interp(current_flow, flow, dP)

    def power(self, current_flow):
        """The power used considering in watts"""
        return self.dPressure(current_flow) * current_flow



@otterize
class PipeSystem(System):

    in_node = SLOT.define(PipeNode) 
    graph: nx.Graph
    items: dict

    def __on_init__(self):
        self.items = {}
        self.flow_solvers = {}
        self.pipe_flow = {}
        self.create_graph_from_pipe_or_node(self.in_node)
        self.assemble_solvers()

    def assemble_solvers(self):

        for i,cycle in enumerate(nx.cycle_basis(self.graph)):
            cycle_attr_name = f'_cycle_redisual_{i}'
            pipes = []
            self.info(f'found cycle: {cycle}')
            for cs,cl in zip(cycle, cycle[1:]+[cycle[0]]):
                
                pipe = self.graph.get_edge_data(cs,cl)['pipe']
                mult = 1
                if cs == pipe.node_e.system_id:
                    #reverse
                    mult = -1
                pipes.append((pipe,mult))
                
            def res_func():
                out = 0
                for pipe,mult in pipes:
                    out += mult*pipe.dP_tot
                return out/1000.
            
            setattr(self,cycle_attr_name,res_func)
            self.flow_solvers[cycle_attr_name] = Ref(self,cycle_attr_name)

        bf = lambda kv: len(kv[1].segments)
        for nid,node in sorted(self.nodes.items(),key=bf):
            if len(node.segments) > 1:
                self.flow_solvers[nid] = Ref(node,'sum_of_flows')
            elif not isinstance(node,FlowInput):
                self.info(f'deadend: {node.identity}')
                #self.flow_solvers[nid] = Ref(node,'sum_of_flows')

        for nid,node in self.nodes.items():
            if isinstance(node,FlowInput):
                self.flow_solvers[nid] = Ref(node,'sum_of_flows')
        

        for pid,pipe  in self.pipes.items():
            self.pipe_flow[pid] = Ref(pipe,'v')

            

    @property
    def _X(self):
        return {k:v for k,v in self.pipe_flow.items()}

    @property
    def _F(self):
        return {k:v for k,v in self.flow_solvers.items()}

    @instance_cached
    def F_keyword_order(self):
        """defines the order of inputs in ordered mode for calcF"""
        return {i: k for i, k in enumerate(self.flow_solvers)}


    @property
    def nodes(self):
        return {k:v for k,v in self.items.items() if isinstance(v,PipeNode)}

    @property
    def pipes(self):
        return {k:v for k,v in self.items.items() if isinstance(v,Pipe)}

    #Pipe System Composition
    def add_to_graph(self,graph,node_or_pipe)->str:
        """recursively add node or pipe elements"""

        idd = node_or_pipe.system_id

        #Early Exit
        if idd in self.items:
            return idd

        elif graph.has_node(idd):
            return idd
        
        log.info(f'adding {idd}')

        if isinstance(node_or_pipe,PipeNode):
            if not graph.has_node(idd):
                graph.add_node(idd,sys_id=idd,node=node_or_pipe)
                self.items[idd] = node_or_pipe
            
            for seg in node_or_pipe.segments:
                self.add_to_graph(graph,seg)

        elif isinstance(node_or_pipe, Pipe):
            nodes = node_or_pipe.node_s
            nodee = node_or_pipe.node_e

            if not graph.has_node(nodes.system_id):
                nsid = self.add_to_graph(graph,nodes)
            else:
                nsid = nodes.system_id

            if not graph.has_node(nodee.system_id):
                neid = self.add_to_graph(graph,nodee)
            else:
                neid = nodee.system_id
            
            if not graph.has_edge(nsid,neid):
                graph.add_edge(nsid,neid,sys_id=idd,pipe=node_or_pipe)
                self.items[idd] = node_or_pipe

        else:
            raise ValueError(f'not a node or pipe: {node_or_pipe}')
        
        return idd


    def create_graph_from_pipe_or_node(self,node_or_pipe)->nx.Graph:
        """Creates a networkx graph from a pipe or node"""
        
        self.graph = nx.Graph()

        if isinstance(node_or_pipe,PipeNode):
            self.add_to_graph(self.graph,node_or_pipe)

        elif isinstance(node_or_pipe, Pipe):
            self.add_to_graph(self.graph,node_or_pipe)

        else:
            raise ValueError(f'not a node or pipe: {node_or_pipe}')
        
        return self.graph


    def draw(self):

        try:
            from pyvis.network import Network

            net = Network(directed=False)

            net.from_nx(self.graph)
            net.show("process_graph.html")            
        
        except:
            pos = pos=nx.spring_layout(self.graph)
            nx.draw(self.graph,pos=pos)
            labels = nx.draw_networkx_labels(self.graph, pos=pos)







if __name__ == "__main__":
    N = 10

    rho = 1000.0
    f = 0.015
    L = 1.0
    Po = 1e5

    A = 1.0 * ones(N)
    u = 0.0 * ones(N)
    p = Po * ones(N)

    Pin = 1.1e5
    Uin = lambda Ps: sqrt((Pin - Ps) / (rho * 2.0))

    def F(i):
        ui = u[i]
        ps = p[i]
        F1 = rho * ui
        F2 = F1 * ui + ps
        return F1, F2

    def J(i):
        return 0.0, -rho * f

    def decodeF(F1, F2):
        up = F1 / rho
        pp = F2 - rho * up**2.0
        return up, pp

    plast = p.copy()
    ulast = u.copy()
    it = 0
    started = False
    while not started or (plast - p).mean() > 1e-3 or (ulast - u).mean() > 1e-3:
        started = True

        plast = p.copy()
        ulast = u.copy()

        for i in range(N):
            if i == 0:  # inlet
                ps = p[i]
                ui = Uin(ps)
                u[i] = ui

            # STATUS
            F1, F2 = F(i)
            J1, J2 = J(i)

            # PREDICT i+1
            dF1, dF2 = J1, J2
            Fp1, Fp2 = F1 + dF1 * L, F2 + dF2 * L

            up1, pp1 = decodeF(Fp1, Fp2)

            # CORRECTOR i+1
            Jp1, Jp2 = 0.0, -rho * f
            dpF1, dpF2 = Jp1, Jp2

            # avgGradients
            dF1n, dF2n = 0.5 * (dpF1 + dF1), 0.5 * (dpF2 + dF2)

            # Calculate i+1 actual
            Fn1, Fn2 = F1 + dF1n * L, F2 + dF2n * L

            un, pn = decodeF(Fn1, Fn2)
            if N - 1 > i:
                u[i + 1] = un
                p[i + 1] = pn

        for i in reversed(range(N)):
            # if i == 0: #inlet
            #    ps = p[i]
            #    ui = Uin(ps)
            #    u[i] = ui

            if i == N - 1:
                p[i] = Po

            # STATUS
            F1, F2 = F(i)
            J1, J2 = J(i)

            # PREDICT i+1
            dF1, dF2 = J1, J2
            Fp1, Fp2 = F1 + dF1 * L, F2 + dF2 * L

            up1, pp1 = decodeF(Fp1, Fp2)

            # CORRECTOR i+1
            Jp1, Jp2 = 0.0, -rho * f
            dpF1, dpF2 = Jp1, Jp2

            # avgGradients
            dF1n, dF2n = 0.5 * (dpF1 + dF1), 0.5 * (dpF2 + dF2)

            # Calculate i+1 actual
            Fn1, Fn2 = F1 + dF1n * L, F2 + dF2n * L

            un, pn = decodeF(Fn1, Fn2)
            if N - 1 > i:
                u[i - 1] = un
                p[i - 1] = pn

        print(u)
        print(p)
        it += 1
        if it >= 2:
            break
