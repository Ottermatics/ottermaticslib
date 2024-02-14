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

from engforge.components import Component
from engforge.configuration import forge
from engforge.tabulation import (
    system_property,
    NUMERIC_VALIDATOR,
    STR_VALIDATOR,
    Ref,
)
from engforge.eng.fluid_material import FluidMaterial
from engforge.common import G_grav_constant
from engforge.slots import *
from engforge.signals import *
from engforge.logging import LoggingMixin
from engforge.system import System
from engforge.properties import *


import networkx as nx

import attr, attrs

import numpy
import fluids

import attrs


class PipeLog(LoggingMixin):
    pass


log = PipeLog()

#TODO: add compressibility effects
@forge
class PipeNode(Component):
    x: float = attrs.field()
    y: float = attrs.field()
    z: float = attrs.field()

    _segments: list  # created on init

    def __on_init__(self):
        self._segments = []

    def add_segment(self, pipe: "PipeFlow"):
        if pipe not in self.segments:
            self._segments.append(pipe)
        else:
            self.warning(f"pipe already added: {pipe}")

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


@forge
class PipeFlow(Component):
    D: float = attrs.field()
    v: float = attrs.field(default=0)
    material = SLOT.define(FluidMaterial)

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
        return max(o, 1)

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


@forge
class Pipe(PipeFlow, Component):
    node_s = SLOT.define(PipeNode, default_ok=False)
    node_e = SLOT.define(PipeNode, default_ok=False)
    roughness: float = attrs.field(default=0.0)
    bend_radius: float = attrs.field(default=None)

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


# Specalized Nodes
class FlowNode(PipeNode):
    """Base For Boundary Condition Nodes of"""

    @system_property
    def dP_f(self) -> float:
        return 0.0

    @system_property
    def dP_p(self) -> float:
        return 0.0

    @system_property
    def dP_tot(self) -> float:
        return 0.0


@forge
class PipeFitting(FlowNode, PipeFlow):
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
#https://neutrium.net/fluid-flow/pressure-loss-from-fittings-excess-head-k-method/
# https://neutrium.net/fluid-flow/discharge-coefficient-for-nozzles-and-orifices/
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
@forge
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


@forge
class Pump(Component):
    """Simulates a pump with power input, max flow, and max pressure by assuming a flow characteristic"""

    max_flow: float = attrs.field()
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


@forge
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
        for i, cycle in enumerate(nx.cycle_basis(self.graph)):
            cycle_attr_name = f"_cycle_redisual_{i}"
            pipes = []
            self.info(f"found cycle: {cycle}")
            for cs, cl in zip(cycle, cycle[1:] + [cycle[0]]):
                pipe = self.graph.get_edge_data(cs, cl)["pipe"]
                mult = 1
                if cs == pipe.node_e.system_id:
                    # reverse
                    mult = -1
                pipes.append((pipe, mult))

            def res_func():
                out = 0
                for pipe, mult in pipes:
                    out += mult * pipe.dP_tot
                return out / 1000.0

            setattr(self, cycle_attr_name, res_func)
            self.flow_solvers[cycle_attr_name] = Ref(self, cycle_attr_name)

        bf = lambda kv: len(kv[1].segments)
        for nid, node in sorted(self.nodes.items(), key=bf):
            if len(node.segments) > 1:
                self.flow_solvers[nid] = Ref(node, "sum_of_flows")
            elif not isinstance(node, FlowInput):
                self.info(f"deadend: {node.identity}")
                # self.flow_solvers[nid] = Ref(node,'sum_of_flows')

        for nid, node in self.nodes.items():
            if isinstance(node, FlowInput):
                self.flow_solvers[nid] = Ref(node, "sum_of_flows")

        for pid, pipe in self.pipes.items():
            self.pipe_flow[pid] = Ref(pipe, "v")

    @property
    def _X(self):
        return {k: v for k, v in self.pipe_flow.items()}

    @property
    def _F(self):
        return {k: v for k, v in self.flow_solvers.items()}

    @instance_cached
    def F_keyword_order(self):
        """defines the order of inputs in ordered mode for calcF"""
        return {i: k for i, k in enumerate(self.flow_solvers)}

    @property
    def nodes(self):
        return {k: v for k, v in self.items.items() if isinstance(v, PipeNode)}

    @property
    def pipes(self):
        return {k: v for k, v in self.items.items() if isinstance(v, Pipe)}

    # Pipe System Composition
    def add_to_graph(self, graph, node_or_pipe) -> str:
        """recursively add node or pipe elements"""

        idd = node_or_pipe.system_id

        # Early Exit
        if idd in self.items:
            return idd

        elif graph.has_node(idd):
            return idd

        log.info(f"adding {idd}")

        if isinstance(node_or_pipe, PipeNode):
            if not graph.has_node(idd):
                graph.add_node(idd, sys_id=idd, node=node_or_pipe)
                self.items[idd] = node_or_pipe

            for seg in node_or_pipe.segments:
                self.add_to_graph(graph, seg)

        elif isinstance(node_or_pipe, Pipe):
            nodes = node_or_pipe.node_s
            nodee = node_or_pipe.node_e

            if not graph.has_node(nodes.system_id):
                nsid = self.add_to_graph(graph, nodes)
            else:
                nsid = nodes.system_id

            if not graph.has_node(nodee.system_id):
                neid = self.add_to_graph(graph, nodee)
            else:
                neid = nodee.system_id

            if not graph.has_edge(nsid, neid):
                graph.add_edge(nsid, neid, sys_id=idd, pipe=node_or_pipe)
                self.items[idd] = node_or_pipe

        else:
            raise ValueError(f"not a node or pipe: {node_or_pipe}")

        return idd

    def create_graph_from_pipe_or_node(self, node_or_pipe) -> nx.Graph:
        """Creates a networkx graph from a pipe or node"""

        self.graph = nx.Graph()

        if isinstance(node_or_pipe, PipeNode):
            self.add_to_graph(self.graph, node_or_pipe)

        elif isinstance(node_or_pipe, Pipe):
            self.add_to_graph(self.graph, node_or_pipe)

        else:
            raise ValueError(f"not a node or pipe: {node_or_pipe}")

        return self.graph

    def draw(self):
        try:
            from pyvis.network import Network

            net = Network(directed=False)

            net.from_nx(self.graph)
            net.show("process_graph.html")

        except:
            pos = pos = nx.spring_layout(self.graph)
            nx.draw(self.graph, pos=pos)
            labels = nx.draw_networkx_labels(self.graph, pos=pos)




#TODO: update compressibility:
# 
# import control as ct
# import numpy as np
# from engforge.components import Component, forge
# from matplotlib.pylab import *
# 
# DT_DFLT = 1E-3
# @forge(auto_attribs=True)
# class Accumulator(Component):
# 
#     Rt: float = 5000
#     Pref: float = 1E6
#     Lref: float = 5
#     Aact: float = 5
#     Cact: float = 0.1
#     Mact: float = 0.1
#     Pmax: float = 7E8
#     tlast: float = 0
#     Kwall: float = 1
# 
#     lim_marg: float = 0.01
#     min_dt: float = DT_DFLT
# 
#     model: 'InputOutputSystem' = None
# 
#     discrete: bool = True
#     pressure_mode: bool = False #false is flow based
# 
#     def __on_init__(self):
#         #handle defined pressure mode
#         if self.pressure_mode:
#             if not self.discrete:
#                 self.model =  ct.NonlinearIOSystem(self.pr_update_accumulator,self.pr_accumulator_output,inputs=('Pi',),outputs=('Pc','Qc','Pout'),states=('x','v'),name= self.name if self.name else 'accumul    ator')
#             else:
#                 self.model =  ct.NonlinearIOSystem(self.pr_discrete_update_accumulator,self.pr_accumulator_output,inputs=('Pi',),outputs=('Pc','Qc','Pout'),states=('x','v'),name= self.name if self.name else 'accumulator',dt=True)
#         else:
#             if not self.discrete:
#                 self.model =  ct.NonlinearIOSystem(self.w_update_accumulator,self.w_output,inputs=('Qc',),outputs=('Pc','Pout'),states=('x'),name= self.name if self.name else 'accumulator')
#             else:
#                 self.model =  ct.NonlinearIOSystem(self.w_discrete_update_accumulator, self.w_output,inputs=('Qc',),outputs=('Pc','Pout'),states=('x'),name= self.name if self.name else 'accumulator',dt=True)                
#                 
#     def w_update_accumulator(self,t,x,u,params):
#         Rt = params.get('Rt',self.Rt) #Pa/(m3/s)
#         Pref = params.get('Pref',self.Pref)
#         Lref = params.get('Lref',self.Lref)
#         Aact = params.get('Aact',self.Aact)
#         Cact = params.get('Cact',self.Cact)
#         Mact = params.get('Mact',self.Mact)
#         Kwall = params.get('Kwall',self.Kwall)
#         Pmax = params.get('Pmax',self.Pmax)
# 
#         qin  = u[0]
#         
#         xact   = x[0]
#         #Pi  = x[0]
#         
#         if self.discrete and isinstance(self.model.dt,float):
#             self.dt = dt = self.model.dt
# 
#         else: #infer dt
#             if t >= self.tlast+self.min_dt:
#                 self.dt = dt = max(abs(t - self.tlast),self.min_dt)
#                 self.tlast = t
#             else:
#                 self.dt = dt = getattr(self,'dt',self.min_dt)
# 
#         #determine rate of x change (positive compresses accumulator (x->0))
#         xdot = -qin/Aact
# 
#         xmin = Pref*Lref/Pmax
# 
#         #limit xdot to prevent over/under shoot
#         if xact + dt * xdot > Lref*(1-self.lim_marg):
#             xdot = (Lref-xact)/dt
#         elif xact + dt * xdot < xmin:
#             xdot = (xmin-xact)/dt
# 
#         return xdot
#     
#     def w_discrete_update_accumulator(self,t,x,u,params):
#         v = self.w_update_accumulator(t,x,u,params)
#         dt = self.model.dt
# 
#         if self.discrete and isinstance(self.model.dt,float):
#             dt = self.model.dt
#         else:
#             dt = self.dt
# 
#         xnew= x[0] + v*dt
# 
#         return xnew
#     
#     def w_output(self,t,x,u,params):
#         Rt = params.get('Rt',self.Rt) #Pa/(m3/s)
#         Pref = params.get('Pref',self.Pref)
#         Lref = params.get('Lref',self.Lref)
#         Aact = params.get('Aact',self.Aact)
#         Cact = params.get('Cact',self.Cact)
#         Mact = params.get('Mact',self.Mact)
#         Kwall = params.get('Kwall',self.Kwall)
#         Pmax = params.get('Pmax',self.Pmax)
# 
#         qin  = u[0]
#         
#         xact   = x[0]   
# 
#         xmin = Pref*Lref/Pmax
# 
#         Pcomp = Pref*Lref / max(xmin,xact)             
# 
#         #junction pressure is higher than pcomp when accumulator is compressed
#         Pi = Pcomp + qin*Rt
# 
#         return [Pcomp,Pi]
#         
# 
#                 
# 
#     def pr_update_accumulator(self,t,x,u,params):
#         """takes the current accumulator position, external pressure, and accumulator constants to determine the rate of change of accumulator position"""
#         Rt = params.get('Rt',self.Rt) #Pa/(m3/s)
#         Pref = params.get('Pref',self.Pref)
#         Lref = params.get('Lref',self.Lref)
#         Aact = params.get('Aact',self.Aact)
#         Cact = params.get('Cact',self.Cact)
#         Mact = params.get('Mact',self.Mact)
#         Kwall = params.get('Kwall',self.Kwall)
#         Pmax = params.get('Pmax',self.Pmax)
# 
#         Pi = u[0]
#         #print(x,u)
#         xact = x[0]
#         v    = x[1]
# 
#         if self.discrete and isinstance(self.model.dt,float):
#             self.dt = dt = self.model.dt
# 
#         else: #infer dt
#             if t >= self.tlast+self.min_dt:
#                 self.dt = dt = max(abs(t - self.tlast),self.min_dt)
#                 self.tlast = t
#             else:
#                 self.dt = dt = getattr(self,'dt',self.min_dt)
# 
#         xmin = Pref*Lref/Pmax
# 
#         Pcomp = Pref*Lref / max(max(xmin,xact),Lref)
#         dP = Pcomp-Pi
#         sgn = np.sign(dP)
# 
#         #FIXME: remove since dP is based on equillibritum
#         #xeq = max(min(Pref*Lref/Pi,Lref),xmin)
#         #K = 0*self.Kwall*(xeq-xact)/Lref
#         K = 0
# 
#         xdot =  sgn * ( abs(dP) / ( Rt * Aact**2.))**0.5/Lref
#         dvdt = ((xdot-v) - K)/Mact - Cact*Aact*v*abs(v)/Mact
# 
#         x_proj = v * dt + xact
# 
#         #Consider projected limits
#         if x_proj >= Lref*(1-self.lim_marg):
#             vin = v
#             v = (Lref-xact)/dt#,-1*Lref/10/dt)
#             dvdt = (v-vin)/dt
# 
#         elif x_proj <= xmin: 
#             vin = v
#             v = (0 - xact)/dt
#             dvdt = (v-vin)/dt
# 
#         return v,dvdt
# 
#     def pr_discrete_update_accumulator(self,t,x,u,params):
#         v,dvdt = self.pr_update_accumulator(t,x,u,params)
#         dt = self.model.dt
# 
#         if self.discrete and isinstance(self.model.dt,float):
#             dt = self.model.dt
#         else:
#             dt = self.dt
# 
#         xnew= x[0] + v*dt
#         vnew = x[1] + dvdt*dt
# 
#         return xnew,vnew
# 
#     def pr_accumulator_output(self,t,x,u,params):
#         """returns accumulator pressure and flow (in positive)"""    
#         Rt = params.get('Rt',self.Rt) #Pa/(m3/s)
#         Pref = params.get('Pref',self.Pref)
#         Lref = params.get('Lref',self.Lref)
#         Aact = params.get('Aact',self.Aact)
#         Cact = params.get('Cact',self.Cact)
#         Mact = params.get('Mact',self.Mact)
#         Pmax = params.get('Pmax',self.Pmax)
#         #dt = params.get('dt',self.dt)
# 
#         Pi = u[0]
#         xact = x[0]
#         v    = x[1]
# 
#         xmin = Pref*Lref/Pmax
#         Pcomp = Pref*Lref / max(xmin,xact)
#         Q = v*Aact
#         return [Pcomp,Q,Pi]
#     
# @forge(auto_attribs=True)
# class Motor(Component):
#     """Nonlinear motor with parasitics with fixed displacement"""
# 
#     I: float = 1
#     D: float = 1E-6
#     Cd: float = 0.0
#     Ckloss: float = 0
#     f: float = 0.0
#     N: float = 1000
# 
#     min_dt: float = DT_DFLT
#     model: 'InputOutputSystem' = None
#     discrete: bool = True
#     tlast = 0   
# 
#     def __on_init__(self):
#         #print(self.name)
#         if not self.discrete:
#             self.model = ct.NonlinearIOSystem(self.update_motor, self.motor_output, inputs=('P1','P2'), outputs=('Trq','Q','Pwr','Pin','Pout'), state=('w',), name=self.name if self.name else 'motor')         
#         else:
#             self.model = ct.NonlinearIOSystem(self.discrete_update_motor, self.motor_output, inputs=('P1','P2'), outputs=('Trq','Q','Pwr','Pin','Pout'), state=('w',), name=self.name if self.name else 'motor',dt=True) 
#             self.tlast = 0   
# 
#     def update_motor(self,t,x,u,params):
#         """updates the motor acceleration based on pressure differential and nonlinear parasitics"""
#         I = params.get('I',self.I)
#         D = params.get('D',self.D)
#         Cd = params.get('Cd',self.Cd)
#         Ckloss = params.get('Ckloss',self.Ckloss)
#         f = params.get('f',self.f)
#         N = params.get('N',self.N)
#         
#         P1 = u[0]
#         P2 = u[1]
#         w = x[0]
# 
#         Q = D*(w/(2*3.14159))
#         dPq = np.sign(Q)*Ckloss*Q**2
#         
#         dw_torque = (D/(2*np.pi))* ((P1 - P2)-dPq)
#         dw_friction = w*f*N
#         dw_drag = w*abs(w)*Cd
#         
#         return (dw_torque - dw_friction - dw_drag)/I
#     
#     def discrete_update_motor(self,t,x,u,params):
#         dw = self.update_motor(t,x,u,params)
# 
# 
#         if self.discrete and isinstance(self.model.dt,float):
#             self.dt = dt = self.model.dt
# 
#         else: #infer dt
#             if t >= self.tlast+self.min_dt:
#                 self.dt = dt = max(abs(t - self.tlast),self.min_dt)
#                 self.tlast = t
#             else:
#                 self.dt = dt = getattr(self,'dt',self.min_dt)
# 
#         return x[0] + dw*self.dt
#         
#     def motor_output(self,t,x,u,params):
#         """calculates torque, flow and power output"""
#         I = params.get('I',self.I)
#         D = params.get('D',self.D)
#         Cd = params.get('Cd',self.Cd)
#         Ckloss = params.get('Ckloss',self.Ckloss)
#         f = params.get('f',self.f)
#         N = params.get('N',self.N)
#         
#         P1 = u[0]
#         P2 = u[1]
#         w = x[0]
#         
#         Q = (w/2*3.14159)*D
#         dPq = np.sign(Q)*Ckloss*Q**2
# 
#         dP = (P1 - P2) - dPq
# 
#         Trq = D * dP / (2*3.14159)
#         Pwr = dP * Q
#         return [Trq, Q, Pwr,P1,P2]   
# 
# 
# @forge(auto_attribs=True)
# class Actuator(Component):
#     Ap: float = 1
#     mp: float = 0.5
#     bp: float = 0.1
#     Kwall: float = 1000
#     Lstroke: float = 1   
#     
#     min_dt = DT_DFLT
#     lim_marg = 0.01
#     
#     lim = None # true if high, false if low and none otherwise
# 
#     model: 'InterConnectedSystem' = None
# 
#     discrete: bool = True
# 
#     tlast = 0
# 
#     def __on_init__(self):
#         if self.discrete:
#             self.model = ct.NonlinearIOSystem(self.discrete_update_fnc,self.output_fnc,state=('x','v'),outputs=('Q1','Q2','P1','P2'),inputs=('F','P1','P2'),dt=True,name=self.name if self.name else 'actuator')
# 
#         else:
#             self.model = ct.NonlinearIOSystem(self.update_fnc,self.output_fnc,state=('x','v'),outputs=('Q1','Q2','P1','P2'),inputs=('F','P1','P2'),name=self.name if self.name else 'actuator')
# 
#             
# 
#     def update_fnc(self,t,x,u,params):
#         Ap = params.get('Ap',self.Ap)
#         mp = params.get('mp',self.mp)
#         bp = params.get('bp',self.bp)
#         Kwall = params.get('Kwall',self.Kwall)
#         Lstroke = params.get('Lstroke',self.Lstroke)
# 
#         xpos = x[0]
#         v = x[1]
# 
#         f = u[0]
#         p1 = u[1]
#         p2 = u[2]
#         
#         if self.discrete and isinstance(self.model.dt,float):
#             dt = self.model.dt
#             self.dt = dt
# 
#         else: #infer dt
#             if t >= self.tlast+self.min_dt:
#                 self.dt = dt = max(abs(t - self.tlast),self.min_dt)
#                 self.tlast = t
#             else:
#                 self.dt = dt = getattr(self,'dt',self.min_dt)
# 
#         if xpos > Lstroke:
#             #print('kpos')
#             K = self.Kwall*(xpos-Lstroke)
#         elif xpos < 0:
#             #print('kneg')
#             K = self.Kwall*(xpos - 0)
#         else:
#             K = 0
#         
#         daf = f - (p1-p2)*Ap
#         dad = bp*v*abs(v)
#         dvdt = (daf - dad - K*daf)/mp    
# 
#         x_proj = v * dt + xpos
#         
#         if x_proj >= Lstroke*(1-self.lim_marg):
#             vin = v
#             v = (Lstroke-xpos)/dt
#             dvdt = min((v-vin)/dt,dvdt)
#     
#         elif x_proj <= self.lim_marg: 
#             vin = v
#             v = (0 - xpos)/dt
#             dvdt = max((v-vin)/dt,dvdt)
# 
# 
#         return [v,dvdt]
#     
#     def discrete_update_fnc(self,t,x,u,params):
#         v,dvdt = self.update_fnc(t,x,u,params)
#         if self.discrete and isinstance(self.model.dt,float):
#             dt = self.model.dt
#         else:
#             dt = self.dt
# 
#         if hasattr(self,'last_a'):
#             dvdt = (self.last_a + dvdt)/2
#             v = (self.last_v + v)/2
# 
#         vnew = v + dvdt*dt
#         newx = x[0] + v*dt
# 
#         if newx < 0:
#             newx = 0
#             if vnew < 0:
#                 vnew = 0
# 
#         elif newx > self.Lstroke:
#             newx = self.Lstroke
#             if vnew > 0:
#                 vnew = 0
# 
#         self.last_x = newx
#         self.last_v = vnew
#         self.last_a = dvdt
# 
#         return newx, vnew
# 
#     def output_fnc(self,t,x,u,params):
#         Ap = params.get('Ap',self.Ap)
# 
#         v = x[1]
# 
#         Q1 = -1*v * Ap
#         Q2 = v * Ap
# 
#         return [Q1,Q2,u[1],u[2]]
#         
# 
# @forge(auto_attribs=True)
# class Pipe(Component):
#     '''models pressure change and flow Q from nodes with pressure P1->P2'''
#     Kp:float = 1 #TODO: better friction model
#     
#     flow_input: bool = False
#     model: 'InputOutputSystem' = None
# 
#     def __on_init__(self):
#         if not self.flow_input:
#             self.model = ct.NonlinearIOSystem(self.pipe_flow,self.pipe_flow_output,state=('Q'),inputs=('P1','P2'),outputs=('P1o','P2o','Q'),dt=True,name=self.name if self.name else 'pipe')
#         else:
#             self.model = ct.NonlinearIOSystem(self.pipe_pressure,self.pipe_pressure_output,state=('P2'),inputs=('P1','Q'),outputs=('P1o','P2','Q'),dt=True,name=self.name if self.name else 'pipe')
# 
#     def pipe_flow(self,t,x,u,params):
#         Kp = params.get('Kp',self.Kp)
#         p1 = u[0]
#         p2 = u[1]
#         dP = (p2-p1)
#         return -1*np.sign(dP)*(abs(dP)/Kp)**0.5
#         
#     def pipe_pressure(self,t,x,u,params):
#         Kp = params.get('Kp',self.Kp)
#         p1 = u[0]
#         q = u[1]
#         return -1*np.sign(q)*Kp*q**2 + p1
#         
#     def pipe_flow_output(self,t,x,u,params):
#         return [u[0],u[1],x[0]]
#         
#     def pipe_pressure_output(self,t,x,u,params):
#         return [u[0],x[0],u[1]]
# 
# 
# @forge(auto_attribs=True)
# class Valve(Component):
#     Ao: float = 0.1 #area
#     ts: float = 30/1000. #seconds
#     rho: float = 1000
#     Cd: float = 0.1
#     discrete_control: bool = True    # contorl signal is zero or one, vs a goal
#     
#     tlast = 0
#     min_dt:float = 1E-3
# 
#     def update_alpha(self,t,x,u,params):        
#         ts = params.get('ts',self.ts)
#         uctl = u[0]
#         alpha = x[0]
#         
#         if t >= self.tlast+self.min_dt:
#             self.dt = dt = max(abs(t - self.tlast),self.min_dt)
#             self.tlast = t
#         else:
#             self.dt = dt = getattr(self,'dt',self.min_dt)
# 
#         dA = 1.0 / ts
#         
#         if self.discrete_control:
#             if uctl==1:
#                 if alpha < 1-dA*dt:
#                     return dA
#                 else:
#                     return 1-alpha
#             elif uctl==0:
#                 if alpha > dA*dt:
#                     return dA
#                 else:
#                     return 0-alpha
#         else:
#             vctl = min(max(uctl,0),1)
#             d = (uctl-alpha)/dt
#             dv = max(min(d,dA),-dA)
#             return dv
#                 
#         
#     def get_flow(self,t,x,u,params):
#         rho = params.get('rho',self.rho)
#         ts = params.get('ts',self.ts)
#         Cd = params.get('Cd',self.Cd)
#         Ao = params.get('Ao',self.Ao)
# 
#         alpha = x[0]
#         if alpha < 0.05:
#             return 0.0 #avoid divide by zero in area
#         
#         dP = u[1] 
#         Av = Ao*min(max(alpha,0),1)
#         return np.sign(dP)*Cd*Av*((2/rho)*abs(dP))**0.5
# 
# @forge(auto_attribs=True)
# class PipeJunction(Component):
#     
#     n_pipes: int = 2
#     Vi: float = 1
#     B: float = 300E3
#     
#     min_dt = DT_DFLT
#     model: 'InputOutputSystem' = None
#     discrete:bool = True
#     
#     def __on_init__(self):
#         state = ('P',)
#         inputs = tuple(f'Q{i+1}' for i in range(self.n_pipes))
#         outins = tuple(f'Qout{i+1}' for i in range(self.n_pipes))
#         outputs = tuple(list(state)+list(outins)+['dQ'])
#         if self.discrete:
#             self.model = ct.NonlinearIOSystem(self.discrete_pressure_update,self.output,inputs=inputs,outputs=outputs,state=state,dt=True,name=self.name if self.name else 'junction')
#             self.tlast = 0
#         else:
#             self.model = ct.NonlinearIOSystem(self.pressure_update,self.output,inputs=inputs,outputs=outputs,state=state,name=self.name if self.name else 'junction')
#             
#     def pressure_update(self,t,x,u,params):
#         #TODO: update compressibility from temp / pressure ect.
#         B = params.get('B',self.B)
#         Vi = params.get('Vi',self.Vi)
#         sumQ = sum(u)
#         dPdt = (B/Vi)*sumQ
#         return dPdt
#         
#     def discrete_pressure_update(self,t,x,u,params):
#         dPdt = self.pressure_update(t,x,u,params)
#         if self.discrete and isinstance(self.model.dt,float):
#             dt = self.model.dt
#             self.dt = dt
# 
#         else: #infer dt
#             if t >= self.tlast+self.min_dt:
#                 self.dt = dt = max(abs(t - self.tlast),self.min_dt)
#                 self.tlast = t
#             else:
#                 self.dt = dt = getattr(self,'dt',self.min_dt)
#         return self.dt*dPdt + x[0]
#         
#     def output(self,t,x,u,parms):
#         return tuple([x[0]]+list(u)+[sum(u)])
#       













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
