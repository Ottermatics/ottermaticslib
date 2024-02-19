"""Combines the tabulation and component mixins to create a mixin for systems and components that have dynamics, such as state space models, while allowing nonlinear dynamics via matrix modification

This module is intended to work alongside the solver module and the TRANSIENT integrating attributes, and will raise an error if a conflict is detected.

The DynamicsMixin works by establishing a state matricies A, B, C, and D, which are used to define the dynamics of the system. The state matrix A is the primary matrix, and is used to define the state dynamics of the system. The input matrix B is used to define the input dynamics of the system. The output matrix C is used to define the output dynamics of the system. The feedthrough matrix D is used to define the feedthrough dynamics of the system.

As opposed to the TRANSIENT attribute, which modifies the state of the system, the DynamicsMixin copies the initial values of the state and input which then are integrated over time. At predefined intervals the control and output will stored in the tabulation.

#TODO: The top level system will collect the underlying dynamical systems and combine them to an index and overall state space model. This will allow for the creation of a system of systems, and the ability to create a system of systems with a single state space model.

#TODO: integration is done by the solver, where DynamicSystems have individual solver control, solver control is set for a smart default scipy 
"""

from engforge.configuration import Configuration, forge
from engforge.tabulation import TabulationMixin
from engforge import properties as prop
from engforge.attributes import ATTR_BASE
from engforge.properties import instance_cached

import numpy as np
import pandas
import attr, attrs


#Index maps are used to translate between different indexes (local & global)
class INDEX_MAP:
    oppo = {str:int,int:str}
    def __init__(self,data:list):
        self.data = data
        self.index = {}

    def get(self,key):
        if key not in self.index:
            self.index[key] = self.data.index(key)
        return self.index[key]

    def __getitem__(self,key):
        return self.get(key)

    def __call__(self, key):
        return self.get(key)

    @staticmethod
    def indify(arr,*args):
        return [arr[arg] if isinstance(arg,int) else arr.index(arg) for arg in args]

    def remap_indexes_to(self,new_index,*args,invert=False,old_data=None):
        if old_data is None:
            old_data = self.data
        opt1 = {arg: self.indify(old_data,arg)[0] for arg in args}
        opt2 = {arg: self.indify(old_data,val)[0] if (not isinstance(val,str)) else val for arg,val in opt1.items()}
        oop1 = {arg: self.indify(new_index,val)[0]  for arg,val in opt2.items()}
        oop2 = {arg: self.indify(new_index,val)[0] if (invert != isinstance(val,self.oppo[arg.__class__])) else val for arg,val in oop1.items()}
        return oop2
    

class DynamicsIntegratorMixin:
    
    nonlinear:bool = True #enables matrix modification for nonlinear dynamics

    def integrate(self,dt,X,U,*args,**kwargs):
        """simulate the system over the course of time.

        Args:
            dt (float): interval to integrate over in time
            X (np.ndarray): state input
            U (np.ndarray): control input
            subsystems (bool, optional): simulate subsystems. Defaults to True.

        Returns:
            dataframe: tabulated data
        """
        if self.nonlinear:
            return self.simulate_nonlinear(dt,*args,**kwargs)
        else:
            return self.simulate_linear(dt,*args,**kwargs)

    def integrate_linear(self,dt,X,U,*args,**kwargs):
        pass

    def integrate_nonlinear(self,dt,X,U,*args,**kwargs):
        pass

#Quickly create a state space model
#TODO: How to add delay, and feedback?
#TODO: How to add control and limits in general?
@forge(auto_attribs=True)
class DynamicsMixin(Configuration,DynamicsIntegratorMixin):
    """dynamic mixin for components and systems that have dynamics, such as state space models, while allowing nonlinear dynamics via matrix modification. This mixin is intended to work alongside the solver module and the TRANSIENT integrating attributes, and will raise an error if a conflict is detected #TODO.
    """

    dynamic_state_parms:list = attrs.field(factory=list)
    dynamic_input_parms:list = attrs.field(factory=list)
    dynamic_output_parms:list = attrs.field(factory=list)

    #state variables
    dynamic_A:np.ndarray = None
    dynamic_B:np.ndarray = None
    dynamic_C:np.ndarray = None
    dynamic_D:np.ndarray = None
    dynamic_F:np.ndarray = None
    dynamic_K:np.ndarray = None
    
    #Static linear state
    static_A:np.ndarray = None
    static_B:np.ndarray = None
    static_C:np.ndarray = None
    static_D:np.ndarray = None
    static_F:np.ndarray = None
    static_K:np.ndarray = None    

    #dynamic_control_module = None
    #control_interval:float = 0.0 #how often to update the physics
    update_interval:float = 0.0 #how often to update the physics, 0 is everytime
    delay_ms:float = None #delay in milliseconds for output buffering
    nonlinear:bool = False
    #TODO: add control module with PID control and pole placement design

    @instance_cached
    def state_size(self):
        return len(self.dynamic_state_parms)
    
    @instance_cached
    def input_size(self):
        return len(self.dynamic_input_parms)    
    
    @instance_cached
    def output_size(self):
        return len(self.dynamic_output_parms)

    @property
    def state(self)->np.array:
        return np.array([getattr(self,parm,np.nan) for parm in self.dynamic_state_parms])
    
    @property
    def input(self)->np.array:
        return np.array([getattr(self,parm,np.nan) for parm in self.dynamic_input_parms])      

    @property
    def output(self)->np.array:
        return np.array([getattr(self,parm,np.nan) for parm in self.dynamic_output_parms])
    
    #TODO: add sparse mode
    def create_state_matrix(self,**kwargs) -> np.ndarray:
        """creates the state matrix for the system"""
        return np.zeros((self.state_size,self.state_size))
    
    def create_state_input_matrix(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros((self.state_size,max(self.input_size,1)))
    
    def create_output_matrix(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros((max(self.output_size,1),self.state_size)) 
    
    def create_feedthrough_matrix(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros((max(self.output_size,1),max(self.input_size,1)))

    def create_state_constants(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros(self.state_size)

    def create_output_constants(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros(max(self.output_size,1))        

    def create_dynamics(self,**kw):
        """creates a dynamics object for the system"""
        #State + Control
        self.static_A = self.create_state_matrix(**kw)
        self.static_B = self.create_state_input_matrix(**kw)
        #Output
        self.static_C = self.create_output_matrix(**kw)
        self.static_D = self.create_feedthrough_matrix(**kw)
        #Constants
        self.static_F = self.create_state_constants(**kw)
        self.static_K = self.create_output_constants(**kw)

    #Nonlinear Support 
    #Override these callbacks to modify the state space model
    def update_state_nonlinear(self,t,A,X)->np.ndarray:
        """override """
        return A

    def update_input_nonlinear(self,t,B,X,U)->np.ndarray:
        """override """
        return B

    def update_output_matrix(self,t,C,X)->np.ndarray:
        """override """
        return C

    def update_feedthrough_nonlinear(self,t,D,X,U)->np.ndarray:
        """override """
        return D

    def update_state_constants(self,t,F,X)->np.ndarray:
        """override """
        return F

    def update_output_constants(self,t,O,X)->np.ndarray:
        """override """
        return O

    def update_dynamics(self,t,X,U):
        """Updates dynamics when nonlinear is enabled, otherwise it will do nothing"""
        if not self.nonlinear:
            return
        
        #State + Control
        self.dynamic_A = self.update_state_nonlinear(t,self.static_A,X)
        self.dynamic_B = self.update_input_nonlinear(t,self.static_B,X,U)

        #Output
        self.dynamic_C = self.update_output_matrix(t,self.static_C,X)
        self.dynamic_D = self.update_feedthrough_nonlinear(t,self.static_D,X,U)

        #Constants
        self.dynamic_F = self.update_state_constants(t,self.static_F,X)
        self.dynamic_K = self.update_output_constants(t,self.static_K,X)

    def integrate_linear(self,t,dt,X,U,*args,**kwargs):
        """simulate the system over the course of time. Return time differential of the state.

        Args:
            t (float): time
            dt (float): interval to integrate over in time
            X (np.ndarray): state input
            U (np.ndarray): control input

        Returns:
            np.array: time differential of the state
        """
        return self.static_A @ X + self.static_B @ U + self.static_F

    def linear_output(self,t,dt,X,U):
        """simulate the system over the course of time. Return time differential of the state.

        Args:
            dt (float): interval to integrate over in time
            X (np.ndarray): state input
            U (np.ndarray): control input

        Returns:
            np.array: time differential of the state
        """
        return self.static_C @ X + self.static_D @ U + self.static_K

    def integrate_nonlinear(self,t,dt,X,U,update=True):
        """simulate the system over the course of time. Return time differential of the state.

        Args:
            t (float): time
            dt (float): interval to integrate over in time
            X (np.ndarray): state input
            U (np.ndarray): control input

        Returns:
            np.array: time differential of the state
        """
        if update: self.update_dynamics(t,X,U)
        return self.dynamics_A @ X + self.dynamics_B @ U + self.dynamics_F

    def nonlinear_output(self,t,dt,X,U,update=True):
        """simulate the system over the course of time. Return time differential of the state.

        Args:
            dt (float): interval to integrate over in time
            X (np.ndarray): state input
            U (np.ndarray): control input

        Returns:
            np.array: time differential of the state
        """
        if update: self.update_dynamics(t,X,U)
        return self.dynamics_C @ X + self.dynamics_D @ U + self.dynamics_K

    def determine_nearest_stationary_state(self,t=0,X=None,U=None)->np.ndarray:
        """determine the nearest stationary state"""

        if X is None:
            X = self.state
        if U is None:
            U = self.input

        if self.nonlinear:
            self.update_dynamics(t,X,U)
            Mb = self.dynamics_B @ U if self.input_size > 0 else 0
            Mx =  self.dynamics_F + Mb
            return np.linalg.solve(self.dynamics_A,-Mx)

        #static state
        Mb = self.static_B @ U if self.input_size > 0 else 0
        Mx = Mb + self.static_F
        return np.linalg.solve(self.static_A,-Mx)

@forge(auto_attribs=True)
class GlobalDynamics(DynamicsMixin):
    """This object is inherited by configurations that collect other dynamicMixins and orchestrates their simulation, and steady state analysis
    
    #TODO: establish bounds in solver
    #TODO: establish steady date analysis
    """

    def collect_internal_dynamics(self,conf:"ConfigurationMixin"=None)->dict:
        """collects the dynamics of the systems"""
        if conf is None:
            conf = self
        
        dynamics = {}
        traces = {}
        signals = {}
        solvers = {}  
        out = dict(
            dynamics = dynamics,
            traces = traces,
            signals = signals,
            solvers = solvers
            )
        
        for key,lvl,conf in conf.go_through_configurations():

            if isinstance(conf,DynamicsMixin):
                dynamics[key] = {'lvl':lvl,'conf':conf}

            tra = conf.transients_attributes()
            if tra:
                trec = {k:at.type for k,at in tra.items()}
                traces[key] = {'lvl':lvl,'conf':conf,
                               'transients':trec}

            #TODO: map signals and slots
            sig = conf.signals_attributes()
            if sig:
                sigc = {k:at.type for k,at in sig.items()}
                signals[key] = {'lvl':lvl,'conf':conf,
                                'signals':sigc}

            solv = conf.solvers_attributes()
            if solv:
                solvc = {k:at.type for k,at in solv.items()}
                solvers[key] = {'lvl':lvl,'conf':conf,
                                'solvers':solvc}

        return out

    def parse_simulation_input(self,**kwargs):
        """parses the simulation input
        
        :param dt: timestep in s, required for transients
        :param endtime: when to end the simulation
        """
        # timestep
        if "dt" not in kwargs:
            raise Exception("transients require `dt` to run")
        # f"transients require timestep input `dt`"
        dt = float(kwargs.pop("dt"))

        # endtime
        if "endtime" not in kwargs:
            raise Exception("transients require `endtime` to run")

        # add data
        _trans_opts = {"dt": None, "endtime": None}
        _trans_opts["dt"] = dt

        # f"transients require `endtime` to specify "
        _trans_opts["endtime"] = endtime = float(kwargs.pop("endtime"))
        _trans_opts["Nrun"] = max(int(endtime / dt) + 1, 1)

        #TODO: expose integrator choices
        #TODO: add delay and signal & feedback options

        return _trans_opts        



    def sim_matrix(self,eval_kw=None,sys_kw=None,*args,**kwargs):
        """simulate the system over the course of time.

        Args:
            dt (float): time step
            endtime (float): end time
            #TODO: subsystems (bool, optional): simulate subsystems. Defaults to True.

        Returns:
            dataframe: tabulated data
        """
        tr_opts = self.parse_simulation_input(**kwargs)
        from engforge.solver import SolveableMixin
        dt = tr_opts['dt']
        endtime = tr_opts['endtime']

        if isinstance(self,SolveableMixin):
            sim = lambda *args,**kw: self.simulate(dt,endtime,*args,**kw)
            self._iterate_input_matrix(sim,dt,endtime,eval_kw=eval_kw,sys_kw=sys_kw,return_results=True,*args,**kwargs)
        else:
            self.simulate(dt,endtime,eval_kw=eval_kw,sys_kw=sys_kw)


    def simulate(self,dt,endtime,cb=None,eval_kw=None,sys_kw=None)->pandas.DataFrame:
        """runs a simulation over the course of time, and returns a dataframe of the results.
        
        A copy of this system is made, and the simulation is run on the copy, so as to not affect the state of the original system.
        """

        Time = np.arange(0, endtime+dt, dt)

        #Prep for Simulation
        scopy = self.copy_config_at_state()
        dyns = scopy.collect_internal_dynamics()
        refs = references = scopy.system_references()

        #Data Storage
        data = {}

        
        #Orchestrate The Simulation
        dyn_sys = {(k,v['lvl']):v['conf'] for k,v in dyns['dynamics'].items()}
        trn_sys = {(k,v['lvl']):v['conf'] for k,v in dyns['traces'].items()}
        signal_sys = {(k,v['lvl']):v['conf'] for k,v in dyns['signals'].items()}
        solver_sys = {(k,v['lvl']):v['conf'] for k,v in dyns['solvers'].items()}

        #Create X & Index For Transient Variables
        #X should be ordered by the order of the systems
        #X = 
        #Index =

        #Run Simulation
        #1. create function to run the simulation by updating X, and recording Xdot
        #1.2. create state matrices for each system
        #1.3. organize systems into a global state space model
        #1.4. organize signals to be executed via state IO
        
        #2. run the simulation, #TODO: respecting bounds and constraints
        #While t < endtime; t += dt:
        #2.1 first run signals
        #2.2 solve the free variables
        #2.3 determine rate change of the state
        #2.4 #TODO: check boundaries & constraints| (Bnd-sign(X)*X)?

        #Simulate The Data


    @property#TODO: make the dataframe_property for the dataframe
    def dataframe(self):
        """overrides the dataframe property to collect the dataframes of the subsystems"""
        raise NotImplementedError()


if __name__ == "__main__":
    from engforge.system import System
    from engforge.components import Component
    from engforge.attr_slots import SLOT
    from engforge.attr_dynamics import TRANSIENT
    from engforge.attr_signals import SIGNAL
    from engforge.attr_solver import SOLVER
    from engforge.properties import system_property

    @forge(auto_attribs=True)
    class DynamicComponent(Component,DynamicsMixin):
        
        dynamic_state_parms:list = ['x','v']
        x:float = 1
        v:float = 0

        b:float = 0.1
        K:float = 10
        M:float = 100

        x0:float = 0.5

        nonlinear:bool = False

        def create_state_matrix(self,**kwargs) -> np.ndarray:
            """creates the state matrix for the system"""
            return np.array([[0,1],[-self.K/self.M,-self.b/self.M]])

        def create_state_constants(self,**kwargs) -> np.ndarray:
            """creates the input matrix for the system, called B"""
            return np.array([0,self.K*self.x0/self.M])



    @forge(auto_attribs=True)
    class DynamicSystem(System,GlobalDynamics):

        dynamic_state_parms:list = ['x','v']

        x:float = 0
        v:float = 0
        a:float = 0

        speed = TRANSIENT.integrate('x','v',mode='euler')
        accel = TRANSIENT.integrate('v','a',mode='euler')

        comp = SLOT.define(DynamicComponent)
        sig = SIGNAL.define('a','spring_accel')
        slv = SOLVER.define('spring_accel','comp.x0')

        @system_property
        def spring_accel(self)->float:
            return (-self.comp.v*self.comp.b - self.comp.x*self.comp.K)/self.comp.M


    dc = DynamicComponent()
    dc.create_dynamics()

    ds = DynamicSystem(comp=dc)
    ds.create_dynamics()
    #ds.update_dynamics()
    ds.collect_internal_dynamics()
    ds2 = ds.copy_config_at_state()

    ds.as_dict
        
    #N_test = 2

#     class TestSystem(DynamicSystem):
#         _x_parms = ['x1','x2']
#         _A = attrs.field(factory=lambda: np.zeros((2,2)))