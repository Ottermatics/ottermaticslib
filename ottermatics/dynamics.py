"""Combines the tabulation and component mixins to create a mixin for systems and components that have dynamics, such as state space models, while allowing nonlinear dynamics via matrix modification

This module is intended to work alongside the solver module and the TRANSIENT integrating attributes, and will raise an error if a conflict is detected.

The DynamicsMixin works by establishing a state matricies A, B, C, and D, which are used to define the dynamics of the system. The state matrix A is the primary matrix, and is used to define the state dynamics of the system. The input matrix B is used to define the input dynamics of the system. The output matrix C is used to define the output dynamics of the system. The feedthrough matrix D is used to define the feedthrough dynamics of the system.

As opposed to the TRANSIENT attribute, which modifies the state of the system, the DynamicsMixin copies the initial values of the state and input which then are integrated over time. At predefined intervals the control and output will stored in the tabulation.

#TODO: The top level system will collect the underlying dynamical systems and combine them to an index and overall state space model. This will allow for the creation of a system of systems, and the ability to create a system of systems with a single state space model.

#TODO: integration is done by the solver, where DynamicSystems have individual solver control, solver control is set for a smart default scipy 
"""

from engforge.components import Component, forge
from engforge.systems import System
from engforge.tabulation import TabulationMixin
from engforge import properties as prop

import numpy as np
import attr, attrs

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

    def reindify(self,new_index,*args,invert=False,old_data=None):
            if old_data is None:
                old_data = self.data
            opt1 = {arg: self.indify(old_data,arg)[0] for arg in args}
            opt2 = {arg: self.indify(old_data,val)[0] if (not isinstance(val,str)) else val for arg,val in opt1.items()}
            oop1 = {arg: self.indify(new_index,val)[0]  for arg,val in opt2.items()}
            oop2 = {arg: self.indify(new_index,val)[0] if (invert != isinstance(val,self.oppo[arg.__class__])) else val for arg,val in oop1.items()}
            return oop2

@forge(auto_attribs=True)
class DynamicsIntegrator(TabulationMixin):

    def simulate(self,dt,endtime,subsystems=True,*args,**kwargs):
        """simulate the system over the course of time.

        Args:
            dt (float): time step
            endtime (float): end time
            subsystems (bool, optional): simulate subsystems. Defaults to True.

        Returns:
            dataframe: tabulated data
        """
        if self.nonlinear:
            return self.simulate_nonlinear(dt,endtime,*args,**kwargs)
        else:
            return self.simulate_linear(dt,endtime,*args,**kwargs)

    def simulate_linear(self,dt,endtime,subsystems=True,*args,**kwargs):
    

    def simulate_nonlinear(self,dt,endtime,subsystems=True,*args,**kwargs):
        pass

@forge(auto_attribs=True)
class DynamicsMixin(TabulationMixin):
    """dynamic mixin for components and systems that have dynamics, such as state space models, while allowing nonlinear dynamics via matrix modification. This mixin is intended to work alongside the solver module and the TRANSIENT integrating attributes, and will raise an error if a conflict is detected #TODO.
    """

    state_parms:list = attrs.field(factory=list)
    input_parms:list = attrs.field(factory=list)
    output_parms:list = attrs.field(factory=list)
    A:np.ndarray = None
    B:np.ndarray = None
    C:np.ndarray = None
    D:np.ndarray = None

    control_module = None

    nonlinear:bool = False
    #TODO: add control module with PID control and pole placement design

    @property
    def state_size(self):
        return len(self.x_parms)
    
    @property
    def input_size(self):
        return len(self.input_parms)

    @property
    def output_size(self):
        return len(self.output_parms)    
    
    @property
    def x_parms(self):
        return self.A.shape[0]    

    def create_state_matrix(self,**kwargs) -> np.ndarray:
        """creates the state matrix for the system"""
        return np.zeros((self.state_size,self.state_size))
    
    def create_state_input_matrix(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros((self.state_size,self.input_size))
    
    def create_output_matrix(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros((self.output_size,self.state_size)) 
    
    def create_feedthrough_matrix(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros((self.output_size,self.input_size))

    def create_state_constants(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros(self.state_size)

    def create_output_constants(self,**kwargs) -> np.ndarray:
        """creates the input matrix for the system, called B"""
        return np.zeros(self.output_size)        

    def create_dynamics(self,**kw):
        """creates a dynamics object for the system"""
        self.A = self.create_state_matrix(**kw)
        self.B = self.create_state_input_matrix(**kw)
        self.C = self.create_output_matrix(**kw)
        self.D = self.create_feedthrough_matrix(**kw)
        self.F = self.create_state_constants(**kw)
        self.O = self.create_output_constants(**kw)

    #Nonlinear Support
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
        return D


@forge
class DynamicsComponent(Component,DynamicsMixin):
    pass


@forge
class DynamicSystem(System,DynamicsMixin):
    pass




if __name__ == "__main__":
    
    N_test = 2

    class TestSystem(DynamicSystem):

        _x_parms = ['x1','x2']
        _A = attrs.attrib(factory=np.zeros((2,2)))