"""These exist as in interface to sectionproperties from PyNite"""

from engforge.configuration import Configuration, forge, LoggingMixin
from engforge.properties import (
    cached_system_property,
    system_property,
    instance_cached,
)
from engforge.typing import Options
from engforge.eng.prediction import PredictionMixin
from engforge.env_var import EnvVariable
import numpy
import attr, attrs

from scipy import optimize as skopt

from sklearn import svm

from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.pre import Material as sec_material
from sectionproperties.analysis.section import Section
import sectionproperties.pre.library.primitive_sections as sections
import numpy as np
import shapely
import attr, attrs
import functools
import itertools
import multiprocessing as mp
import threading
import random
import pandas
import pickle
import hashlib
import tempfile,os
import json

# generic cross sections from
# https://mechanicalbase.com/area-moment-of-inertia-calculator-of-certain-cross-sectional-shapes/
temp_path = os.path.join(tempfile.gettempdir(),'shapely_sections')
section_cache = EnvVariable('FORGE_SECTION_CACHE',default = temp_path,desc='directory to cache section properties')
if 'FORGE_SECTION_CACHE' not in os.environ and not os.path.exists(temp_path):
    os.mkdir(temp_path)

section_cache.info(f'loading section from {section_cache.secret}')

class GeometryLog(LoggingMixin):
    pass
log = GeometryLog()


def conver_np(inpt):
    if isinstance(inpt,np.ndarray):
        return inpt
    elif isinstance(inpt,(list,tuple)):
        return np.array(inpt)
    elif isinstance(inpt,(int,float)):
        return np.array([inpt])
    else:
        raise ValueError(f'got non numpy array/float: {inpt}')



@attrs.define(slots=False)
class ParametricSpline:
    """a multivariate spline defined by uniform length vector input, with points P1,P2 and their slopes P1ds,P2ds"""

    P1=attrs.field(converter=conver_np)
    P2=attrs.field(converter=conver_np)
    P1ds=attrs.field(converter=conver_np)
    P2ds=attrs.field(converter=conver_np)

    def __attrs_post_init__(self):
        assert len(self.P1) == len(self.P2)
        assert len(self.P1ds) == len(self.P2ds)
        assert len(self.P1) == len(self.P1ds)

    @functools.cached_property
    def a0(self):
        return self.P1

    @functools.cached_property
    def a1(self):
        return self.P1ds

    @functools.cached_property
    def a2(self):
        return 3*(self.P2-self.P1 ) - 2*self.P1ds - self.P2ds
    
    @functools.cached_property
    def a3(self):
        return (self.P2ds - self.P1ds - 2*self.a2)/3.
    
    def coords(self,s:float):
        if isinstance(s,(float,int)):
            assert s <= 1
            assert 0 <= s
            return self._coords(s)
        elif isinstance(s,(list,tuple,numpy.ndarray)):
            assert max(s) <= 1
            assert min(s) >= 0
            ol = [self._coords(si) for si in s]
            return numpy.array(ol)
        else:
            raise ValueError(f'non array/float input {s}')

    def _coords(self,s:float):
        return self.a0 + self.a1*s + self.a2*s**2 +self.a3*s**3



# TODO: cache SectionProperty sections and develop auto-mesh refinement system.

@forge
class Profile2D(Configuration,PredictionMixin):
    name: str = attr.ib(default="generic cross section")

    # provide relative interface over
    y_bounds: tuple = None
    x_bounds: tuple = None

    @property
    def A(self):
        return 0

    @property
    def Ao(self):
        """outside area, over ride for hallow sections"""
        return self.A

    @property
    def Ixx(self):
        return 0

    @property
    def Iyy(self):
        return 0

    @property
    def J(self):
        return 0

    def display_results(self):
        self.info("mock section, no results to display")

    def plot_mesh(self):
        self.info("mock section, no mesh to plot")

    def calculate_stress(self, N, Vx, Vy, Mxx, Myy, M11, M22, Mzz):
        # TODO: Implement stress object, make fake mesh, and mock?
        # sigma_n = N / self.A
        # sigma_bx = self.Myy * self.max_y / self.Ixx
        # sigma_by = self.Mxx * self.max_x / self.Iyy
        self.warning(f'calculating stress in simple profile!')
        return np.nan

    def estimate_stress(self, N, Vx, Vy, Mxx, Myy, M11, M22, Mzz):
        # TODO: Implement stress object, make fake mesh, and mock?
        # sigma_n = N / self.A
        # sigma_bx = self.Myy * self.max_y / self.Ixx
        # sigma_by = self.Mxx * self.max_x / self.Iyy
        self.warning(f'estimating stress in simple profile!')
        return np.nan


@forge
class Rectangle(Profile2D):
    """models rectangle with base b, and height h"""

    b: float = attr.ib()
    h: float = attr.ib()
    name: str = attr.ib(default="rectangular section")

    def __on_init__(self):
        self.y_bounds = (self.h / 2, -self.h / 2)
        self.x_bounds = (self.b / 2, -self.b / 2)

    @property
    def A(self):
        return self.h * self.b

    @property
    def Ixx(self):
        return self.b * self.h**3.0 / 12.0

    @property
    def Iyy(self):
        return self.h * self.b**3.0 / 12.0

    @property
    def J(self):
        return (self.h * self.b) * (self.b**2.0 + self.h**2.0) / 12


@forge
class Triangle(Profile2D):
    """models a triangle with base, b and height h"""

    b: float = attr.ib()
    h: float = attr.ib()
    name: str = attr.ib(default="rectangular section")

    def __on_init__(self):
        self.y_bounds = (self.h / 2, -self.h / 2)
        self.x_bounds = (self.b / 2, -self.b / 2)

    @property
    def A(self):
        return self.h * self.b / 2.0

    @property
    def Ixx(self):
        return self.b * self.h**3.0 / 36.0

    @property
    def Iyy(self):
        return self.h * self.b**3.0 / 36.0

    @property
    def J(self):
        return (self.h * self.b) * (self.b**2.0 + self.h**2.0) / 12


@forge
class Circle(Profile2D):
    """models a solid circle with diameter d"""

    d: float = attr.ib()
    name: str = attr.ib(default="rectangular section")

    def __on_init__(self):
        self.x_bounds = self.y_bounds = (self.d / 2, -self.d / 2)

    @property
    def A(self):
        return (self.d / 2.0) ** 2.0 * numpy.pi

    @property
    def Ixx(self):
        return numpy.pi * (self.d**4.0 / 64.0)

    @property
    def Iyy(self):
        return numpy.pi * (self.d**4.0 / 64.0)

    @property
    def J(self):
        return numpy.pi * (self.d**4.0 / 32.0)


@forge
class HollowCircle(Profile2D):
    """models a hollow circle with diameter d and thickness t"""

    d: float = attr.ib()
    t: float = attr.ib()
    name: str = attr.ib(default="rectangular section")

    def __on_init__(self):
        self.y_bounds = (self.d / 2, -self.d / 2)
        self.x_bounds = (self.d / 2, -self.d / 2)

    @property
    def di(self):
        return self.d - self.t * 2

    @property
    def Ao(self):
        """outside area, over ride for hallow sections"""
        return (self.d**2.0) / 4.0 * numpy.pi

    @property
    def A(self):
        return (self.d**2.0 - self.di**2.0) / 4.0 * numpy.pi

    @property
    def Ixx(self):
        return numpy.pi * ((self.d**4.0 - self.di**4.0) / 64.0)

    @property
    def Iyy(self):
        return numpy.pi * ((self.d**4.0 - self.di**4.0) / 64.0)

    @property
    def J(self):
        return numpy.pi * ((self.d**4.0 - self.di**4.0) / 32.0)


# ADVANCED CUSTOM SECTIONS
def get_mesh_size(inst):
    if isinstance(inst.shape, Geometry):
        shape = inst.shape.geom
    else:
        shape = inst.shape
    
    dec = inst.mesh_extent_decimation
    x, y = shape.exterior.coords.xy
    dx = abs(max(x) - min(x))
    dy = abs(max(y) - min(y))
    ddx = np.abs(np.diff(x))
    ddx = ddx[ddx > inst.min_mesh_size].tolist()
    ddy = np.abs(np.diff(y))
    ddy = ddy[ddy > inst.min_mesh_size].tolist()
    
    A = shape.area
    dAmin = A *0.95/ (inst.goal_elements)

    cans = [dx / dec, dy / dec]
    if ddx:
        cans.append(min(ddx))
    if ddy:
        cans.append(min(ddy))

    ms = max(min(cans),inst.min_mesh_size)
    return max(ms**2.,dAmin) #length to area conversion

    
def calculate_stress(section, n=0, vx=0, vy=0, mxx=0, myy=0, mzz=0,raw=False,row=False,record=True,value=True)->float:
    """returns the maximum vonmises stress in the section and returns the ratio of the allowable stress, also known as the failure fracion
    :param raw: if raw is true, the stress object is returned, otherwise the failure fraction is returned
    """
    inp = dict(n=n, vx=vx, vy=vy, mxx=mxx, myy=myy, mzz=mzz)

    stress = section._sec.calculate_stress(**inp).get_stress()[0]
    fail_stress = section.determine_failure_stress(stress)
    inp['fail_stress'] = fail_stress
    inp['fail_frac'] = ff =  fail_stress / section.material.allowable_stress
    inp['fails'] = int(ff >= 1)

    if record:
        section.record_stress(inp)
    
    if value:
        return fail_stress
    
    if raw:
        return stress
    
    if row:
        return inp
    
    return ff

@forge(hash=False)
class ShapelySection(Profile2D):
    """a 2D profile that takes a shapely section to calculate section properties, use a sectionproperties section with hidden variable `_geo` to bypass shape calculation"""

    name: str = attrs.field(default="shapely section")
    shape: shapely.Polygon = attrs.field()

    #Mesh sizing
    coarse: bool = attrs.field(default=False)
    mesh_extent_decimation = attrs.field(default=100)
    min_mesh_angle: float = attrs.field(default=20) #below 20.7 garunteed to work
    min_mesh_size: float = attrs.field(default=1E-5) #multiply by min
    goal_elements: float = attrs.field(default=1000) #multiply by min
    _mesh_size: float = attrs.field(default=attrs.Factory(get_mesh_size, True))

    material: sec_material = attrs.field(default=None)
    failure_mode = Options('von_mises','max_norm','maximum_strain')

    #Stress classification & prediction
    prediction: bool = attr.field(default=False)
    prediction_goal_error: float = attrs.field(default=0.025)
    max_records: list = attr.field(default=10000)
    prediction_records: list
    near_margin=0.1
    max_margin=1.5
    save_threshold=0.90
    max_rec_parm = 'fail_frac'

    _use_symmetric: bool = attrs.field(default=True)
    _prediction_parms = ['n','vx','vy','mxx','myy','mzz']
    _do_print: bool = attrs.field(default=False)
    _sec: Section = None
    _geo: Geometry  = None
    _A: float
    _symmetric: bool

    def __on_init__(self):
        self.init_with_material(self.material)
            
        if self.prediction:
            self.add_prediction_record({'n':0,'vx':0,'vy':0,'mxx':0,'myy':0,'mzz':0,'fails':0,'fail_frac':0})
            self._symmetric = self.check_symmetric()
            if self._symmetric and self._use_symmetric:
                self._prediction_models = {'fails':{'mod':svm.SVC(C=2000,gamma=0.1,probability=True),'N':0},'fail_frac':{'mod':svm.SVR(C=5,gamma=0.5),'N':0}}
            else:
                self._prediction_models = {'fails':{'mod':svm.SVC(C=5000,gamma=5,probability=True),'N':0},'fail_frac':{'mod':svm.SVR(C=10,gamma=0.5),'N':0}}
            # self.determine_failure_front(pareto_front=False)
            # self.basis_expand(expand_values=[0.75,0.5,0.1],Nparm=3,est=True)

    def training_callback(self,models):
        """when training is complete save the model to a pickle with `ShapelySection_<hash>.pkl`"""
        score = models['fail_frac']['train_score']
        Nscore = models['fail_frac']['N']
        test_score = score * Nscore #compare to meta file

        #Opt out of saving if score is below threshold
        if score < self.save_threshold:
            return
        #Opt out of saving if score is worse than previous
        if os.path.exists(self.meta_path):
            with open(self.meta_path,'r') as f:
                meta = json.load(f)
            if test_score < meta['test_score']:
                self.info(f'new model score {test_score} is worse than {meta["test_score"]}, not saving')
                return

        new_meta = {'test_score':test_score,'N':Nscore,'train_score':score}
        #add a meta json file to show info regarding data (num points better ect)
        with open(self.meta_path,'w') as f:
            json.dump(new_meta,f)

        #Finally save the geometry
        self.info(f'saving model with score: {Nscore}x{score}=>{test_score}')
        fil = self.cache_path
        with open(fil,'wb') as f:
            pickle.dump(self,f)
        self.info(f'saved section to {fil}')

    

    @property
    def section_cache(self) -> str:
        return section_cache.secret

    @property
    def cache_name(self) -> str:
        return f'ShapelySection_{self.hash_id()}.pkl'

    @property
    def meta_name(self) -> str:
        return f'ShapelySection_meta_{self.hash_id()}.json'        

    @property
    def cache_path(self):
        return os.path.join(self.section_cache,self.cache_name)

    @property
    def meta_path(self):
        return os.path.join(self.section_cache,self.meta_name)

    @classmethod
    def from_cache(cls,hash_id):
        cchc = section_cache.secret
        log.info(f'loading section {hash_id} from cache {cchc}')
        fil = os.path.join(cchc,f'ShapelySection_{hash_id}.pkl')
        with open(fil,'rb') as f:
            model = pickle.load(f)
        return model

    def prediction_weights(self,df,window,initial_weight=10):
        weights = numpy.ones(min(len(df),window))
        weights[0] = initial_weight**2 #zero value is important!
        weights[:getattr(self,'N_base',100)] = initial_weight #then base values
        if hasattr(self,'N_pareto'):
            weights[:getattr(self,'N_pareto')] = initial_weight**0.5 #then pareto values
        #Dont emphasise fit above max margin
        dm = (df.fail_frac - self.max_margin).to_numpy()
        penalize_inx = (dm>0)
        weights[penalize_inx] = np.maximum(1.0/((1.0+dm[penalize_inx])),0.1)
        return weights   

    def _subsample_data(self,X,y,window,weights):
        """subsamples the data to the window size"""
        inx = getattr(self,'N_pareto',getattr(self,'N_base',window))
        x1 = X.iloc[:inx]
        y1 = y.iloc[:inx]
        w1 = weights[:inx]
        if inx != window:
            x2 = X.iloc[inx:].sample(frac=0.5)
            y2 = y.iloc[x2.index]
            w2 = weights[x2.index]
            return pandas.concat((x1,x2)),pandas.concat((y1,y2)),numpy.concatenate((w1,w2))
        return X.iloc[:window],y.iloc[:window],weights[:window]      

    def reset_prediction(self):
        self._fitted = False
        self._basis = None
        self._training_history = None
        self._running_error = None
        self.add_prediction_record({'n':0,'vx':0,'vy':0,'mxx':0,'myy':0,'mzz':0,'fails':0,'fail_frac':0})
        #self._symmetric = self.check_symmetric()

    @property
    def mesh_size(self):
        return self._mesh_size
    
    @property
    def _prediction_record(self):
        """not a property of state, just return an empty dict, we add record manually"""
        return {}
    
    @mesh_size.setter
    def mesh_size(self, value):
        #print(f'setting mesh size to {value}')
        #BUG: something going on with setattrs on mesh_size, setting to None, hacky fix to set __dict__ directly 
        _mesh_size = max(value,self.min_mesh_area)
        self.__dict__['_mesh_size'] = _mesh_size
        self.mesh_section()

    def init_with_material(self, material=None):
        if self._sec is not None:
            raise Exception(f"already initalized!")
        
        if isinstance(self.shape,Geometry):
            self._geo = self.shape
            if self._geo.material and self.material:
                self.warning(f'overriding material {self._geo.material} with {self.material}')
                self._geo.material = self.material
            elif self._geo.material:
                self.info(f'setting beam material from section')
                self.material = self._geo.material
            elif self.material:
                self._geo.material  = self.material
        elif isinstance(self.shape,shapely.Polygon):
            self._geo = Geometry(self.shape, self.material)
        else:
            raise ValueException(f'got invalid shape: {self.shape}')
        
        self.calculate_mesh_size()
        self.mesh_section()

    def calculate_mesh_size(self):
         self.mesh_size = get_mesh_size(self)

    @property
    def min_mesh_area(self):
        if isinstance(self.shape, Geometry):
            shape = self.shape.geom
        else:
            shape = self.shape        
        A = shape.area
        dAmin = A *0.95/ (self.goal_elements)
        return dAmin
        
    def mesh_section(self):    
        """caches section properties and mesh""" 
        self._cross_section = None #reset cross section 
        self._mesh = self._geo.create_mesh(mesh_sizes=self.mesh_size, coarse=self.coarse,min_angle=self.min_mesh_angle)
        self._sec = Section(self._geo)
        self._sec.calculate_geometric_properties()
        self._sec.calculate_warping_properties()
        self._sec.calculate_frame_properties()

        self._A = self._sec.get_area()
        if self.material:
            self._Ixx, self._Iyy, self._Ixy = self._sec.get_eic(e_ref=self.material)
            self._J = self._sec.get_ej()
        else:
            self._Ixx, self._Iyy, self._Ixy = self._sec.get_ic()
            self._J = self._sec.get_j()

        self.calculate_bounds()


    def calculate_bounds(self):
        self.info(f"calculating shape bounds!")
        xcg, ycg = self._geo.calculate_centroid()
        minx, maxx, miny, maxy = self._geo.calculate_extents()
        self.y_bounds = (miny - ycg, maxy - ycg)
        self.x_bounds = (minx - xcg, maxx - xcg)

    @property
    def A(self):
        return self._A

    @property
    def Ao(self):
        """outside area, over ride for hallow sections"""
        return self.A

    @property
    def Ixx(self):
        return self._Ixx

    @property
    def Iyy(self):
        return self._Iyy

    @property
    def J(self):
        return self._J

    @property
    def Ixy(self):
        return self._Ixy

    def display_results(self):
        self.info("mock section, no results to display")

    def plot_mesh(self):
        self._sec.display_mesh_info()
        

    def plot_mesh(self):
        return self._sec.plot_centroids()

    def calculate_stress(self, n=0, vx=0, vy=0, mxx=0, myy=0, mzz=0,**kw)->float:
        return calculate_stress(self,n=n, vx=vx, vy=vy, mxx=mxx, myy=myy, mzz=mzz,**kw)
        
    def estimate_stress(self, n=0, vx=0, vy=0, mxx=0, myy=0, mzz=0,value=False,calc_margin=2,min_est_records=100,calc_every=25,pre_train_margin=2,force_calc=False)->float:
        """uses a support vector machine to estimate stresses and returns the ratio of the allowable stress, also known as the failure fracion if prediction is set to True, otherwise calculates stress"""
        Nrec = len(self._prediction_records) if self._prediction_records else 0
        under_size = Nrec <= min_est_records
        do_calc = not self.prediction or not self._fitted or under_size
        if do_calc or force_calc:
            if self._do_print:
                print(f'calc till {len(self._prediction_records)} <= {min_est_records}')
            stress = calculate_stress(self,n=n, vx=vx, vy=vy, mxx=mxx, myy=myy, mzz=mzz,value=value)
            return stress 
        else:
            parms = self._prediction_parms
            data = dict(n=n,vx=vx,vy=vy,mxx=mxx,myy=myy,mzz=mzz)
            
            #Format data for prediction
            if self._symmetric and self._use_symmetric:
                inp = {k:abs(data[k]/v) for k,v in zip(parms,self._basis)}
            else:
                inp = {k:data[k]/v for k,v in zip(parms,self._basis)}
            X = pandas.DataFrame([inp])
            val = self._prediction_models['fail_frac']['mod'].predict(X)[0]

            #Provide margin of error and use estimate where possible
            if not self.trained:
                #calc when in doubt   
                calc_margin = calc_margin*pre_train_margin 
                
            #calculate stress if close to failure within saftey margin
            err = 1-val
            mrg = self.fail_frac_criteria(calc_margin=calc_margin)
            do_calc = abs(err)<=mrg or all([calc_every,(Nrec%calc_every)==0])
            oob = val <= self.max_margin and self.check_out_of_domain(data,0.1)
            if self._do_print:
                self.info(f'{"calc" if do_calc or oob else "est"} stress {abs(err):5.3f}>{mrg*calc_margin:5.3f}| {calc_margin} | oob {oob}')

            if do_calc:
                return self.calculate_stress(n=n, vx=vx, vy=vy, mxx=mxx, myy=myy, mzz=mzz,value=value)

            elif oob:
                # if self._do_print:
                #     self.info(f'out of domain {val:5.3f}<={self.max_margin:5.3f}')
                return self.calculate_stress(n=n, vx=vx, vy=vy, mxx=mxx, myy=myy, mzz=mzz,value=value)
            

                
            #otherwise prediction is value
            if value:
                return val * self.material.allowable_stress

            return val
        
    def fail_frac_criteria(self,calc_margin=2,min_rec=1000):
        MargRec = max(min_rec/len(self._prediction_records),1)

        if self._training_history:
            score = self._training_history[-1]['fail_frac']['scr']
        else:
            score = self._prediction_models['fail_frac']['train_score']

        mrg = max(abs((1-score)),self.prediction_goal_error)
        return mrg*calc_margin*MargRec
    
    def estimate_failure(self, n=0, vx=0, vy=0, mxx=0, myy=0, mzz=0)->float:
        """uses a support vector machine to estimate stresses and returns a number zero or one to indicate failure, if prediction is set to True, otherwise calculates stress"""
        if not self.prediction or not self._fitted:
            ff = calculate_stress(self,n=n, vx=vx, vy=vy, mxx=mxx, myy=myy, mzz=mzz,row=True,value=False)
            return ff['fails']
        else:
            #TODO: add logic to determine if calculation should be done close to failure
          
            parms = self._prediction_parms
            data = dict(n=n,vx=vx,vy=vy,mxx=mxx,myy=myy,mzz=mzz)
            if self._symmetric and self._use_symmetric:
                inp = {k:abs(data[k]/v) for k,v in zip(parms,self._basis)}
            else:
                inp = {k:data[k]/v for k,v in zip(parms,self._basis)}
            X = pandas.DataFrame([inp])
            return self._prediction_models['fails']['mod'].predict(X)[0]  

    def record_stress(self,stress_dict):
        """determines if stress record should be added to prediction_records"""
        if not self.prediction:
            return
        if len(self._prediction_records) > self.max_records:
            return
        #Add the data to the stress records
        #TODO: add logic to determine if stress record should be added to prediction_records
        ff = stress_dict['fail_frac']
        if ff == 0:
            #null included
            return

        #Convert to positive domain
        if self._symmetric and self._use_symmetric:
            stress_dict = {k:abs(v) for k,v in stress_dict.items()}
        
        if not getattr(self,'N_base',None):
            max_margin = None
        else:
            max_margin = self.max_margin
        
        if self.near_margin and abs(ff-1) < self.near_margin:
            #near failure always add to map resolution
            self.add_prediction_record(stress_dict)
        elif max_margin and ff < max_margin*2:
            #under max marging
            self.add_prediction_record(stress_dict,False,0.1)
        elif not self.near_margin and not max_margin:
            #no checks
            self.add_prediction_record(stress_dict,False)
            

    def determine_failure_stress(self,stress_obj):
        """uses the failure mode to compare to allowable stress"""
        if self.failure_mode == 'von_mises':
            return stress_obj['sig_vm'].max()
        elif self.failure_mode == 'max_norm':
            return np.nan #TODO: make this work
        elif self.failure_mode == 'maximum_strain':
            return np.nan #TODO: make this work
        else:
            raise ValueError(f'invalid failure mode: {self.failure_mode}')

    def fail_learning(self,X,parm,base_kw,mult=1):
        '''optimizes stress until failure, given base_kw arguments for extra parameters'''
        base_kw = base_kw.copy()
        inpt = {parm:mult*X}
        base_kw.update(inpt)
        
        base_kw['value'] = False #override value for calcstress to fail frac
        ff = self.calculate_stress(**base_kw).item()
        return 1 - ff

    def solve_fail(self,fail_parm,base_kw,guess=None,tol=1E-4,mult=1):
        if guess is None:
            guess = 1000*random.random()
        kw = {}
        # if self._basis is not None:
        #     fpinx = self._prediction_parms.index(fail_parm)
        #     bracket = (0,self._basis[fpinx]*1.25)
        #     kw['bracket']  = bracket
            
        ans = skopt.root_scalar( self.fail_learning , x0=guess*0.1, x1=guess, xtol = tol, args=(fail_parm,base_kw,mult))#,**kw)
        if self._do_print:
            self.info(f'{mult}x{fail_parm:<6}| success: {ans.converged} , ans:{ans.root}, base: {base_kw}')
        else:
            self.debug(f'{mult}x{fail_parm:<6}| success: {ans.converged} , ans:{ans.root}, base: {base_kw}')
        if ans.converged:
            return ans.root
        return 1E6

    #Determine Outer Bound Of Failures
    def determine_failure_front(self,pareto_inx = [0.5,0.1],pareto_front=False):
        self.info(f'determining failure front for cross section, with pareto inx: {pareto_inx}')
        null_kw = {}
        if self._symmetric:
            mvec = [1]
        else:
            mvec = [-1,1]
        
        if self._basis is None:
            self.info(f'determining normalization basis')
            res = {}
            for mult in mvec:
                res[mult] = {}
                for p in self._prediction_parms:
                    res[mult][p] = self.solve_fail(p,null_kw,mult)

            self._basis = np.array([max([abs(v.get(p,1E6))  for k,v in res.items() ]) for p in self._prediction_parms])

            self.N_base = len(self._prediction_records)

        #Second Pareto Value Calc
        if pareto_front:
            for mult in mvec:
                for aux_frac in pareto_inx:
                    #TODO: expand parato combos past 2
                    for fail_parm,aux_parm in itertools.combinations(self._prediction_parms,2):
                        ainx = self._prediction_parms.index(aux_parm)
                        aux_val = self._basis[ainx]*aux_frac
                        base_kw = {aux_parm:aux_val}
                        finx = self._prediction_parms.index(fail_parm)
                        guesstimate = self._basis[finx]*(1-aux_frac)
                        self.solve_fail(fail_parm,base_kw,guess=guesstimate,mult=mult)
                        if not self._symmetric:
                            self.solve_fail(fail_parm,base_kw,guess=guesstimate,mult=-1*mult) #alternato
    
            self.N_pareto = len(self._prediction_records)
    
    def basis_expand(self,expand_values=[0.9,0.75,0.5,0.1,0.01],Nparm=4,est=True,normalize=True):
        """run combinations of parameters and permutations of weights against the basis values to populate the stress records, by default using estimation logic to speed up the process"""
        i = 0
        Nparm = min(min(Nparm,len(self._prediction_parms)),len(expand_values))
        for parms in itertools.combinations(self._prediction_parms,Nparm):
            for weight in itertools.permutations(expand_values,Nparm):
                wt = sum(weight)
                q = max(wt,1) if normalize else 1
                inxs = [self._prediction_parms.index(p) for p in parms]
                base_kw = {p:w*self._basis[i]/q for p,w,i in zip(parms,weight,inxs)}
                
                #print(base_kw)
                if est:
                    self.estimate_stress(**base_kw)
                else:
                    self.calculate_stress(**base_kw)
                i+=1
                if i%100 == 0:
                    self.info(f'basis expansion... {i}')
                    

    def random_force_input(self,exp_mg=12,exp_min=6,min_ex=1,max_ex=3):
        porp = np.random.random(size=6)**np.random.randint(numpy.random.randint(min_ex,max_ex),numpy.random.randint(exp_min,exp_mg),size=(6,))
        stress = porp * self._basis
        return {k:v for k,v in zip(self._prediction_parms,stress)}        
        
    def train_until_valid(self,print_interval=50,max_iter=1000,est=False):
        """trains the prediction models until the error is below the goal error"""
        i = 0
        goal = self.prediction_goal_error
        while not self.trained:
            inp = self.random_force_input()
            if est:
                self.estimate_stress(**inp)
            else:
                self.calculate_stress(**inp)
            
            if i % print_interval == 0:
                self.info(f'training... current error: {self._training_history[-1]}')
            i+= 1
            if i >= max_iter:
                self.info(f'training... max iterations reached')
                return
            
    
    #Geometric Prediction Solutions
    def check_symmetric(self,precision=3,Nincr=180):
        """checks if the section is symmetric about the x and y axis, by finding the intersection of """
        if isinstance(self.shape, Geometry):
            shape = self.shape.geom
        else:
            shape = self.shape
        x,y = (np.array(v) for v in shape.exterior.coords.xy)
        xc = shape.centroid.x
        yc = shape.centroid.y
        dx= x-xc
        dy= y-yc

        dth = np.arctan2(dx,dy)
        rth = (dx**2 + dy**2)**0.5
        inx = np.cumsum(np.ones(dth.size))-1

        Nincr = 1000
        Imax = inx.max()
        inx2 = np.linspace(0,Imax,Nincr)
        oppo = lambda i: (i+Imax/2)%Imax

        R = np.interp(inx2,inx,rth)
        T = np.interp(inx2,inx,dth)
        X = np.cos(T)*R
        Y = np.sin(T)*R

        #finds set of radius at a given angle
        def find_radii(targetth):
            """targetth must be positive from 0->pi"""
            r = (dth - targetth)
            #print(r)
            if np.all(r < 0):
                r = r + np.pi
            elif np.all(r > 0):
                r = r - np.pi
            sgn = np.concatenate([ r[1:]*r[:-1], [r[0]*r[-1]] ] )
            possibles = np.where( sgn < 0)[0]
            #print(f'\n{targetth}')
            out = set()
            for poss in possibles:
                poss2 = int((poss+1)%Imax)
                x_ = np.array([r[poss],r[poss2]])
                y_ = np.array([inx[poss],inx[poss2]])
                itarget = (0 - x_[0])*(y_[1]-y_[0])/(x_[1]-x_[0]) + y_[0]
                r_ = np.interp(itarget,inx2,R)
                out.add(round(r_,precision))
                #print(itarget,r_)
            return out

        #check symmetry from 0-180 deg
        syms = []
        for targetth in np.linspace(0,np.pi,Nincr):
            o1 = find_radii(targetth)
            o2 = find_radii(targetth-np.pi)
            sym_point = len(o1.intersection(o2)) >= 1
            syms.append(sym_point)
            
        return np.all(syms)
    

    def __hash__(self):
        """uniqueness based on geometry and material"""
        #print('hash shape...')
        if isinstance(self.shape, Geometry):
            shape = self.shape.geom
        else:
            shape = self.shape        
        vals = (shape.wkb_hex,str(hash(self.material)),str(self.goal_elements),str(self.min_mesh_angle) )
        h=hashlib.md5()
        for hv in vals:
            h.update(hv.encode())
        return int(h.hexdigest(),16)  
        
    def hash_id(self)->str:
        """string for saving to persisting"""
        return str(hash(self))
            
ALL_CROSSSECTIONS = [
    cs for cs in locals() if type(cs) is type and issubclass(cs, Profile2D)
]
