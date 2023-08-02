import attr
import numpy
import functools
import shapely
import pandas
from shapely.geometry import Polygon, Point

from ottermatics.tabulation import (
    TABLE_TYPES,
    NUMERIC_VALIDATOR,
    system_property,
)
from ottermatics.configuration import otterize, Configuration
from ottermatics.components import Component

# from ottermatics.analysis import Analysis
from ottermatics.properties import system_property
from ottermatics.system import System
from ottermatics.component_collections import ComponentDict
from ottermatics.eng.solid_materials import *
from ottermatics.common import *

import sectionproperties
import sectionproperties.pre.geometry as geometry
import sectionproperties.pre.library.primitive_sections as sections
import PyNite as pynite
from PyNite import Visualization
import pandas as pd
import ray
import collections


from ottermatics.eng.structure_beams import Beam,rotation_matrix_from_vectors


SECTIONS = {
    k: v
    for k, v in filter(
        lambda kv: issubclass(kv[1], geometry.Geometry)
        if type(kv[1]) is type
        else False,
        sections.__dict__.items(),
    )
}

class StopAnalysis(Exception): 
    pass 

nonetype = type(None)

def beam_collection():
    return ComponentDict(component_type=Beam,name='beams')

# TODO: Make analysis, where each load case is a row, but how sytactically?
@otterize
class Structure(System):
    """A integration between sectionproperties and PyNite, with a focus on ease of use

    Right now we just need an integration between Sections+Materials and Members, to find, CG, and inertial components

    Possible Future Additions:
    1) Structure Motion (free body ie vehicle , multi-strucutre ie robots)
    """

    frame: pynite.FEModel3D = None
    #_beams: dict = None
    _materials: dict = None

    default_material: SolidMaterial = attrs.field(default=None)

    #Default Loading!
    add_gravity_force: bool = attrs.field(default=True)
    gravity_dir: str = attrs.field(default="FZ")
    gravity_mag: float = attrs.field(default=9.81)
    gravity_scalar: float = attrs.field(default=-1)
    gravity_cases:list = attrs.field(default=['gravity'])
    default_case:str = attrs.field(default='gravity')
    default_combo:str = attrs.field(default='gravity')

    #this orchestrates load retrieval
    current_combo:str = attrs.field(default='gravity')

    #beam_collection
    beams: ComponentDict = attrs.field(default=attrs.Factory(beam_collection))

    _always_save_data = True #we dont respond to inputs so use this
    _meshes = None

    def __on_init__(self):
        self._materials = {}
        self._meshes = {}        
        self.initalize_structure()
        self.frame.add_load_combo('gravity',{'gravity':1.0},'gravity')

    #Execution
    def run(self,combos:list=None,*args,**kwargs):
        """wrapper allowing saving of data by load combo"""

        #the string input case, with csv support
        if isinstance(combos,str):
            combos = combos.split(',') #will be a list!

        self.info(f"running with combos: {combos} | {args} | {kwargs} ")
        self._run_id = int(uuid.uuid4())
        for combo in self.LoadCombos:
            if combos is not None and combo not in combos:
                continue
            self.index += 1
            self.info(f'running load combo : {combo}')
            self.current_combo = combo
            self.pre_execute(combo)
            self.analyze(combos=[combo],*args,**kwargs)
            self.post_execute(combo)
            self.save_data(force=True) #backup data saver.


    def pre_execute(self,combo):
        """yours to override to prep solver or dataframe"""
        pass

    def post_execute(self,combo):
        """yours to override dataframe"""
        pass

    
    def initalize_structure(self):
        self.frame = pynite.FEModel3D()
        #self.beams = {}  # this is for us!
        self._materials = {}
        self.debug("created frame...")

        if self.default_material:
            material = self.default_material
            uid = material.unique_id
            if uid not in self._materials:
                self.debug(f'add material {uid}')
                self.frame.add_material(
                    uid, material.E, material.G, material.nu, material.rho
                )
                self._materials[uid] = material

    #Merged Feature Calls
    def add_eng_material(self,material:'SolidMaterial'):

        if material.unique_id not in self._materials:
            self.add_material(material.unique_id,material.E,material.G,material.nu,material.rho)
            self._materials[material.unique_id] = material

        
    def add_mesh(self,meshtype,*args,**kwargs):
        """maps to appropriate PyNite mesh generator but also applies loads
        
        Currently these types are supported ['rect','annulus','cylinder','triangle']
        """
        mesh = None
        types = ['rect','annulus','cylinder','triangle']
        if meshtype not in types:
            raise KeyError(f'type {type} not in {types}')
        
        if meshtype == 'rect':
            mesh = self.add_rectangle_mesh(*args,**kwargs)
        elif meshtype == 'annulus':
            mesh = self.add_annulus_mesh(*args,**kwargs)
        elif meshtype == 'cylinder':
            mesh = self.add_cylinder_mesh(*args,**kwargs)
        elif meshtype == 'triangle':
            mesh = self.add_frustrum_mesh(*args,**kwargs)

        assert mesh is not None, f'no mesh made!'

        self._meshes[mesh] = self.frame.Meshes[mesh]

        self.merge_duplicate_nodes()

        if self.add_gravity_force:
            self.apply_gravity_to_mesh(mesh)

    def apply_gravity_to_mesh(self,meshname):
        self.debug(f'applying gravity to {meshname}')
        mesh = self._meshes[meshname]

        mat = self._materials[mesh.material]
        rho = mat.rho

        for elid,elmt in mesh.elements.items():
            
            #make node vectors
            n_vecs = {}
            nodes = {}
            for nname in ['i_node','j_node','m_node','n_node']:
                ni = getattr(elmt,nname)
                nodes[nname] = ni
                n_vecs[nname] = numpy.array([ni.X,ni.Y,ni.Z])
            ni = n_vecs['i_node']
            nj = n_vecs['j_node']
            nm = n_vecs['m_node']
            nn = n_vecs['n_node']
            
            #find element area
            im = ni-nm
            jn = nj-nn

            A = numpy.linalg.norm(numpy.cross(im,jn))/2 
             
            #get gravity_forcemultiply area x thickness x density/4 
            mass = A * elmt.t * rho 
            fg = mass * self.gravity_mag * self.gravity_scalar
            fnode = fg / 4
            
            #apply forces to corner nodes / 4
            for case in self.gravity_cases:
                for nname,disnode in nodes.items():
                    self.add_node_load(disnode.name,self.gravity_dir,fnode,case=case)


    def add_member(self, name, node1, node2, section, material=None, **kwargs):
        assert node1 in self.nodes
        assert node2 in self.nodes

        if material is None:
            material = self.default_material

        uid = material.unique_id
        if uid not in self._materials:
            self.debug(f'add material {uid}')
            self.frame.add_material(
                uid, material.E, material.G, material.nu, material.rho
            )
            self._materials[uid] = material

        beam_attrs = {k:v for k,v in kwargs.items() if k in Beam.input_attrs()}
        kwargs = {k:v for k,v in kwargs.items() if k not in beam_attrs}
        # print(Beam.input_attrs())
        # print(beam_attrs)
        # print(kwargs)


        B = beam = Beam(
            structure=self, name=name, material=material, section=section,**beam_attrs
        )
        self.beams[name] = beam
        self.frame.add_member(
            name,
            i_node=node1,
            j_node=node2,
            material=uid,
            Iy=B.Iy,
            Iz=B.Ix,
            J=B.J,
            A=B.A,
            **kwargs,
        )
        
        if self.add_gravity_force:
            beam.apply_gravity_force(z_dir=self.gravity_dir,z_mag=self.gravity_scalar)

        return beam

    def add_member_with(
        self, name, node1, node2, E, G, Iy, Ix, J, A, material_rho=0,**kwargs
    ):
        """a way to add specific beam properties to calculate stress,
        This way will currently not caluclate resulatant beam load.
        #TOOD: Add in a mock section_properties for our own stress calcs
        """
        assert node1 in self.nodes
        assert node2 in self.nodes

        material = SolidMaterial(
            density=material_rho, elastic_modulus=E, shear_modulus=G
        )
        uid = material.unique_id
        if uid not in self._materials:
            self.debug(f'add material {uid}')
            self.frame.add_material(
                uid, material.E, material.G, material.nu, material.rho
            )
            self._materials[uid] = material

        B = beam = Beam(
            structure=self,
            name=name,
            material=material,
            in_Iy=Iy,
            in_Ix=Ix,
            in_J=J,
            in_A=A,
            section=None,
        )
        self.beams[name] = beam

        if self.add_gravity_force:
            beam.apply_gravity_force(z_dir=self.gravity_dir,z_mag=self.gravity_scalar)    

        self.frame.add_member(name, i_node=node1, j_node=node2, material=uid, Iy=Iy, Iz=Ix, J=J, A=A,**kwargs)

        return beam

    #OUTPUT
    def beam_dataframe(self,univ_parms:dict=None):
        """creates a dataframe entry for each beam and combo
        :param univ_parms: keys represent the dataframe parm, and the values represent the lookup value
        """
        df = self.dataframe
        beam_col = set([ c for c in df.columns if c.startswith('beams.') ])
        beams = set([c.split('.')[1] for c in beam_col])
        parms = set(['.'.join(c.split('.')[2:]) for c in beam_col])

        #defaults
        if univ_parms is None:
            univ_parms = {}

        blade_data = []
        for i in range(len(df)):
            row = df.iloc[i]
            out = {k:row[v] for k,v in univ_parms.items()}
            for beam in beams:
                bc = out.copy() #this is the data entry
                bc['name'] = bc
                for parm in parms:
                    k = f'beams.{beam}.{parm}'
                    v = row[k]
                    bc[parm] = v
                    
                blade_data.append(bc) #uno mas

        dfb = pd.DataFrame(blade_data)
        return dfb

    #TODO: add mesh COG
    @property
    def cog(self):
        XM = numpy.sum(
            [bm.mass * bm.centroid3d for bm in self.beams.values()], axis=0
        )
        return XM / self.mass

    # TODO: add cost and mass of meshes
    @system_property
    def mass(self) -> float:
        """sum of all beams mass"""
        return sum([bm.mass for bm in self.beams.values()])

    @system_property
    def cost(self) -> float:
        """sum of all beams cost"""
        return sum([bm.cost for bm in self.beams.values()])

    def visulize(self, **kwargs):
        if 'combo_name' not in kwargs:
            kwargs['combo_name'] = 'gravity'
        Visualization.render_model(self.frame, **kwargs)

    # TODO: add mesh stress / deflection info min/max ect.
    @property
    def node_dataframes(self):
        out = {}
        for case in self.frame.LoadCombos:
            rows = []
            for node in self.nodes.values():
                row = {
                    "name": node.name,
                    "dx": node.DX[case],
                    "dy": node.DY[case],
                    "dz": node.DZ[case],
                    "rx": node.RX[case],
                    "ry": node.RY[case],
                    "rz": node.RZ[case],
                    "rxfx": node.RxnFX[case],
                    "rxfy": node.RxnFY[case],
                    "rxfz": node.RxnFZ[case],
                    "rxmx": node.RxnMX[case],
                    "rxmy": node.RxnMY[case],
                    "rxmz": node.RxnMZ[case],
                }
                rows.append(row)

            out[case] = pandas.DataFrame(rows)
        return out

    @property
    def INERTIA(self):
        """Combines all the mass moments of inerita from the internal beams (and other future items!) into one for the sturcture with the parallel axis theorm"""
        cg = self.cog
        I = numpy.zeros((3, 3))

        for name, beam in self.beams.items():
            self.debug(f"adding {name} inertia...")
            I += beam.GLOBAL_INERTIA

        return I

    @property
    def Fg(self):
        """force of gravity"""
        return numpy.array([0, 0, -self.mass * g])

    def __getattr__(self, attr):
        #print(f'get {attr}')
        if attr in dir(self.frame):
            return getattr(self.frame, attr)           
        #if attr in dir(self):
            #return getattr(self,attr)         
        #raise AttributeError(f'{attr} not found!')
        return self.__getattribute__(attr)
        


    def __dir__(self):
        return sorted(set(dir(super(Structure, self))+dir(Structure)+ dir(self.frame)+dir(self.__class__)))


    @property
    def nodes(self):
        return self.frame.Nodes

    @property
    def members(self):
        return self.frame.Members


    #Custom Failure Analysis Tools
    @solver_cached
    def full_failures(self):
        return self.estimated_failures()

    def estimated_failures(self,concentrationFactor=10,methodFactor=2.5,fail_frac=0.9,combos:list=None)->dict:
        """uses the beam estimated stresses with specific adders to account for beam 2D stress concentrations and general inaccuracy as well as a fail_fraction to analyze. The estimated loads are very pessimistic with a 250x overestimate"""

        #the string input case, with csv support
        if isinstance(combos,str):
            combos = combos.split(',') #will be a list!        

        #what fraction of allowable stress is generated for this case
        fail_1 = fail_frac/(concentrationFactor*methodFactor)
        
        failures = collections.defaultdict(dict)

        df = self.dataframe
        run_combos = df.current_combo.to_list()

        failures_count = 0
        cases = 0

        #gather forces for each beam
        for nm,beam in self.beams.items():
            self.debug(f'estimate beam stress: {beam.name}')
              
            #try each load combo
            for combo in self.LoadCombos:

                if combo not in run_combos:
                    continue #theres, no point

                if combos and combo not in combos:
                    continue #you are not the chose one(s)
                
                combo_df = df[df.current_combo==combo]
                combo_dict = combo_df.to_dict('records')
                assert len(combo_dict) == 1, 'beam bad combo!'
                combo_dict = combo_dict[0]
                #loop positions start,mid,end
                for x in [0,0.5,1.0]:
                    cases += 1
                    f = beam.get_forces_at(x,combo=combo)
                    st = beam.estimate_max_stress(**f)
                    beam_fail_frac = st/beam.material.allowable_stress
                    
                    #record!
                    if beam_fail_frac >= fail_1:
                        self.debug(f'estimated beam {beam.name} failure at L={x*100:3.2f}% | {combo}')
                        failures_count += 1
                        
                        failures[nm][(combo,x)] = _d = {'beam':nm,'est_stress':st,'combo':combo,'x':x,'allowable_stress':beam.material.allowable_stress,'fail_frac':beam_fail_frac,'loads':f}
                        
                        failures[nm][(combo,x)].update(**{k.split(nm+'.')[-1] if nm in k else k:v for k,v in combo_dict.items() if ('beams' not in k or nm in k) and k not in _d})
                        
        if failures_count:
            self.warning(f'{failures_count} @ {failures_count*100/cases:3.2f}% estimated failures: {set(failures.keys())}')


        return failures
    
    def sort_max_estimated_failures(self,failures):
        """a generator to iterated through (beam,combo,x) as they return the highest fail_fraction for structures"""

        failures = {(beamnm,combo,x):dat 
                    for beamnm,estfail in failures.items() 
                    for (combo,x),dat in estfail.items()}

        for (beamnm,combo,x),dat in sorted(failures.items(),key=lambda kv:kv[1]['fail_frac'],reverse=True):
            yield (beamnm,combo,x),dat

              

    def check_failures(self,failures:dict)->bool:
        """check if any failures exist in the nested dictonary and return True if there are"""

        if 'actual' in failures and failures['actual']:
            for bm,cases in failures['actual'].items():
                for (combo,x),dat in cases.items():
                    if 'fails' in dat and dat['fails']:
                        return True
        return False


    def check_est_failures(self,failures:dict,top=True)->bool:
        """check if any failures exist in the nested dictonary and return True if there are"""
        for bm,cases in failures.items():
            if isinstance(cases,dict):
                if top: 
                    self.debug(f'checking est fail {bm}')
                for k,v in cases.items():
                    if isinstance(v,dict):
                        val = self.check_est_failures(v,top=False)
                        if val:
                            return True
                    elif v:
                        return True
            elif cases:
                return True
        return False    
    
    def run_combo_failure_analysis(self,combo,run_full:bool=False,SF:float=1.0):
        """runs a single load combo and adds 2d section failures"""
        self.run(combos=combo)
        out = {}
        failures = self.estimated_failures(combos=combo)
        out['estimate'] = failures
        out['actual'] = check_fail = {}
        #perform 2d analysis if any estimated
        if self.check_est_failures(failures):
            self.info(f'testing actual failures...')
            out['actual'] = dfail =  self.run_failure_sections(failures,run_full,SF)

            if dfail and not run_full:
                self.info(f'found actual failure...')
                return out
            
        return out
                

    def run_failure_sections(self,failures,run_full:bool=False,SF:float=1.0,fail_fast=True):
        """takes estimated failures and runs 2D section FEA on those cases"""
        secton_results = {}

        #random beam analysis
        #for beamnm,est_fail_cases in failpairs:

        for (beamnm,combo,x),dat in self.sort_max_estimated_failures(failures):
            self.info(f'running beam sections: {beamnm},{combo} @ {x} {dat["fail_frac"]*100.}%')
            beam = self.beams[beamnm]
            
            #prepare for output
            if beamnm not in secton_results:
                secton_results[beamnm] = {}
                
            f = dat['loads']
            s = beam.get_stress_with_forces(**f)

            secton_results[beamnm][(combo,x)] = d_ = dat.copy()
            d_['combo'] = combo
            d_['x'] = x
            d_['beam'] = beamnm
            d_['stress_analysis'] = s
            d_['stress_results'] =sss = s.get_stress()
            d_['stress_vm_max'] = max([max(ss['sig_vm']) for ss in sss])
            d_['fail_frac'] =ff= max([max(ss['sig_vm']/beam.material.allowable_stress) for ss in sss])
            d_['fails'] = fail = ff > 1/SF

            if fail:
                self.warning(f'beam {beamnm} failed @ {x*100:3.0f}%| {c}')
                if fail_fast:
                    return secton_results
                if not run_full:
                    break #next beam!
            else:
                self.debug(f'beam {beamnm} ok @ {x*100:3.0f}%| {c}')

        return secton_results
    
    #Remote Utility Calls (pynite frame integration)
    def merge_displacement_results(self,combo,D):
        """adds a remotely calculated displacement vector and applies to nodes ala pynite convention"""
        self.frame._D[combo] = D

        for node in self.frame.Nodes.values():
            node.DX[combo] = D[node.ID*6 + 0, 0]
            node.DY[combo] = D[node.ID*6 + 1, 0]
            node.DZ[combo] = D[node.ID*6 + 2, 0]
            node.RX[combo] = D[node.ID*6 + 3, 0]
            node.RY[combo] = D[node.ID*6 + 4, 0]
            node.RZ[combo] = D[node.ID*6 + 5, 0]

    def prep_remote_analysis(self):
        """copies PyNite analysis pre-functions"""
        # Generate all meshes
        for mesh in self.frame.Meshes.values():
            if mesh.is_generated == False:
                mesh.generate()

        # Activate all springs and members for all load combinations
        for spring in self.frame.Springs.values():
            for combo_name in self.LoadCombos.keys():
                spring.active[combo_name] = True

        # Activate all physical members for all load combinations
        for phys_member in self.frame.Members.values():
            for combo_name in self.frame.LoadCombos.keys():
                phys_member.active[combo_name] = True

        # Assign an internal ID to all nodes and elements in the model
        self._renumber()

        # Get the auxiliary list used to determine how the matrices will be partitioned
        D1_indices, D2_indices, D2 = self._aux_list()

    #Failure Analysis Interface (Override your for your cases!)
    def failure_load_combos(self,report_info:dict,run_full=False,**kw):
        """returns or yields a set of load combo names to check failure for.
        You're encouraged to override this for your specific cases
        :param report_info: the report that is generated in `run_failure_analysis` which will dynamically change with
        """
        failures = report_info['failures']

        #gravity case first
        o = ['gravity']
        yield o[0] 
        if report_info['fails']:
            if not run_full:
                return #you are weak

        for combo_name,combo in self.LoadCombos.items():
            if combo_name != 'gravity':
                yield combo_name
                cfail = failures[combo_name]
                if self.check_failures(cfail):
                    if not run_full:
                        return
                        
    def failure_load_exithandler(self,combo,res,report_info,run_full):
        """a callback that is passed the structural combo, the case results
        
        Override this function to provide custom exit capability
        :param combo: the load combo analyzed
        :param res: the combo failure results
        :param report_info: the report dictionary
        :run_full: option to run everything
        """
        if self.check_failures(res):
            self.warning(f'combo {combo} failed!')
            report_info['fails'] = True
            if not run_full:
                raise StopAnalysis()   


    def run_failure_analysis(self,SF=1.0,run_full=False,**kwargs):
        """
        Failure Determination:
        Beam Stress Estiates are used to determine if a 2D FEA beam/combo analysis should be run. The maximum beam vonmises stress is compared the the beam material allowable stress / saftey factor.

        Override:
        Depending on number of load cases, this analysis could take a while. Its encouraged to use a similar pattern of check and exit to reduce load combo size in an effort to fail early. Override this function if nessicary.
        
        Case / Beam Ordering:
        runs the gravity case first and checks for failure to exit early, 
        then all other combos are run
        :returns: a dictionary with 2D fea failures and estimated failures as well for each beam
        """

        report = {'fails':False} #innocent till proven guilty
        report['failures'] = failures = {}
        try:
            #Run the rest of the load cases
            for combo in self.failure_load_combos(report,**kwargs):
                fail_res = self.run_combo_failure_analysis(combo,run_full,SF)
                failures[combo] = fail_res
                #will call 
                self.failure_load_exithandler(combo,fail_res,report,run_full)

        
        except KeyboardInterrupt:
            return report

        except StopAnalysis:
            return report

        except Exception as e:
            self.error(e,'issue in failur analysis')
        
        return report #tada!        

@ray.remote(max_calls=5)
def remote_run_combo(struct,combo,sync):

    try:
        struct.resetSystemLogs()
        struct.info(f'start combo {combo}')
        #print(struct.logger.__dict__)
        struct.run(combos=combo)
        #sync.put(struct.table[1])
        row = struct.table[-1]
        out = {'row':row,'D':struct._D[combo],'combo':combo,'st_id':struct.run_id}
        d = sync.put(out)
        ray.wait(d)        

        del struct
        return out

    except Exception as e:
        struct.error(e,'issue with run combo')

        return e
    


def remote_combo_failure_analysis(inst,combo,run_full:bool=False,SF:float=1.0):
    """runs a single load combo and adds 2d section failures"""
    inst.resetSystemLogs()
    inst.info(f'determine combo {combo} failures...')

    out = {}
    failures = inst.estimated_failures(combos=combo)
    out['estimate'] = failures
    out['actual'] = check_fail = {}
    #perform 2d analysis if any estimated
    if inst.check_est_failures(failures):
        inst.info(f'testing estimated failures...')
        
        dfail = remote_failure_sections(inst,failures,run_full,SF)
        out['actual'] = dfail

        if dfail and not run_full:
            inst.info(f'found actual failure...')
            return out
    else:
        inst.info(f'no estimated failurs found')
        
    return out

@ray.remote
def remote_section(beamsection,beamname,forces,allowable_stress,combo,x,SF=1.,return_section=False):
    """optimized method to run 2d fea and return a failure determination"""
    #TODO: allow post processing ops
    s = beamsection.calculate_stress(**forces)


    sss = s.get_stress()
    d_ = {'loads':forces,'beam':beamname,'combo':combo,'x':x}
    if return_section:
        d_['stress_analysis'] = s
        d_['stress_results'] =sss  
    d_['stress_vm_max'] = max([max(ss['sig_vm']) for ss in sss])
    d_['fail_frac'] =ff= max([max(ss['sig_vm']/allowable_stress) for ss in sss])
    d_['fails'] = fail = ff > 1/SF
    
    return d_

def remote_failure_sections(struct,failures,run_full:bool=False,SF:float=1.0,fail_fast=True,group_size=12):
    """takes estimated failures and runs 2D section FEA on those cases"""

    struct.resetSystemLogs()

    secton_results = {}
    skip_beams = set()
    if not hasattr(struct,'_putbeam'):
        struct._putbeam = put_beams = {}
    else:
        put_beams = struct._putbeam

    cur = []

    for (beamnm,c,x),dat in struct.sort_max_estimated_failures(failures):
        struct.info(f'run remote beam sections: {beamnm},{c} @ {x} {dat["fail_frac"]*100.}%')
        if beamnm in skip_beams:
            continue

        #prep for output
        if beamnm not in secton_results:
            secton_results[beamnm] = {}
        beam = struct.beams[beamnm]

        forces = beam.get_forces_at(x,c)

        #lazy beam puts
        if beamnm not in put_beams:
            beam_ref = ray.put(beam._section_properties)
            put_beams[beamnm] = beam_ref
        else:
            beam_ref = put_beams[beamnm]

        cur.append(remote_section.remote(beam_ref,beam.name,forces,beam.material.allowable_stress,c,x,SF=SF))

        #run the damn thing
        if len(cur) >= group_size:
            struct.info(f'waiting on {group_size}')
            ray.wait(cur)

            for res in ray.get(cur):
                fail = res['fails']
                beamnm = res['beam']
                combo = res['combo']
                x = res['x']
                secton_results[res['beam']][(combo,x)] = res

                if fail:
                    struct.warning(f'beam {beamnm} failed @ {x*100:3.0f}%| {c}')
                    if fail_fast:
                        return secton_results
                    if not run_full:
                        skip_beams.add(beamnm)
                else:
                    struct.debug(f'beam {beamnm} ok @ {x*100:3.0f}%| {c}')

            cur = []
            struct.info(f'done waiting, continue...')

    #finish them!
    if cur:
        struct.info(f'waiting on {len(cur)}')
        ray.wait(cur)
        struct.info(f'done, continue...')

        for res in ray.get(cur):
            fail = res['fails']
            beamnm = res['beam']
            combo = res['combo']
            x = res['x']
            secton_results[res['beam']][(combo,x)] = res            

            if fail:
                struct.warning(f'beam {beamnm} failed @ {x*100:3.0f}%| {c}')
                if fail_fast:
                    return secton_results
                if not run_full:
                    skip_beams.add(beamnm)
            else:
                struct.debug(f'beam {beamnm} ok @ {x*100:3.0f}%| {c}')

    return secton_results

def remote_run_failure_analysis(struct,SF=1.0,run_full=False,**kw):
    """
    Failure Determination:
    Beam Stress Estiates are used to determine if a 2D FEA beam/combo analysis should be run. The maximum beam vonmises stress is compared the the beam material allowable stress / saftey factor.

    Override:
    Depending on number of load cases, this analysis could take a while. Its encouraged to use a similar pattern of check and exit to reduce load combo size in an effort to fail early. Override this function if nessicary.
    
    Case / Beam Ordering:
    runs the gravity case first and checks for failure to exit early, 
    then all other combos are run
    :returns: a dictionary with 2D fea failures and esimated failures as well for each beam
    """

    report = {'fails':False} #innocent till proven guilty
    report['failures'] = failures = {}

    try:
        #Run the rest of the load cases
        for combo in struct.failure_load_combos(report,run_full,**kw):

            #TODO: handle running list of combos to do simultanious running
            
            #run and store data
            struct.run(combos=combo)
            #struct_ref = ray.put(struct)

            #check failures
            fail = remote_combo_failure_analysis(struct,combo,run_full,SF)
            
            failures[combo] = fail
            struct.failure_load_exithandler(combo,fail,report,run_full)
    
    except KeyboardInterrupt:
        return report

    except Exception as e:
        struct.error(e,'issue in failur analysis')
    
    return report #tada!

@ray.remote
class RemoteStructuralAnalysis:
    """represents a stateful object in a ray cloud contetxt"""

    def __init__(self,structure,*kw):
        self.struct = structure
        self.struct.resetSystemLogs()

        self.struct.prep_remote_analysis()

    def get(self,run_id=None):
        self.struct.info(f'getting {len(self.table)}')
        if run_id is None:
            return self.struct._table
        else:
            return [v for v in self.table if v['st_id'] == run_id]

    def put(self,inv):
        """add the data from remotely running the load combo"""
        self.struct.info(f'adding {len(self.struct.table)+1}')
        self.struct._table[len(self.struct.table)+1] = inv['row']
        self.struct.merge_displacement_results(inv['combo'],inv['D'])

    def get_beam_forces(self,beam_name,combo,x):
        #BUG: .remote() not working in some remote method??
        beam = self.struct.beams[beam_name]
        return beam.get_forces_at(x,combo)

    def return_structure(self):
        return self.struct

    
#     def run_failure_analysis(self,SF=1.0,run_full=False,**kw):
#         """
#         Failure Determination:
#         Beam Stress Estiates are used to determine if a 2D FEA beam/combo analysis should be run. The maximum beam vonmises stress is compared the the beam material allowable stress / saftey factor.
# 
#         Override:
#         Depending on number of load cases, this analysis could take a while. Its encouraged to use a similar pattern of check and exit to reduce load combo size in an effort to fail early. Override this function if nessicary.
#         
#         Case / Beam Ordering:
#         runs the gravity case first and checks for failure to exit early, 
#         then all other combos are run
#         :returns: a dictionary with 2D fea failures and esimated failures as well for each beam
#         """
# 
#         report = {'fails':False} #innocent till proven guilty
#         report['failures'] = failures = {}
# 
#         try:
#             #Run the rest of the load cases
#             for combo in self.struct.failure_load_combos(report,run_full,**kw):
#                 
#                 #run and store data
#                 self.struct.run(combos=combo)
#                 #struct_ref = ray.put(self.struct)
# 
#                 #check failures
#                 fail = remote_combo_failure_analysis(self.struct,combo,run_full,SF)
#                 
#                 failures[combo] = fail
#                 self.struct.failure_load_exithandler(combo,fail,report,run_full)
#         
#         except KeyboardInterrupt:
#             return report
# 
#         except Exception as e:
#             self.struct.error(e,'issue in failur analysis')
#         
#         return report #tada!

