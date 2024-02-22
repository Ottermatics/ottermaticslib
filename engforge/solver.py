"""solver defines a SolverMixin for use by System.

Additionally the SOLVER attribute is defined to add complex behavior to a system as well as add constraints and transient integration.
"""

import attrs
import uuid
import numpy
import scipy.optimize as scopt
from contextlib import contextmanager
import copy
import datetime

#from engforge.dynamics import DynamicsMixin
from engforge.properties import *
from engforge.solveable import SolveableMixin,refmin_solve,refset_get,refset_input
import itertools,collections

SOLVER_OPTIONS = ["root", "minimize"]


class SolverLog(LoggingMixin):
    pass


log = SolverLog()
class SolverMixin(SolveableMixin):
    _run_id: str = None
    _solved = None
    _trans_opts = None

    # TODO: implement custom solver with unified derivative equation.
    # _system_jacobean = None
    # _last_X = None
    # _current_X = None
    # _last_F = None
    # _current_F = None

    _converged = False
    # solver_option = attrs.field(default='root',validator=attrs.validators.in_(SOLVER_OPTIONS))
    solver_option = "root"

    # Configuration Information
    @property
    def solved(self):
        return self._solved

    def run(self,*args,**kwargs):
        """the steady state run method for the system. It will run the system with the input parameters and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""

        self._iterate_input_matrix(self._run,*args,**kwargs)

    def _run(self,refs,icur,eval_kw=None,sys_kw=None,*args,**kwargs):
        """the steady state run method for the system. It will run the system with the input parameters and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""
        
        #TODO: what to do with eval / sys kw
        #TODO: option to preserve state
        refset_input(refs,icur)
        self.info(f"running with {icur}|{kwargs}")
        self.run_method(*args,**kwargs)
        self.debug(f"{icur} run time: {self._run_time}")

    def run_method(self,eval_kw=None,sys_kw=None,cb=None):
        """runs the case as currently defined by the system. This is the method that is called by the solver matrix to run the system with the input parameters. It will run the system with the input parameters and return the system with the results. Dynamics systems will be run so they are in a steady state nearest their initial position."""
        #set the values
        from engforge.system import System

        # Transeint
        self._run_start = datetime.datetime.now()
        if isinstance(self,System):
            self.system_references(recache=True) #recache is important for iterators, #TODO: only recache when iterators are present
        #steady state
        if self._run_id is None:
            self._run_id = int(uuid.uuid4())
        #Recache system references
        self.evaluate(cb=cb,eval_kw=eval_kw,sys_kw=sys_kw)

        self._run_end = datetime.datetime.now()
        self._run_time = self._run_end - self._run_start
                
    
    def post_run_callback(self,**kwargs):
        """user callback for when run is complete"""
        pass

    def pre_run_callback(self,**kwargs):
        """user callback for when run is beginning"""
        pass    

    # def simulate(self, dt, N, cb=None,eval_kw=None,sys_kw=None):
    #     """integrates the time series over N points at a increment of dt"""
    #     for i in range(N):
    #         #TODO: adapt timestep & integrate ODE system integrator
    #         # Run The SS Timestep
    #         self.evaluate(cb=cb)
    #         # Run Integrators
    #         for k, v in self.transients.items():
    #             v.integrate(dt)
    #         # Mark Time
    #         self.time = self.time + dt

    def run_internal_systems(self,sys_kw=None):
        """runs internal systems with potentially scoped kwargs"""
        # Pre Execute Which Sets Fields And PRE Signals
        from engforge.system import System
        
        #Record any changed state in components here, important for iterators
        if isinstance(self,System):
            self.system_references(recache=True)                              
        
        #System Solver Loop
        for key, comp in self.internal_systems().items():
            #self.debug(f"checking sys {key}.{comp}")
            if sys_kw and key in sys_kw:
                sys_kw_comp = sys_kw[key]
            else:
                sys_kw_comp = {}   
                         
            #Systems solve cycle
            if isinstance(comp, System): #should always be true
                self.info(f"solving {key} with {sys_kw_comp}")
                comp.evaluate(**sys_kw_comp)

    # Single Point Flow
    def evaluate(self, cb=None,eval_kw:dict=None,sys_kw:dict=None, *args,**kw):
        """Evaluates the system with additional inputs for execute()
        :param cb: an optional callback taking the system as an argument of the form (self,eval_kw,sys_kw,*args,**kw)
        """

        if self.log_level < 20:
            self.debug(f"running with X: {self.X} & args:{args} kw:{kw}")

        #0. run pre execute (signals=pre,both)
        self.pre_execute()

        # prep index
        self.index += 1

        #1.Runs The Solver
        try:
            out = self.execute( *args,**kw)
        except Exception as e:
            self.error(e, f"solver failed @ {self.X}")
            out = None
            raise e

        #2. Update Internal Elements
        self.update_internal(eval_kw=eval_kw,*args,**kw)
        
        #run components and systems recursively
        self.run_internal_systems(sys_kw=sys_kw)

        # Post Execute (signals=post,both)
        self.post_execute()

        #Post Update Each Internal System
        self.post_update_internal(eval_kw=eval_kw,*args,**kw)
             
        # Save The Data
        self.save_data(index=self.index)

        if cb:
            cb(self,eval_kw,sys_kw,*args,**kw)

        return out

    def pre_execute(self):
        """runs the solver of the system"""
        if self.log_level <= 10:
            self.msg(f"pre execute")

        # TODO: set system fields from input
        for signame, sig in self.signals.items():
            if sig.mode == "pre" or sig.mode == "both":
                sig.apply()

    def post_execute(self):
        """runs the solver of the system"""
        if self.log_level <= 10:
            self.msg(f"post execute")
            
        for signame, sig in self.signals.items():
            if sig.mode == "post" or sig.mode == "both":
                sig.apply()

    def execute(self, *args,**kw):
        """Solves the system's system of constraints and integrates transients if any exist

        Override this function for custom solving functions, and call `system_solver` to use default solver functionality.

        :returns: the result of this function is returned from evaluate()
        """
        self.system_solver()

    def system_solver(self):
        #TODO: use ref algorithm to solve system
        #TODO: add transient balances to the system with constraints ect.
        # Check independents to run system
        if not self.X.size > 0:
            if self.log_level < 10:
                self.debug(f"nothing to solve...")
            return

        self.debug(f"running system solver")

        # solver objective functions
        def f(*x):
            res = self.calcF(x[0], pre_execute=True)
            return res

        def f_min(*x):
            res = self.calcF(x[0], pre_execute=True)
            ans = numpy.linalg.norm(res, 2)

            if ans < 1:
                return ans**0.5
            return ans

        if self.solver_option == "root" and not self.has_constraints:
            self._ans = scopt.root(f, x0=self.X)
            if self._ans.success:
                self.setX(self._ans.x, pre_execute=True)
                self._converged = True
            else:
                self._converged = False
                raise Exception(f"solver didnt converge: {self._ans}")

            return self._ans

        elif self.solver_option == "minimize" or self.has_constraints:
            cons = self.solver_constraints
            opts = {"rhobeg": 0.01, "catol": 1e-4, "tol": 1e-6}
            self._ans = scopt.minimize(
                f_min, x0=self.X, method="COBYLA", options=opts, **cons
            )
            if self._ans.success and abs(self._ans.fun) < 0.01:
                self.setX(self._ans.x)
                self._converged = True

            elif self._ans.success:
                self.setX(self._ans.x)
                self.warning(
                    f"solver didnt fully solve equations! {self._ans.x} -> residual: {self._ans.fun}"
                )
                self._converged = False

            else:
                self._converged = False
                raise Exception(f"solver didnt converge: {self._ans}")

            return self._ans

        else:
            self.warning(f"no solution attempted!")

    def _iterate_components(self):
        """sets the current component for each product combination of iterable_components"""

        components = self.iterable_components

        if not components:
            yield  # enter once
        else:

            def _gen(gen, compkey):
                for itemkey, item in gen:
                    yield compkey, itemkey

            iter_vals = {
                cn: _gen(comp._item_gen(), cn)
                for cn, comp in components.items()
            }

            for out in itertools.product(*list(iter_vals.values())):
                for ck, ikey in out:
                    # TODO: progress bar or print location
                    components[ck].current_item = ikey
                yield out

            # finally reset the data!
            for ck, comp in components.items():
                comp.reset()

    def setX(self, x_ordered=None, pre_execute=True, **x_kw):
        """Apply full X data to X. #TODO allow partial data"""
        # assert len(input_values) == len(self._X), f'bad input data'
        assert x_ordered is not None or x_kw, f"have to have some input"
        if x_ordered is not None:
            if isinstance(x_ordered, numpy.ndarray):
                x_ordered = x_ordered.tolist()
            assert not x_kw, f"there must only be ordered inputs or keyword"
            assert len(x_ordered) == len(self._X), f"must be a full sized"
            for f, x in zip(self._X.values(), x_ordered):
                f.set_value(x)
        else:
            assert set(x_kw).issubset(set(self._X)), f"incorrect parms {x_kw}"
            _Xref = self._X

            for k, v in x_kw.items():
                self.debug(f"setting X value {k}={v}")
                _Xref[k].set_value(v)

        if pre_execute:
            self.pre_execute()

    # Function Calculation
    def calcF(self, x_ordered=None, pre_execute=True, **x_kw):
        """calculates the orderdered or keyword input per Ref in `_F`"""
        assert x_ordered is not None or x_kw, f"have to have some input"
        with self.revert_X() as rx:
            if x_ordered is not None:
                # print(x_ordered)
                if isinstance(x_ordered, numpy.ndarray):
                    x_ordered = x_ordered.tolist()
                assert not x_kw, f"there must only be ordered inputs or keyword"
                assert len(x_ordered) == len(
                    self._X
                ), f"must be a full sized input"
                # Set Em
                self.setX(x_ordered, pre_execute=pre_execute)
                # Get Em
                o = [f.value() for f, x in zip(self.Flist, x_ordered)]
                # o = [f.value() for f  in self.Flist]
                return numpy.array(o)
            elif x_kw:
                assert (
                    set(x_kw) in self._X
                ), f"incompatable keyword args in {x_kw}"
                # Set Em
                self.setX(**x_kw, pre_execute=pre_execute)
                # Get Em
                o = [self._F[k].value(v) for k, v in x_kw.items()]
                return numpy.array(o)
        # Forget Em

        # raise NotImplemented('#TODO')

    def calcJacobean(self, x, pct=0.001, diff=0.0001):
        """
        returns the jacobiean by modifying X' <= X*pct + diff and recording the differences. When abs(x) < pct x' = x*1.1 + diff
        """
        with self.revert_X():
            # initalize here
            self.setX(x)
            self.pre_execute()

            rows = []
            dxs = []
            Fbase = self.F
            for k, v in self._X.items():
                x = v.value()
                if abs(x) > pct:
                    new_x = x * (1 + pct) + diff
                else:
                    new_x = x * (1.1) + diff * x
                dx = new_x - x
                dxs.append(dx)

                v.set_value(new_x)
                self.pre_execute()

                F_ = self.F - Fbase

                rows.append(F_ / dx)

        return numpy.column_stack(rows)

    # Useful Properties (F & X)
    @instance_cached
    def Flist(self):
        """returns F() for each solver dependent as an anonymous function"""
        return [self._F[self.F_keyword_order[i]] for i in range(len(self._F))]

    @instance_cached
    def _X(self):
        """stores the internal references to the solver references"""
        return {k: v.independent for k, v in self.solvers.items()}

    @instance_cached
    def _F(self):
        """stores the internal references to the solver references"""
        return {k: v.dependent for k, v in self.solvers.items()}

    @instance_cached
    def F_keyword_order(self):
        """defines the order of inputs in ordered mode for calcF"""
        return {i: k for i, k in enumerate(self.solvers)}

    @instance_cached
    def F_keyword_rev_order(self):
        """defines the order of inputs in ordered mode for calcF"""
        return {k: i for i, k in self.F_keyword_order.items()}

    @property
    def X(self) -> numpy.array:
        """The current state of the system"""
        return numpy.array([v.value() for v in self._X.values()])

    @property
    def F(self) -> numpy.array:
        """The current solution to the system"""
        return numpy.array([v.value() for v in self._F.values()])

    @contextmanager
    def revert_X(self, *ing):
        """
        Stores the _X parameter at present, the reverts to that state when done
        #TODO: add pre_execute / post_execute options
        """
        X_now = self.X
        try:  # Change Variables To Input
            yield self
        finally:
            self.setX(X_now)
            self.pre_execute()

    # Solution Managment
    @instance_cached
    def has_constraints(self):
        """checks for any active constrints"""
        for solvname, solv in self.solvers.items():
            soltype = solv.solver
            if any([s is not None for s in soltype.constraints.values()]):
                return True
        return False

    @instance_cached
    def constraints(self):
        """returns a list of constraitns by type"""
        out = []
        for solvname, solv in self.solvers.items():
            soltype = solv.solver
            for stype, sv in soltype.constraints.items():
                if sv is None:
                    continue
                out.append(
                    {
                        "name": solvname,
                        "type": stype,
                        "value": sv,
                        "solver": soltype,
                        "independent": soltype.independent,
                        "solv_instance": solv,
                    }
                )
        return out

    @instance_cached
    def solver_constraints(self):
        """formatted as arguments for the solver"""
        out = {"bounds": [], "constraints": []}

        # boudns must be X length wide
        for parm, solv in self.solvers.items():
            out["bounds"].append([None, None])

        groups = {}
        for const in self.constraints:
            name = const["name"]
            if name not in groups:
                groups[name] = [const]
            else:
                groups[name].append(const)

        for group, values in groups.items():
            for vcon in values:
                con = self.create_constraint(vcon, out, group)
                out["constraints"].append(con)
        return out

    def create_constraint(self, vcon, out, group):
        contype = vcon["type"]
        value = vcon["value"]
        x_inx = self.F_keyword_rev_order[vcon["name"]]
        # print(group,contype,value,x_inx,isinstance(value,(int,float)))

        if isinstance(value, (int, float)):
            # its a number
            val = float(value)
            if contype == "max":
                # make objective that is negative when x > lim
                def fun(x):
                    return val - x[x_inx]

            else:

                def fun(x):
                    return x[x_inx] + val

            cons = {"type": "ineq", "fun": fun}
            return cons
        else:
            val = copy.copy(value)
            # its a function
            if contype == "max":
                # make objective that is negative when x > lim
                def fun(x):
                    with self.revert_X():
                        self.setX(x)
                        return val(self) - x[x_inx]

            else:

                def fun(x):
                    with self.revert_X():
                        self.setX(x)
                        return x[x_inx] - val(self)

            cons = {"type": "ineq", "fun": fun}
            return cons

    #Replaces Tabulation Method
    @solver_cached
    def data_dict(self):
        """records properties from the entire system, via system references cache"""
        out = collections.OrderedDict()
        sref = self.system_references()
        for k, v in sref["attributes"].items():
            val = v.value()
            if isinstance(val, TABLE_TYPES):
                out[k] = val
            else:
                out[k] = numpy.nan
        for k, v in sref["properties"].items():
            val = v.value()
            if isinstance(val, TABLE_TYPES):
                out[k] = val
            else:
                out[k] = numpy.nan
        return out

