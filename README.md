# ottermaticslib
A library to tabulate information from complex systems with various ways to store data and act as glue code for tough engineering problems.

### Installation
```bash
pip install git+https://github.com/Ottermatics/ottermaticslib.git
```

### Core Functions
1. Tabulation Of Complex Systems
2. Modular Post Processing (dataframes)
3. Exploratory Analysis (ipython + functions / docs)
4. Workflows for core engineering problemes (structures + cost, thermal+fluids solve)

### MVP Features (WIP)
1. Tabulation, use `attrs.field` and `system_property` to capture `y=f(x)` [Done]
2. Dynamic Programing ensures work is only done when new data is available with `cached_system_property`. [Done]
3. Quick Calculation provided by direct cached references to attribues and properties [Done]
4. Solver based on `NPSS` strategy of balances and integrators [Done]
5. Reporting to google sheets, csv and excel. 

### Example Engineering Problems:
These problems demonstrate functionality

#### Air Filter
run a throttle sweep with filter loss characteristic and fan afinity law based pressure based off of a design point.
```python
@otterize
class Fan(Component):

    n:float = attrs.field(default=1)
    dp_design= attrs.field(default=100)
    w_design = attrs.field(default=2)


    @system_property
    def dP_fan(self) -> float:
        return self.dp_design*(self.n*self.w_design)**2.0
    
@otterize
class Filter(Component):

    w:float = attrs.field(default=0)
    k_loss:float = attrs.field(default=50)

    @system_property
    def dP_filter(self) -> float:
        return self.k_loss*self.w

@otterize
class Airfilter(System):

    throttle:float = attrs.field(default=1)
    w:float = attrs.field(default=1)
    k_parasitic:float = attrs.field(default=0.1)

    fan: Fan = SLOT.define(Fan)
    filt: Filter = SLOT.define(Filter)

    set_fan_n = SIGNAL.define('fan.n','throttle',mode='both')
    set_filter_w = SIGNAL.define('filt.w','w',mode='both')

    flow_solver = SOLVER.define('sum_dP','w')
    flow_solver.add_constraint('min',0)

    flow_curve = PLOT.define(
        "throttle", "w", kind="lineplot", title="Flow Curve"
    )    

    @system_property
    def dP_parasitic(self) -> float:
        return self.k_parasitic * self.w**2.0

    @system_property
    def sum_dP(self) -> float:
        return self.fan.dP_fan - self.dP_parasitic - self.filt.dP_filter


#Run the system
from ottermatics.logging import change_all_log_levels
from matplotlib.pylab import *

fan = Fan()
filt = Filter()
af = Airfilter(fan=fan,filt=filt)

af.run(throttle=list(np.arange(0.1,1.1,0.1)))

df = af.dataframe

fig,(ax,ax2) = subplots(2,1)
ax.plot(df.throttle*100,df.w,'k--',label='flow')
ax2.plot(df.throttle*100,filt.dataframe.dp_filter,label='filter')
ax2.plot(df.throttle*100,df.dp_parasitic,label='parasitic')
ax2.plot(df.throttle*100,fan.dataframe.dp_fan,label='fan')
ax.legend(loc='upper right')
ax.set_title('flow')
ax.grid()
ax2.legend()
ax2.grid()
ax2.set_title(f'pressure')
ax2.set_xlabel(f'throttle%')
```

##### Results
![air_filter_calc.png](media/air_filter_calc.png)


#### Spring Mass Damper
##### Overview
Test case results in accurate resonance frequency calculation
```python
@otterize
class SpringMass(System):

    k:float = attrs.field(default=50)
    m:float = attrs.field(default=1)
    g:float = attrs.field(default=9.81)
    u:float = attrs.field(default=0.3)

    a:float = attrs.field(default=0)
    x:float = attrs.field(default=0.0)
    v:float = attrs.field(default=0.0)
    t:float = attrs.field(default=0.0)

    x_neutral:float = attrs.field(default=0.5)
    
    #a is solved for to ensure sumF is zero
    res = SOLVER.define('sumF','a')

	#a is integrated to provide v, similar to v integrated to supply x
    vtx = TRANSIENT.define('v','a')
    xtx = TRANSIENT.define('x','v')

    @system_property
    def dx(self)-> float:
        return self.x_neutral- self.x 

    @system_property
    def Fspring(self)-> float:
        return self.k * self.dx
    
    @system_property
    def Fgrav(self)-> float:
        return self.g * self.m
    
    @system_property
    def Faccel(self)-> float:
        return self.a * self.m
    
    @system_property
    def Ffric(self)->float:
        return self.u*self.v

    @system_property
    def sumF(self) -> float:
        return self.Fspring - self.Fgrav - self.Faccel - self.Ffric


#Run The System, Compare damping `u`=0 & 0.1
sm = SpringMass(x=0.0)
sm.run(dt=0.01,endtime=10,u=[0.0,0.1])

df = sm.dataframe
df.groupby('run_id').plot('time','x')
```

##### Results Damping Off
![olib_spring_mass_clac.png](media/olib_spring_mass_clac.png)

##### Results - Damping On
<img src="media/olib_spring_mass_clac_damp 1.png" />


## Reporting & Analysis
`Analysis` is capable of tabulation as a `Component` or `System` and wraps a top level `System` and will save data for each system interval. `Analysis` stores several reporters for tables and plots that may be used to store results in multiple locations.

Reporting is supported for tables via dataframes in CSV,Excel and Gsheets (WIP).

For plots reporting is supported in disk storage.

```python

from ottermatics.analysis import Analysis
from ottermatics.reporting import CSVReporter,DiskPlotReporter
from ottermatics.properties import system_property
import numpy as np
import os,pathlib

this_dir = str(pathlib.Path(__file__).parent)
this_dir = os.path.join(this_dir,'airfilter_report')
if not os.path.exists(this_dir):
    os.path.mkdir(this_dir)

csv = CSVReporter(path=this_dir,report_mode='daily')
csv_latest = CSVReporter(path=this_dir,report_mode='single')

plots = DiskPlotReporter(path=this_dir,report_mode='monthly')
plots_latest = DiskPlotReporter(path=this_dir,report_mode='single')


@otterize
class AirfilterAnalysis(Analysis):
    """Does post processing on a system"""
    
    efficiency = attrs.field(defualt=0.95)

    @system_property
    def clean_air_delivery_rate(self) -> float:
        return self.system.w*self.efficiency

    def post_process(self,*run_args,**run_kwargs):
        pass
        #TODO: something custom!

#Air Filter as before
fan = Fan()
filt = Filter()
af = Airfilter(fan=fan,filt=filt)

#Make The Analysis
sa = AirfilterAnalysis(
                    system = af,
                    table_reporters = [csv,csv_latest],
                    plot_reporters = [plots,plots_latest]
                   )

#Run the analysis! Input passed to system
sa.run(throttle=list(np.arange(0.1,1.1,0.1)))

#CSV's & Plots available in ./airfilter_report!

```




## Documentation:
https://ottermatics.github.io/ottermaticslib/build/html/index.html


