"""tests airfilter system operation by solving for airflow between filter and and fan"""
import unittest

from ottermatics.configuration import otterize
from ottermatics.system import System
from ottermatics.components import Component
from ottermatics.solver import SOLVER, TRANSIENT
from ottermatics.signals import SIGNAL
from ottermatics.slots import SLOT
from ottermatics.properties import *
from ottermatics.plotting import *
from ottermatics.analysis import Analysis

from scipy.optimize import curve_fit, least_squares
import numpy as np
from matplotlib.pyplot import *

import attrs


@otterize
class Fan(Component):
    n: float = attrs.field(default=1)
    dp_design = attrs.field(default=100)
    w_design = attrs.field(default=2)

    @system_property
    def dP_fan(self) -> float:
        return self.dp_design * (self.n * self.w_design) ** 2.0


@otterize
class Filter(Component):
    w: float = attrs.field(default=0)
    k_loss: float = attrs.field(default=50)

    @system_property
    def dP_filter(self) -> float:
        return self.k_loss * self.w


@otterize
class Airfilter(System):
    throttle: float = attrs.field(default=1)
    w: float = attrs.field(default=1)
    k_parasitic: float = attrs.field(default=0.1)

    fan: Fan = SLOT.define(Fan, default_ok=True)
    filt: Filter = SLOT.define(Filter, default_ok=True)

    set_fan_n = SIGNAL.define("fan.n", "throttle", mode="both")
    set_filter_w = SIGNAL.define("filt.w", "w", mode="both")

    flow_solver = SOLVER.define("sum_dP", "w")
    flow_solver.add_constraint("min", 0)

    flow_curve = PLOT.define(
        "throttle", "w", kind="lineplot", title="Flow Curve"
    )

    @system_property
    def dP_parasitic(self) -> float:
        return self.k_parasitic * self.w**2.0

    @system_property
    def sum_dP(self) -> float:
        return self.fan.dP_fan - self.dP_parasitic - self.filt.dP_filter

    @system_property
    def dp_positive(self) -> float:
        return self.dP_parasitic - self.filt.dP_filter


@otterize
class AirFilterAnalysis(Analysis):
    efficiency: float = attrs.field(default=0.9)
    system = SLOT.define(Airfilter, default_ok=True)

    @system_property
    def cadr(self) -> float:
        return self.system.w * self.efficiency


class TestFilterSystem(unittest.TestCase):
    def setUp(self):
        self.af = Airfilter()

    def test_plot(self):
        self.af.run(throttle=np.linspace(0, 1, 10))
        fig = self.af.flow_curve()
        self.assertIsNotNone(fig)


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.af = AirFilterAnalysis()

    def test_plot(self):
        self.af.run(throttle=np.linspace(0, 1, 10))
        fig = self.af.system.flow_curve()
        ofig = self.af._stored_plots["airfilteranalysis.airfilter.flow_curve"]
        print(fig)
        print(ofig)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ofig)


# from ottermatics.logging import change_all_log_levels
# from ottermatics.test.test_airfilter import *
# from matplotlib.pylab import *
#
# fan = Fan()
# filt = Filter()
# af = Airfilter(fan=fan, filt=filt)
#
# af.run(throttle=list(np.arange(0.1, 1.1, 0.1)))
#
# df = af.dataframe
#
# fig, (ax, ax2) = subplots(2, 1)
# ax.plot(df.throttle * 100, df.w, "k--", label="flow")
# ax2.plot(df.throttle * 100, filt.dataframe.dp_filter, label="filter")
# ax2.plot(df.throttle * 100, df.dp_parasitic, label="parasitic")
# ax2.plot(df.throttle * 100, fan.dataframe.dp_fan, label="fan")
# ax.legend(loc="upper right")
# ax.set_title("flow")
# ax.grid()
# ax2.legend()
# ax2.grid()
# ax2.set_title(f"pressure")
# ax2.set_xlabel(f"throttle%")
