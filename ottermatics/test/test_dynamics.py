"""test transient cases and match against real world results"""
import unittest

from engforge.configuration import forge
from engforge.system import System
from engforge.components import Component
from engforge.solver import SOLVER, TRANSIENT
from engforge.signals import SIGNAL
from engforge.slots import SLOT
from engforge.properties import *
from engforge.plotting import *

from scipy.optimize import curve_fit, least_squares
import numpy as np
from matplotlib.pyplot import *

import attrs


@forge
class SpringMass(System):
    k: float = attrs.field(default=50)
    m: float = attrs.field(default=1)
    g: float = attrs.field(default=9.81)
    u: float = attrs.field(default=0.3)

    a: float = attrs.field(default=0)
    x: float = attrs.field(default=0.0)
    v: float = attrs.field(default=0.0)
    t: float = attrs.field(default=0.0)

    x_neutral: float = attrs.field(default=0.5)

    res = SOLVER.define("sumF", "a")

    vtx = TRANSIENT.define("v", "a")
    xtx = TRANSIENT.define("x", "v")

    pos = TRACE.define(y="x", y2=["v", "a"])

    @system_property
    def dx(self) -> float:
        return self.x_neutral - self.x

    @system_property
    def Fspring(self) -> float:
        return self.k * self.dx

    @system_property
    def Fgrav(self) -> float:
        return self.g * self.m

    @system_property
    def Faccel(self) -> float:
        return self.a * self.m

    @system_property
    def Ffric(self) -> float:
        return self.u * self.v

    @system_property
    def sumF(self) -> float:
        return self.Fspring - self.Fgrav - self.Faccel - self.Ffric


class TestDynamics(unittest.TestCase):
    def setUp(self) -> None:
        self.sm = SpringMass()

        # the analytical answer
        self.w_ans = np.sqrt(self.sm.k / self.sm.m)

    def test_sim(self):
        dt = 0.001
        self.sm.run(dt=dt, u=0, endtime=10)
        self.assertTrue(self.sm.solved)
        self.assertTrue(self.sm.converged)

        df = self.sm.dataframe
        X = df.x
        T = df.t
        t = T[T < 2]
        x = X[T < 2]

        fit = lambda t, a, b, w: a * np.sin((2 * np.pi / w) * t) + b
        jac = lambda t, a, b, w: np.array(
            [
                np.sin(w * t),
                1 * np.ones(t.shape),
                a * w * np.cos(w * t),
                0 * np.ones(t.shape),
            ]
        ).T
        ls = lambda x, t, y: (x[0] * np.sin(x[1] * t + x[3]) + x[2]) - y
        f = lambda x, t, y: (x[0] * np.sin(x[1] * t + x[3]) + x[2])
        p0 = (x.max() - x.min(), self.w_ans + 0.01, x.mean(), -3.14)
        # ans,cov = curve_fit(fit,t,x,p0=p0)
        ans = least_squares(
            ls,
            p0,
            args=(t, x),
            ftol=1e-12,
            jac="cs",
            xtol=1e-12,
            gtol=1e-12,
            x_scale=2,
        )
        if not ans.success:
            raise Exception("failure")

        self.assertAlmostEqual(ans.x[1], self.w_ans, delta=0.01)

        # fig,ax = subplots()
        # ax.plot(t,x,label='data')
        # ax.plot(t,f(p0,t,x),label='guess')
        # ax.plot(t,f(ans.x,t,x),label='fit')
        # legend()
