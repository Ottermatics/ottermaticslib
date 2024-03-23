"""test transient cases and match against real world results"""
import unittest

from engforge import *
from engforge.attr_plotting import *
from engforge.test.solver_testing_components import SpringMass

from scipy.optimize import curve_fit, least_squares
import numpy as np
from matplotlib.pyplot import *

import attrs

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

class TestDynamics(unittest.TestCase):

    def test_sim(self):
        """tests free undamped motion"""

        sm = SpringMass(Fa=0,u=0)

        #the analytical answer
        self.w_ans = np.sqrt(sm.k / sm.m)

        dt = 0.001
        #FIXME: input on simulate not working 

        df = sm.simulate(dt=dt, endtime=10,run_solver=False)
        #TODO: add passing flag to context
        #self.assertTrue(sm.solved)
        #self.assertTrue(sm.converged)
        
        X = df.x
        T = df.time
        t = T[T < 2]
        x = X[T < 2]

        p0 = (x.max() - x.min(), self.w_ans + 0.01, x.mean(), -3.14)
        # ans,cov = curve_fit(fit,t,x,p0=p0)
        ans = least_squares(
            ls,
            p0,
            args=(t, x),
            ftol=1e-14,
            jac="cs",
            xtol=1e-14,
            gtol=1e-14,
            x_scale=2,
        )
        if not ans.success:
            raise Exception("failure")

        self.assertAlmostEqual(ans.x[1], self.w_ans, delta=0.01)


    def test_damping(self):
        """test that the ss answer is equal to the result with damping"""
        sm = SpringMass(Fa=0,u=5)
        sm.run(dxdt=0)

        dfss =sm.dataframe
        df = sm.simulate(dt=0.001, endtime=10,run_solver=False)

        self.assertAlmostEqual(df.iloc[-1].x,dfss.iloc[0].x, delta=0.01)

    def test_forcing(self):
        """test that the ss answer is equal to the result with damping"""
        sm = SpringMass(Fa=10,u=0,wo_f=1)
        sm.wo_f = w_ans = np.sqrt(sm.k / sm.m) * 0.1 #very different frequency

        sm.run(dxdt=0)
        dfss =sm.dataframe
        sm.x = dfss.iloc[0].x #no residual input
        
        
        df = sm.simulate(dt=0.001, endtime=10,run_solver=False)

        X = df.x
        T = df.time
        t = T[T < 2]
        x = X[T < 2]

        p0 = (x.max() - x.min(), w_ans + 0.01, x.mean(), -3.14)
        # ans,cov = curve_fit(fit,t,x,p0=p0)
        ans = least_squares(
            ls,
            p0,
            args=(t, x),
            ftol=1e-14,
            jac="cs",
            xtol=1e-14,
            gtol=1e-14,
            x_scale=2,
        )
        if not ans.success:
            raise Exception("failure")

        self.assertAlmostEqual(ans.x[1], w_ans, delta=0.01)       
