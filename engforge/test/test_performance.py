

#TODO:
#1. ensure total evaluation time is not in Refs
#2. minimize call redundancy (how)
#3. validate caching performance and safe reset of values

from engforge.test.solver_testing_components import *
from engforge.logging import LoggingMixin
from cProfile import Profile
from pstats import Stats
import unittest
import time

class PerfTest(LoggingMixin): pass
log = PerfTest()

def eval_steady_state(dxdt=0):
    """test that the ss answer is equal to the result with damping"""
    sm = SpringMass()
    sm.run(dxdt=dxdt,combos='time')
    df = sm.dataframe

def eval_transient(endtime=10,dt=0.001,run_solver=False):
    """test that the ss answer is equal to the result with damping"""
    sm = SpringMass()
    df = sm.simulate(dt=dt, endtime=endtime,run_solver=run_solver)

class TestPerformance(unittest.TestCase):

    def test_steady_state(self):
        
        with Profile() as pr:
            start = time.time()
            eval_steady_state()
            
        stats = Stats(pr)
        stats.sort_stats('tottime').print_stats(10)
        
        end = time.time()
        self.assertLessEqual((end-start), 1.0)

    def test_transient(self):
        endtime = 10
        with Profile() as pr:
            start = time.time()
            eval_transient(endtime=endtime,dt=0.001,run_solver=False)
            
        stats = Stats(pr)
        stats.sort_stats('tottime').print_stats(10)
        end = time.time()
        self.assertLessEqual((end-start),endtime)





if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ss", "--steady-state", action="store_true", dest="steady_state", help="run the steady state test")
    parser.add_argument("-tr",'--transient', action="store_true", dest="transient", help="run the transient test")
    parser.add_argument("-a", "--all", action="store_true", dest="all", help="run all tests")
    parser.add_argument("-b","--base", action="store_true", help="run all tests")    

    args = parser.parse_args()

    run_all = True if not sys.argv else args.all
    
    if not args.base:
        log.info(f'running all: {run_all} | {sys.argv}')
        if args.steady_state or run_all:
            log.info(f'running steady state test')
            eval_steady_state()
        if args.transient or run_all:
            log.info(f'running transient integration test')
            eval_transient()