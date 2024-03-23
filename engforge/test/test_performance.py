

#TODO:
#1. ensure total evaluation time is not in Refs
#2. minimize call redundancy (how)
#3. validate caching performance and safe reset of values

from engforge.test.solver_testing_components import *
from engforge.logging import LoggingMixin

class PerfTest(LoggingMixin): pass
log = PerfTest()

def test_steady_state():
    """test that the ss answer is equal to the result with damping"""
    sm = SpringMass()
    sm.run(dxdt=0)
    df = sm.dataframe

def test_transient():
    """test that the ss answer is equal to the result with damping"""
    sm = SpringMass()
    df = sm.simulate(dt=0.001, endtime=10,run_solver=False)


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
            test_steady_state()
        if args.transient or run_all:
            log.info(f'running steady state test')
            test_transient()