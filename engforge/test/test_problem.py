import unittest

"""
#TODO: make this context test
from engforge.problem_context import *
from engforge.system import System,forge
import random


@forge(auto_attribs=True)
class Test(System):
    one: int = 1
    two: int = 2

    def set_rand(self):
        self.one = random.random()
        self.two = random.random()
    
    def __str__(self):
        return f'{self.one}_{self.two}'

tst = Test()
tst.change_all_log_lvl(1)
print(tst)
with ProblemExec(tst) as pb1:
    tst.set_rand()
    print(tst)
    with ProblemExec(tst,level_name='2') as pb2:
        tst.set_rand()
        print(tst)        
        with ProblemExec(tst) as pb3:
            tst.set_rand()
            print(tst)
            pb3.exit_to_level('top',revert = True)
            print('why?',tst)
        print('E3',tst)
    print('E2',tst)
print('E1',tst)
"""