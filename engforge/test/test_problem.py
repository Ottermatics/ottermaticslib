import unittest
from engforge.problem_context import *
from engforge.system import System,forge
import random


@forge(auto_attribs=True)
class SimpleContext(System):
    one: int = 1
    two: int = 2

    def set_rand(self):
        self.one = random.random()
        self.two = random.random()
    
    def __str__(self):
        return f'{self.one}_{self.two}'
    


    
class TestContextExits(unittest.TestCase):

    def test_exit_top(self):
        tst = SimpleContext()

        with ProblemExec(tst) as pb1:
            tst.set_rand()
            with ProblemExec(tst,level_name='2') as pb2:
                tst.set_rand() 
                with ProblemExec(tst) as pb3:
                    tst.set_rand()
                    pb3.exit_to_level('top',revert = True)
                    raise Exception('Wrong Level')
                raise Exception('Wrong Level')
            raise Exception('Wrong Level')
        self.assertEqual(tst.one,1)
        self.assertEqual(tst.two,2)

    def test_exit_top_with_state(self):
        tst = SimpleContext()

        with ProblemExec(tst,level_name='super') as pb1:
            tst.set_rand()
            with ProblemExec(tst,level_name='2') as pb2:
                tst.set_rand() 
                with ProblemExec(tst) as pb3:
                    tst.set_rand()
                    final_one = tst.one
                    final_two = tst.two
                    pb3.exit_to_level('top',revert = False)
                    raise Exception('Wrong Level')
                raise Exception('Wrong Level')
            raise Exception('Wrong Level')
        self.assertEqual(tst.one,final_one)
        self.assertEqual(tst.two,final_two)

    def test_exit_2_with_state(self):
        tst = SimpleContext()

        with ProblemExec(tst) as pb1:
            tst.set_rand()
            with ProblemExec(tst,level_name='2') as pb2:
                tst.set_rand() 
                with ProblemExec(tst) as pb3:
                    tst.set_rand()
                    final_one = tst.one
                    final_two = tst.two
                    pb3.exit_to_level('2',revert = False)
                    raise Exception('Wrong Level')
            self.assertEqual(tst.one,final_one)
            self.assertEqual(tst.two,final_two)
        self.assertEqual(tst.one,1)
        self.assertEqual(tst.two,2)

    def test_exit_2_wo_state(self):
        tst = SimpleContext()
        with ProblemExec(tst) as pb1:
            tst.set_rand()
            final_one = tst.one
            final_two = tst.two     
            with ProblemExec(tst,level_name='2') as pb2:
                tst.set_rand()            
                with ProblemExec(tst) as pb3:
                    tst.set_rand()
                    pb3.exit_to_level('2',revert = True)
                    raise Exception('Wrong Level')
            self.assertEqual(tst.one,final_one)
            self.assertEqual(tst.two,final_two)
        self.assertEqual(tst.one,1)
        self.assertEqual(tst.two,2)             


    def test_exit_with_state(self):
        tst = SimpleContext()

        with ProblemExec(tst) as pb1:
            tst.set_rand()
            mid_one = tst.one
            mid_two = tst.two            
            with ProblemExec(tst,level_name='2') as pb2:
                tst.set_rand() 
                final_one = tst.one
                final_two = tst.two                      
                with ProblemExec(tst) as pb3:
                    tst.set_rand()
                    final_one = tst.one
                    final_two = tst.two
                    pb3.exit_with_state()
                    raise Exception('Wrong Level')
                self.assertEqual(tst.one,final_one)
                self.assertEqual(tst.two,final_two)
            self.assertEqual(tst.one,mid_one)
            self.assertEqual(tst.two,mid_two)                
        self.assertEqual(tst.one,1)
        self.assertEqual(tst.two,2)                     



    def test_exit_and_revert(self):
        tst = SimpleContext()

        with ProblemExec(tst) as pb1:
            tst.set_rand()
            mid_one = tst.one
            mid_two = tst.two
            with ProblemExec(tst,level_name='2') as pb2:
                tst.set_rand()
                final_one = tst.one
                final_two = tst.two 
                with ProblemExec(tst) as pb3:
                    tst.set_rand()
                   
                    pb3.exit_and_revert()
                    raise Exception('Wrong Level')
                self.assertEqual(tst.one,final_one)
                self.assertEqual(tst.two,final_two)
            self.assertEqual(tst.one,mid_one)
            self.assertEqual(tst.two,mid_two)                
        self.assertEqual(tst.one,1)
        self.assertEqual(tst.two,2)