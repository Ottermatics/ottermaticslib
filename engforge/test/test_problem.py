import unittest
from engforge.problem_context import *
from engforge.system import System,forge
from engforge.test.solver_testing_components import *
from engforge.test.test_slider_crank import SliderCrank
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
    

chk = lambda d,k: set(d.get(k)) if k in d else set()

class TestSession(unittest.TestCase):
    """Test lifecycle of a problem and IO of the context"""

    def test_system_last_context(self):
        sm = SpringMass(Fa=0,u=5)
        sm.run(dxdt=0)
        ssid = sm.last_context._session_id
        sm.run(dxdt=0)
        trid = sm.last_context._session_id
        self.assertNotEqual(ssid,trid,'Session ID should change after a run')

    def test_system_change_context(self):
        sm = SpringMass(Fa=0,u=5)
        sm.run(dxdt=0)
        ssid = sm.last_context._session_id
        trsm,df = sm.simulate(dt=0.01, endtime=0.1,return_all=True)
        trid = trsm.last_context._session_id
        self.assertNotEqual(ssid,trid,'Session ID should change after a run')

    def test_slide_crank_empty(self):
        sm = SliderCrank(Tg=0)
        pbx = ProblemExec(sm,{'combos':'','slv_vars':'','dxdt':None})
        atx = pbx.ref_attrs
        self.assertEqual(chk(atx,'solver.var'),set())
        self.assertEqual(chk(atx,'solver.obj'),set())
        self.assertEqual(chk(atx,'solver.ineq'),set())
        self.assertEqual(chk(atx,'solver.eq'),set())
        self.assertEqual(chk(atx,'dynamics.output'),set())
        self.assertEqual(chk(atx,'dynamics.rate'),set())
        self.assertEqual(chk(atx,'dynamics.state'),set())
        self.assertEqual(chk(atx,'dynamics.input'),set())

        cons = pbx.constraints
        self.assertEqual(cons['constraints'],[])
        self.assertEqual(len(cons['bounds']),len(atx['solver.var']))      

    def test_slide_crank_dflt(self):
        sm = SliderCrank(Tg=0)
        pbx = ProblemExec(sm,{'dxdt':None})
        atx = pbx.ref_attrs
        self.assertEqual(chk(atx,'solver.var'),set())
        self.assertEqual(chk(atx,'solver.obj'),set())
        self.assertEqual(chk(atx,'solver.ineq'),set())
        self.assertEqual(chk(atx,'solver.eq'),set())
        self.assertEqual(chk(atx,'dynamics.output'),set())
        self.assertEqual(chk(atx,'dynamics.rate'),set())
        self.assertEqual(chk(atx,'dynamics.state'),set())
        self.assertEqual(chk(atx,'dynamics.input'),set())

        cons = pbx.constraints
        self.assertEqual(len(cons['constraints']),0)
        self.assertEqual(len(cons['bounds']),0)      

    def test_slide_crank_dxdt_true(self):
        sm = SliderCrank(Tg=0)
        pbx = ProblemExec(sm,{'dxdt':True})
        atx = pbx.ref_attrs
        self.assertEqual(chk(atx,'solver.var'),set())
        self.assertEqual(chk(atx,'solver.obj'),set())
        self.assertEqual(chk(atx,'solver.ineq'),set())
        self.assertEqual(chk(atx,'solver.eq'),set())
        self.assertEqual(chk(atx,'dynamics.output'),set())
        self.assertEqual(chk(atx,'dynamics.rate'),set(('omega','theta')))
        self.assertEqual(chk(atx,'dynamics.state'),set(('omega','theta')))
        self.assertEqual(chk(atx,'dynamics.input'),set(('Tg',)))

        cons = pbx.constraints
        self.assertEqual(cons['constraints'],[])
        self.assertEqual(len(cons['bounds']),len(atx['solver.var'])) 

    def test_slide_crank_dxdt_zero(self):
        sm = SliderCrank(Tg=0)
        pbx = ProblemExec(sm,{'dxdt':0})
        atx = pbx.ref_attrs
        self.assertEqual(chk(atx,'solver.var'),set())
        self.assertEqual(chk(atx,'solver.obj'),set())
        self.assertEqual(chk(atx,'solver.ineq'),set())
        self.assertEqual(chk(atx,'solver.eq'),set())
        self.assertEqual(chk(atx,'dynamics.output'),set())
        self.assertEqual(chk(atx,'dynamics.rate'),set(('omega','theta')))
        self.assertEqual(chk(atx,'dynamics.state'),set(('omega','theta')))
        self.assertEqual(chk(atx,'dynamics.input'),set(('Tg',)))

        cons = pbx.constraints
        self.assertEqual(cons['constraints'],[])
        self.assertEqual(len(cons['bounds']),2) 


    def test_slide_crank_design(self):
        sm = SliderCrank(Tg=0)
        pbx = ProblemExec(sm,{'combos':'design','dxdt':None})
        atx = pbx.ref_attrs
        self.assertEqual(chk(atx,'solver.var'),set(('Lo','Rc','Ro','x_offset','y_offset')))
        self.assertEqual(chk(atx,'solver.obj'),set(('cost_slv',)))
        self.assertEqual(chk(atx,'solver.ineq'),set(('crank_pos_slv','gear_pos_slv','motor_pos_slv')))
        self.assertEqual(chk(atx,'solver.eq'),set(('gear_speed_slv','range_slv')))
        self.assertEqual(chk(atx,'dynamics.output'),set())
        self.assertEqual(chk(atx,'dynamics.rate'),set())
        self.assertEqual(chk(atx,'dynamics.state'),set())
        self.assertEqual(chk(atx,'dynamics.input'),set())

        cons = pbx.constraints
        self.assertEqual(len(cons['constraints']),5)
        self.assertEqual(len(cons['bounds']),5)

    def test_slide_crank_design_slv(self):
        sm = SliderCrank(Tg=0)
        pbx = ProblemExec(sm,{'combos':'design','slv_vars':'*slv','dxdt':None})
        atx = pbx.ref_attrs
        self.assertEqual(chk(atx,'solver.var'),set())
        self.assertEqual(chk(atx,'solver.obj'),set(('cost_slv',)))
        self.assertEqual(chk(atx,'solver.ineq'),set(('crank_pos_slv','gear_pos_slv','motor_pos_slv')))
        self.assertEqual(chk(atx,'solver.eq'),set(('gear_speed_slv','range_slv')))
        self.assertEqual(chk(atx,'dynamics.output'),set())
        self.assertEqual(chk(atx,'dynamics.rate'),set())
        self.assertEqual(chk(atx,'dynamics.state'),set())
        self.assertEqual(chk(atx,'dynamics.input'),set())

        cons = pbx.constraints
        self.assertEqual(len(cons['constraints']),5)
        self.assertEqual(len(cons['bounds']),2)

    def test_slide_crank_design_slv(self):
        sm = SliderCrank(Tg=0)
        pbx = ProblemExec(sm,{'combos':'design', 'ign_combos':'max*' ,'slv_vars':'*slv','dxdt':None})
        atx = pbx.ref_attrs
        self.assertEqual(chk(atx,'solver.var'),set())
        self.assertEqual(chk(atx,'solver.obj'),set(('cost_slv',)))
        self.assertEqual(chk(atx,'solver.ineq'),set(('crank_pos_slv','gear_pos_slv','motor_pos_slv')))
        self.assertEqual(chk(atx,'solver.eq'),set(('gear_speed_slv','range_slv')))
        self.assertEqual(chk(atx,'dynamics.output'),set())
        self.assertEqual(chk(atx,'dynamics.rate'),set())
        self.assertEqual(chk(atx,'dynamics.state'),set())
        self.assertEqual(chk(atx,'dynamics.input'),set())

        cons = pbx.constraints
        self.assertEqual(len(cons['constraints']),5)
        self.assertEqual(len(cons['bounds']),len(atx['solver.var']))        

    def test_slide_crank_add_var(self):
        sc = SliderCrank()
        out = sc.run(combos='design',slv_vars='*',revert_last=False,add_vars='*gear*')
        self.assertEqual(out['gear_speed_slv'],0)    




    
class TestContextExits(unittest.TestCase):

    def test_context_singularity(self):
        tst = SimpleContext()

        with ProblemExec(tst) as pb1:
            self.assertEqual(pb1,tst.last_context)
            self.assertTrue(pb1.entered)
            self.assertFalse(pb1.exited)                
            with ProblemExec(tst,level_name='2') as pb2:
                self.assertIs(pb1,pb2)
                self.assertTrue(pb1.entered)
                self.assertFalse(pb1.exited)                
                with ProblemExec(tst) as pb3:
                    self.assertIs(pb1,pb3)
                    self.assertTrue(pb3.entered)
                    self.assertFalse(pb3.exited)
                    pb3.exit_to_level('top',revert = True)
                    raise Exception('Wrong Level')
                raise Exception('Wrong Level')
            raise Exception('Wrong Level')
        self.assertEqual(pb1,tst.last_context)
        self.assertTrue(pb1.entered)
        self.assertTrue(pb1.exited)      

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