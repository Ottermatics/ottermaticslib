import unittest
from engforge.test.solver_testing_components import *
from engforge.problem_context import ProblemExec
import itertools


class SolverRefSelection(unittest.TestCase):

    def setUp(self) -> None:
        self.sc = CubeSystem()

    def test_comps_solver_act(self):
        inv = 'ineq_cost,ineq_length,min_len'
        acts = ['*costA','*lenA']
        for act in [True,False]:
            for cmb in [inv,inv.split(',')]:
                extra = dict(combos=cmb,slv_vars='*x,*y,*z',activate=acts if act else None,only_active=act)
                info = self.sc.solver_vars(**extra)['attrs']
                ans = {'x','y','z','comp.x','comp.y','comp.z'}
                self.assertEqual(set(info['solver.var']),ans)
                self.assertEqual(set(info['dynamics.state']),ans)
                self.assertEqual(set(info['dynamics.rate']),ans) 

                empt = ['solver.eq','solver.obj','time','slot','signal']
                for emp in empt:
                    self.assertEqual(info[emp],{})
                
                if act:
                    ineq = {'comp.lenA','comp.costA','costA','lenA'}
                    self.assertEqual(set(info['solver.ineq']),ineq)

    def test_comps_solver_obj(self):
        inv = 'obj_size,ineq_cost,ineq_length,min_len'
        acts = ['*costA','*lenA']
        aacts = ['*costA','*lenA','*size']
        for act in [True,False]:
            for cmb in [inv,inv.split(',')]:
                actv = acts if act else aacts
                extra = dict(combos=cmb,slv_vars='*x,*y,*z',activate=actv,only_active=True)
                info = self.sc.solver_vars(**extra)['attrs']
                ans = {'x','y','z','comp.x','comp.y','comp.z'}
                self.assertEqual(set(info['solver.var']),ans)
                self.assertEqual(set(info['dynamics.state']),ans)
                self.assertEqual(set(info['dynamics.rate']),ans) 

                ineq = {'comp.lenA','comp.costA','costA','lenA'}
                self.assertEqual(set(info['solver.ineq']),ineq)    

                empt = ['solver.eq','time','slot','signal']
                for emp in empt:
                    self.assertEqual(info[emp],{})
                
                if not act:
                    obj = {'size','comp.size'}
                    self.assertEqual(set(info['solver.obj']),obj)
                else:
                    self.assertEqual(set(info['solver.obj']),set())

    def test_sys_solver(self):
        #test combos
        inv = 'volume,height,min_len,total_*'
        for act in [True,False]:
            for cmb in [inv,inv.split(',')]:
                extra = dict(combos=cmb,slv_vars='*x,*y,*z',activate=[],only_active=act)
                info = self.sc.solver_vars(**extra)['attrs']
                ans = {'x','y','z','comp.x','comp.y','comp.z'}
                self.assertEqual(set(info['solver.var']),ans)
                self.assertEqual(set(info['dynamics.state']),ans)
                self.assertEqual(set(info['dynamics.rate']),ans)

                obj = {'hght','obj'}
                self.assertEqual(set(info['solver.obj']),obj)
                ineq = {'sys_budget','sys_length'}
                self.assertEqual(set(info['solver.ineq']),ineq)

                empt = ['solver.eq','time','slot','signal']
                for emp in empt:
                    self.assertEqual(info[emp],{})            


    def test_xyz_var_only(self):
        '''only xyz via vars'''
        inv = '*x,*y,*z'
        for act in [True,False]:
            for cmb in [inv,inv.split(',')]:        
                extra = dict(combos=[],slv_vars=cmb,only_active=act)
                info = self.sc.solver_vars(**extra)['attrs']
                ans = {'x','y','z','comp.x','comp.y','comp.z'}
                self.assertEqual(set(info['solver.var']),ans)
                self.assertEqual(set(info['dynamics.state']),ans)
                self.assertEqual(set(info['dynamics.rate']),ans)

                empt = ['solver.eq','solver.ineq','solver.obj','time','slot','signal']
                for emp in empt:
                    self.assertEqual(info[emp],{})

    def test_xyz_combo_only(self):
        '''only xyz via combos'''
        inv = 'x,y,z'
        for act in [True,False]:
            for cmb in [inv,inv.split(',')]:        
                extra = dict(combos=cmb,slv_vars=[],only_active=act)
                info = self.sc.solver_vars(**extra)['attrs']
                ans = {'x','y','z','comp.x','comp.y','comp.z'}
                self.assertEqual(set(info['solver.var']),ans)
                self.assertEqual(set(info['dynamics.state']),ans)
                self.assertEqual(set(info['dynamics.rate']),ans)

                empt = ['solver.eq','solver.ineq','solver.obj','time','slot','signal']
                for emp in empt:
                    self.assertEqual(info[emp],{})

    def test_xyz_act_nocombo(self):
        '''activated aren't included because no combos are given'''
        extra = dict(combos=[],slv_vars='*x,*y,*z',activate=['*costA','*lenA','*size'],only_active=True)

        info = self.sc.solver_vars(**extra)['attrs']
        ans = {'x','y','z','comp.x','comp.y','comp.z'}
        self.assertEqual(set(info['solver.var']),ans)
        self.assertEqual(set(info['dynamics.state']),ans)
        self.assertEqual(set(info['dynamics.rate']),ans)

        empt = ['solver.eq','solver.ineq','solver.obj','time','slot','signal']
        for emp in empt:
            self.assertEqual(info[emp],{})



import pprint

class SingleCompSolverTest(unittest.TestCase):
    inequality_min = -1E-6
    def setUp(self) -> None:
        self.sc = CubeSystem()

    def test_exec_results(self):
        extra = dict(combos=[],slv_vars='*x,*y,*z',activate=[],only_active=True)
        with ProblemExec(self.sc,extra) as pb:
            o = self.sc.execute(**extra)
            self.assertDictEqual(o['Xstart'],o['Xans'])

    def test_run_results(self):
        """test that inputs stay the same when no objecives present"""
        extra = dict(combos=[],slv_vars='*x,*y,*z',activate=[],only_active=True)

        scx,scy,scz = self.sc.x,self.sc.y,self.sc.z
        sccx,sccy,sccz = self.sc.comp.x,self.sc.comp.y,self.sc.comp.z
        vray = [scx,scy,scz, sccx,sccy,sccz]

        o = self.sc.run(**extra)

        #dfray = self.sc.dataframe[['x','y','z','comp_x','comp_y','comp_z']].iloc[0].values
        #self.assertFalse(np.all(dfray==vray),msg=f'{dfray}\n{vray}')

        scx,scy,scz = self.sc.x,self.sc.y,self.sc.z
        sccx,sccy,sccz = self.sc.comp.x,self.sc.comp.y,self.sc.comp.z
        v2ray = [scx,scy,scz, sccx,sccy,sccz]

        self.assertTrue(np.all(vray==v2ray),msg='revert values changed!')
        
    def test_comp_method_equivalence(self):
        objs = {'obj_eff':['eff','effF'],'obj_size':['size','sizeF']}
        cons = {'ineq_cost':['costA','costF','costP'],'ineq_length':['lenA','lenF','lenP']}

        for ok,ck in itertools.product(objs,cons):
            attempts = []
            for var in objs[ok]:
                for con in cons[ck]:
                    extra = dict(combos=[ok,ck,'min_len'],slv_vars='*x,*y,*z',activate=[var,con],only_active=True)
                    o = self.sc.execute(**extra)
                    attempts.append(o)
            
            #check for equivalence
            a0 = attempts[0]
            for atmpt in attempts[1:]:
                ak1 = a0['Xans']
                ak2 = atmpt['Xans']
                self.assertEqual(set(ak1),set(ak2))
                for k in ak1:
                    self.assertAlmostEqual(ak1[k],ak2[k],places=4)
            
    def test_system_execute(self):
        """check the system solver methods work as expected (objectives are optimized)"""
        extra = dict(combos='volume,height,min_len,total_*',slv_vars='*x,*y,*z',activate=[],only_active=True)

        H = self.sc.execute(**extra,weights={'hght':1,'obj':0})
        V = self.sc.execute(**extra,weights={'hght':0,'obj':1})

        self.assertGreater(H['Yobj']['hght'],V['Yobj']['hght'])
        self.assertGreater(V['Yobj']['obj'],H['Yobj']['obj'])


    def test_system_run(self):
        """check the system solver methods work as expected (objectives are optimized)"""
        extra = dict(combos='volume,height,min_len,total_*',slv_vars='*x,*y,*z',activate=[],only_active=True)

        H = self.sc.run(**extra,weights={'hght':1,'obj':0})
        V = self.sc.run(**extra,weights={'hght':0,'obj':1})

        self.assertGreater(H['Yobj']['hght'],V['Yobj']['hght'])
        self.assertGreater(V['Yobj']['obj'],H['Yobj']['obj'])

    def test_system_solver(self):
        """check the system solver methods work as expected (objectives are optimized)"""
        extra = dict(combos='volume,height,min_len,total_*',slv_vars='*x,*y,*z',activate=[],only_active=True)

        H = self.sc.solver(**extra,weights={'hght':1,'obj':0})
        V = self.sc.solver(**extra,weights={'hght':0,'obj':1})

        self.assertGreater(H['Yobj']['hght'],V['Yobj']['hght'])
        self.assertGreater(V['Yobj']['obj'],H['Yobj']['obj'])

    def test_system_eval(self):
        """check the system solver methods work as expected (objectives are optimized)"""
        extra = dict(combos='volume,height,min_len,total_*',slv_vars='*x,*y,*z',activate=[],only_active=True)

        H = self.sc.eval(**extra,weights={'hght':1,'obj':0})
        V = self.sc.eval(**extra,weights={'hght':0,'obj':1})

        self.assertGreater(H['Yobj']['hght'],V['Yobj']['hght'])
        self.assertGreater(V['Yobj']['obj'],H['Yobj']['obj'])                     

#TODO: rewrite these tests
#     def test_selective_exec(self):
#         extra = dict(combos='*fun*,goal',ign_combos=['z_sym_eq','*size*','*eff*'])
#         o = self.sc.execute(**extra)
#         res = self.test_results(o)
#         self.test_selection(o,res,extra)
# 
#     def test_runmtx(self):
#         extra = dict(budget=[100,500,1000],max_length=[50,100,200])
#         o = self.sc.run(**extra)
#         res = self.test_results(o)
#         df = self.test_dataframe(o)
#         self.test_selection(o,res,extra)  
# 
#     def test_selective_runmtx(self):
#         extra = dict(budget=[100,500,1000],max_length=[50,100,200],combos='*fun*,goal',ign_combos=['z_sym_eq','*size*','*eff*'])        
#         o = self.sc.run(**extra)
#         res = self.test_results(o)
#         df = self.test_dataframe(o)
#         self.test_selection(o,res,extra)
# 
#     def test_run_wildcard(self):
#         extra = dict(combos='*fun*,goal',ign_combos='*',budget=[100,500,1000],max_length=[50,100,200])
#         o = self.sc.run(**extra)
#         res = self.test_results(o)
#         df = self.test_dataframe(o)
#         self.test_selection(o,res,extra)               
# 
#     #Checks
#     def test_dataframe(self,results)->dict:
#         #print(results)
#         print(NotImplementedError("#FIXME!"))
# 
#     def test_selection(self,results,df_res,extra):
#         #print(results,df_res,extra)
#         print(NotImplementedError("#FIXME!"))
# 
#     def test_results(self,results):
#         Ycon = results['Ycon']
#         costmg = {k:v for k,v in Ycon.items() if 'cost' in k}
#         lenmg = {k:v for k,v in Ycon.items() if 'len' in k}
# 
#         #Test input equivalence of methods
#         self.assertEqual(len(set(costmg.values())),1)
#         self.assertEqual(len(set(lenmg.values())),1)
# 
#         #Test inequality constraints
#         yvals = (v >= self.inequality_min for v in Ycon.values())
#         self.assertTrue(all(yvals))
