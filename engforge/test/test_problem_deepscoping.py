from engforge import *

import unittest
import attrs
import random

def test_obj(sys,prob):
    print(sys,prob)
    comp = sys
    assert comp is not prob.system, f'{comp} is problems system, in sub comp'
    return comp.val - comp.comp_val


def test_con(sys,prob):
    print(sys,prob)
    comp = sys
    assert comp is not prob.system, f'{comp} is problems system, in sub comp'
    return -1

def test_zero(sys,prob):
    print(sys,prob)
    comp = sys
    assert comp is not prob.system, f'{comp} is problems system, in sub comp'
    return 0


@forge
class TestComp(Component):
    val = attrs.field(factory=random.random,type=float)
    
    comp_val = attrs.field(default=0,type=float)
    slv_var = Solver.declare_var('comp_val')
    slv_var.add_var_constraint(0.0,kind='max')
    slv_var.add_var_constraint(test_con,kind='min')

    var_obj = Solver.objective(test_obj)
    var_ineq = Solver.con_ineq(test_zero)
    var_2ineq = Solver.con_ineq(test_zero,test_zero)  
    var_eq = Solver.con_eq(test_zero)
    var_2eq = Solver.con_eq(test_zero,test_zero)  

@forge
class DeepComp(TestComp):
    comp= Slot.define(TestComp,default_ok=False,none_ok=True)

@forge
class DeepSys(System):
    comp=Slot.define(DeepComp)


class TestDeep(unittest.TestCase):


    def test_one_deep(self):
        comp = DeepComp(comp=None)
        sys = DeepSys(comp=comp)

        sys.run(combos='default',slv_vars='*')