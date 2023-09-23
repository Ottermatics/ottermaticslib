from ottermatics.components import Component
from ottermatics.component_collections import ComponentDict,ComponentIter
from ottermatics.system import System
from ottermatics.slots import SLOT
from ottermatics.configuration import otterize
from ottermatics.tabulation import system_property

import attrs,attr

import random
import unittest
import itertools
# from ottermatics.logging import LoggingMixin, change_all_log_levels
# change_all_log_levels(10)


@otterize
class TestConfig(Component):
    attrs_prop: float = attr.ib(1.0)
    attrs_str: str = attr.ib("hey now")

    @system_property
    def test_one(self) -> int:
        return 1

    @system_property(stochastic=True)
    def test_two(self) -> float:
        return random.random()

    @system_property
    def test_three(self) -> int:
        return 2

    @system_property(stochastic=True)
    def test_four(self) -> float:
        return random.random()

@otterize
class DictComp(ComponentDict):
    component_type = TestConfig

@otterize
class ListComp(ComponentDict):
    component_type = TestConfig

@otterize
class WideSystem(System):

    cdict = SLOT.define_iterator(DictComp)
    citer = SLOT.define_iterator(ComponentIter)

@otterize
class NarrowSystem(System):

    cdict = SLOT.define_iterator(DictComp,wide=False)
    citer = SLOT.define_iterator(ComponentIter,wide=False)



class TestWide(unittest.TestCase):

    def setUp(self):

        self.item_in = {i:'hey'*(i+1) for i in range(5)}
        
        lc = ListComp(component_type=TestConfig)
        for i,v in self.item_in.items():
            lc[i] = TestConfig(name=f'citer_{i}',attrs_prop=i,attrs_str=v)

        dc = DictComp(component_type=TestConfig)
        for i,v in self.item_in.items():
            dc[v] = TestConfig(name=f'cdict_{i}',attrs_prop=i,attrs_str=v)

        self.system = WideSystem(cdict=dc,citer=lc)

    def test_keys(self):
        self.assertFalse(bool(self.system.table))

        dat = self.system.data_dict

        all_keys = set(dat.keys())

        comps = {'cdict':['hey'*(i+1) for i in range(5)],'citer':list(range(5))}

        props = ['test_one','test_two','test_three','test_four','name','attrs_prop','attrs_str']

        should_keys = set()
        for ck,vlist in comps.items():
            for v in vlist:
                for p in props:
                    tkn = f'{ck}.{v}.{p}'
                    should_keys.add(tkn)
        
        self.assertTrue(should_keys.issubset(set(self.system.data_dict.keys())))

        #save the data to table
        self.system.run()

        self.assertTrue(len(self.system.table)==1)
        self.assertTrue(should_keys.issubset(set(self.system.table[1].keys())))
        self.assertTrue(should_keys.issubset(set(self.system.dataframe.keys())))

        

class TestNarrow(unittest.TestCase):

    def setUp(self):

        self.item_in = {i:'hey'*(i+1) for i in range(5)}
        
        lc = ListComp(component_type=TestConfig)
        for i,v in self.item_in.items():
            lc[i] = TestConfig(name=f'citer_{i}',attrs_prop=i,attrs_str=v)

        dc = DictComp(component_type=TestConfig)
        for i,v in self.item_in.items():
            dc[v] = TestConfig(name=f'cdict_{i}',attrs_prop=i,attrs_str=v)

        self.system = NarrowSystem(cdict=dc,citer=lc)

    def test_keys(self):
        self.assertFalse(bool(self.system.table))

        dat = self.system.data_dict

        all_keys = set(dat.keys())

        comps = {'cdict':['hey'*(i+1) for i in range(5)],'citer':list(range(5))}

        props = ['test_one','test_two','test_three','test_four','name','attrs_prop','attrs_str']

        should_keys = set()
        for ck,vlist in comps.items():
            #for v in vlist:
            for p in props:
                tkn = f'{ck}.{p}'
                should_keys.add(tkn)
        
        self.assertTrue(should_keys.issubset(set(self.system.data_dict.keys())))

        #save the data to table
        self.system.run()

        self.assertTrue(len(self.system.table)==5**len(comps))
        self.assertTrue(should_keys.issubset(set(self.system.table[1].keys())))
        self.assertTrue(should_keys.issubset(set(self.system.dataframe.keys())))

        #test item existence
        v1 = set(self.system.dataframe['cdict.current_item'])
        v2 = set(comps['cdict'])

        self.assertEqual(v1,v2)

        d1 = set(self.system.dataframe['citer.current_item'])
        d2 = set(comps['citer'])

        self.assertEqual(d1,d2)

        dvs = self.system.dataframe['cdict.current_item']
        cvs = self.system.dataframe['citer.current_item']
        
        al = set(zip(dvs,cvs))
        sh = set(itertools.product(v2,d2))

        self.assertEqual(al,sh)