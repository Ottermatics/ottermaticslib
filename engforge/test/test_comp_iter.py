from engforge.components import Component
from engforge.component_collections import ComponentDict, ComponentIter
from engforge.system import System
from engforge.attr_slots import Slot
from engforge.configuration import forge
from engforge.tabulation import system_property

import attrs, attr

import random
import unittest
import itertools

# from engforge.logging import LoggingMixin, change_all_log_levels
# change_all_log_levels(10)


@forge
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


@forge
class DictComp(ComponentDict):
    component_type = TestConfig


@forge
class ListComp(ComponentDict):
    component_type = TestConfig


@forge
class WideSystem(System):
    cdict = Slot.define_iterator(DictComp)
    citer = Slot.define_iterator(ComponentIter)


@forge
class NarrowSystem(System):
    cdict = Slot.define_iterator(DictComp, wide=False)
    citer = Slot.define_iterator(ComponentIter, wide=False)


class TestWide(unittest.TestCase):
    def setUp(self):
        self.item_in = {i: "hey" * (i + 1) for i in range(5)}

        lc = ListComp(component_type=TestConfig)
        for i, v in self.item_in.items():
            lc[i] = TestConfig(name=f"citer_{i}", attrs_prop=i, attrs_str=v)

        dc = DictComp(component_type=TestConfig)
        for i, v in self.item_in.items():
            dc[v] = TestConfig(name=f"cdict_{i}", attrs_prop=i, attrs_str=v)

        self.system = WideSystem(cdict=dc, citer=lc)

    def test_keys(self):
        self.assertFalse(len(self.system.dataframe))

        dat = self.system.data_dict

        all_keys = set(dat.keys())

        comps = {
            "cdict": ["hey" * (i + 1) for i in range(5)],
            "citer": list(range(5)),
        }

        props = [
            "test_one",
            "test_two",
            "test_three",
            "test_four",
            "name",
            "attrs_prop",
            "attrs_str",
        ]

        should_keys = set()
        dataframe_keys = set()
        for ck, vlist in comps.items():
            for v in vlist:
                for p in props:
                    tkn = f"{ck}.{v}.{p}"
                    should_keys.add(tkn)
                    dataframe_keys.add(tkn.replace('.','_'))

        sys_key = set(self.system.data_dict.keys())
        mtch =should_keys.issubset(sys_key)
        self.system.debug(f'keys: {should_keys} vs {sys_key}')
        self.assertTrue(mtch,msg=f"missing keys: {should_keys - sys_key}")

        # save the data to table
        self.system.run(revert_last=False,revert_every=False,save_on_exit=True)

        df = self.system.last_context.dataframe
        self.assertTrue(len(df) == 1,msg=f"len: {len(df)}|\n{str(df)}")
        self.assertTrue(dataframe_keys.issubset(set(df.iloc[0].keys())))
        self.assertTrue(dataframe_keys.issubset(set(df.keys())))


class TestNarrow(unittest.TestCase):
    def setUp(self):
        self.item_in = {i: "hey" * (i + 1) for i in range(5)}

        lc = ListComp(component_type=TestConfig)
        for i, v in self.item_in.items():
            lc[i] = TestConfig(name=f"citer_{i}", attrs_prop=i, attrs_str=v)

        dc = DictComp(component_type=TestConfig)
        for i, v in self.item_in.items():
            dc[v] = TestConfig(name=f"cdict_{i}", attrs_prop=i, attrs_str=v)

        self.system = NarrowSystem(cdict=dc, citer=lc)

    def test_keys(self):
        self.assertFalse(len(self.system.dataframe)>0)

        dat = self.system.data_dict

        all_keys = set(dat.keys())

        comps = {
            "cdict": ["hey" * (i + 1) for i in range(5)],
            "citer": list(range(5)),
        }

        props = [
            "test_one",
            "test_two",
            "test_three",
            "test_four",
            "name",
            "attrs_prop",
            "attrs_str",
        ]

        should_keys = set()
        dataframe_keys = set()
        for ck, vlist in comps.items():
            # for v in vlist:
            for p in props:
                tkn = f"{ck}.{p}"
                should_keys.add(tkn)
                dataframe_keys.add(tkn.replace('.', '_'))

        sys_key = set(self.system.data_dict.keys())
        mtch =should_keys.issubset(sys_key)
        self.system.info(f'keys: {should_keys} vs {sys_key}')
        self.assertTrue(mtch,msg=f"missing keys: {should_keys - sys_key}")

        # save the data to table
        self.system.run()

        df = self.system.last_context.dataframe
        self.assertTrue(len(df) == 5 ** len(comps))
        self.assertTrue(dataframe_keys.issubset(set(df.iloc[0].keys())))
        self.assertTrue(dataframe_keys.issubset(set(df.keys())))

        # test item existence
        v1 = set(self.system.dataframe["cdict_current_item"])
        v2 = set(comps["cdict"])

        self.assertEqual(v1, v2)

        d1 = set(self.system.dataframe["citer_current_item"])
        d2 = set(comps["citer"])

        self.assertEqual(d1, d2)

        dvs = self.system.dataframe["cdict_current_item"]
        cvs = self.system.dataframe["citer_current_item"]

        al = set(zip(dvs, cvs))
        sh = set(itertools.product(v2, d2))

        self.assertEqual(al, sh)
