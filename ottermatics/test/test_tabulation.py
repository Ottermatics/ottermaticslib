import unittest
import io
import tempfile

from ottermatics.configuration import otterize
from ottermatics.tabulation import system_property
from ottermatics.components import Component
import attr
import os
import numpy
import random

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

    @system_property(label="three", desc="some words")
    def test_three(self) -> int:
        return 2

    @system_property(label="four", desc="would make sense", stochastic=True)
    def test_four(self) -> float:
        return random.random()


class Test(unittest.TestCase):
    test_file_name = "test_dataframe_file"
    test_dir = "~/"

    def setUp(self):
        self.test_config = TestConfig()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        contents_local = os.listdir(self.test_dir)
        for fil in contents_local:
            if self.test_file_name in fil:
                rfil = os.path.join(self.test_dir, fil)
                # print('removing '+rfil)
                # os.remove(rfil)
        self.test_config.reset_table()

    def test_attrs_labels(self):
        # Default
        self.assertEqual(
            self.test_config.attr_labels, ["name", "attrs_prop", "attrs_str"]
        )

    def test_attrs_vals(self):
        self.assertEqual(self.test_config.attr_row, ["default", 1.0, "hey now"])

    def test_property_labels(self):
        ans = set(["four", "test_two", "test_one", "three"])
        self.assertEqual(set(self.test_config.system_properties_labels), ans)

    def test_property_types(self):
        ans = [int, float, int, float]
        self.assertEqual(self.test_config.system_properties_types, ans)

    def test_property_desc(self):
        ans = ["", "", "some words", "would make sense"]
        self.assertEqual(self.test_config.system_properties_description, ans)

    def test_table_vals(self):
        vals_ans = (1, 2)
        for val in vals_ans:
            self.assertTrue(
                any(
                    [
                        val == valcan if type(valcan) is int else False
                        for valcan in self.test_config.system_properties
                    ]
                )
            )

    def test_assemble_data_always_save(self):
        # print(f"testing data assembly {self.test_config.always_save_data}")
        # Run before test_table_to...
    
        self.assertFalse(self.test_config.TABLE)
        self.test_config.save_data()

        self.assertTrue(self.test_config.TABLE)
        self.assertTrue(1 in self.test_config.TABLE)

        # Stochastic Props so this will work without input change.
        self.test_config.save_data()
        self.assertFalse(2 in self.test_config.TABLE)

    def test_assemble_data_on_input(self):
        # change this to check behavior to adjust stochastic behavior
        # print(f"testing data assembly {self.test_config.always_save_data}")
        # Run before test_table_to...

        self.assertFalse(self.test_config.TABLE)
        self.test_config.save_data()

        self.assertTrue(self.test_config.TABLE)
        self.assertTrue(1 in self.test_config.TABLE)

        # No Update So This Wont Save
        self.test_config.save_data()
        self.assertFalse(2 in self.test_config.TABLE)

        # Change Something
        cur_val = self.test_config.attrs_prop
        new_val = 6 + cur_val
        self.test_config.info(f'setting attrs prop on in {cur_val } => {new_val}')
        self.test_config.attrs_prop = new_val
        self.test_config.save_data()
        self.assertTrue(2 in self.test_config.TABLE)

    def file_in_format(self, fileextension, path=True):
        fille = "{}.{}".format(self.test_file_name, fileextension)
        if path:
            path = os.path.join(self.test_dir, fille)
            return path
        else:
            return fille

    def test_dataframe(self, iter=5):
        attr_in = {}
        for i in range(iter):
            cur_val = self.test_config.attrs_prop
            attr_in[i] = val = cur_val + i**2.            
            self.test_config.info(f'setting attrs prop df {cur_val } => {val}')            
            self.test_config.attrs_prop = val
            self.test_config.save_data()

        df = self.test_config.dataframe
        self.assertEqual(len(df.index), iter)

        self.assertEqual(df["test_one"].max(), 1)
        self.assertEqual(df["test_one"].min(), 1)

        self.assertEqual(df["test_three"].max(), 2)
        self.assertEqual(df["test_three"].min(), 2)

        l1 = list(df["attrs_prop"])
        l2 = [attr_in[i] for i in range(iter)]
        self.assertListEqual(l1, l2)


# TODO: move these to reporting
#     def test_table_to_csv(self):
#         pass
#         print('saving '+self.file_in_format('xlsx'))
#         self.test_config.save_csv(self.file_in_format('csv'))
#         self.assertIn(self.file_in_format('csv',False), os.listdir(self.test_dir))

#
#     def test_table_to_excel(self):
#         pass
#         #print('saving '+self.file_in_format('xlsx'))
#         #self.test_config.save_excel(self.file_in_format('xlsx'))
#         #self.assertIn(self.file_in_format('xlsx',False),os.listdir(self.test_dir))
#
#     def test_table_to_gsheets(self):
#         pass
#
#     def test_table_to_db(self):
#         pass


@otterize
class Static(Component):
    attrs_prop: float = attr.ib(1.0)
    attrs_str: str = attr.ib("hey now")

    @system_property
    def test_one(self) -> int:
        return 1

    @system_property()
    def test_two(self) -> int:
        return 2


class TestStatic(unittest.TestCase):
    def setUp(self):
        self.test_config = Static()

    def test_static(self, num=10):
        for i in range(num):
            self.test_config.save_data()

        self.assertEqual(len(self.test_config.TABLE), 1)

    def test_input(self, num=10):
        for i in range(num):
            self.test_config.attrs_prop = i + self.test_config.attrs_prop
            self.test_config.save_data()

        self.assertEqual(len(self.test_config.TABLE), num)
