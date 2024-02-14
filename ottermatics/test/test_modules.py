import pathlib

testpath = pathlib.Path(__file__).parent.absolute()

import logging

log = logging.getLogger("neptunya-test")

import sys

run_path = "{}/../analysis/".format(testpath)
log.info("adding neptunya path: {}".format(run_path))
sys.path.append(run_path)

from matplotlib import pylab
import numpy as n
import unittest

import engforge


import copy, traceback
import random, json
import time


log.info("testing with path: {}".format(sys.path))


class ImportTest(unittest.TestCase):
    """We test the compilation of all included modules"""

    def test_import_analysis(self):
        import engforge.analysis

    def test_import_common(self):
        import engforge.common

    def test_import_components(self):
        import engforge.components

    def test_import_configuration(self):
        import engforge.configuration

    def test_import_data(self):
        import engforge.datastores.data

    # def test_import_gdocs(self):
    #     import engforge.datastores.gdocs

    def test_import_locations(self):
        import engforge.locations

    def test_import_logging(self):
        import engforge.logging

    def test_import_plotting(self):
        import engforge.plotting

    # def test_import_process(self):
    #     import engforge.process

    def test_import_patterns(self):
        import engforge.patterns

    def test_import_solid_materials(self):
        import engforge.eng.solid_materials

    def test_import_tabulation(self):
        import engforge.tabulation

    def test_import_thermodynamics(self):
        import engforge.eng.thermodynamics


if __name__ == "__main__":
    print(("\n" * 10))
    unittest.main()
