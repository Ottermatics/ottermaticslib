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

import ottermatics


import copy, traceback
import random, json
import time


log.info("testing with path: {}".format(sys.path))


class ImportTest(unittest.TestCase):
    """We test the compilation of all included modules"""

    def test_import_analysis(self):
        import ottermatics.analysis

    def test_import_common(self):
        import ottermatics.common

    def test_import_components(self):
        import ottermatics.components

    def test_import_configuration(self):
        import ottermatics.configuration

    def test_import_data(self):
        import ottermatics.datastores.data

    # def test_import_gdocs(self):
    #     import ottermatics.datastores.gdocs

    def test_import_locations(self):
        import ottermatics.locations

    def test_import_logging(self):
        import ottermatics.logging

    def test_import_plotting(self):
        import ottermatics.plotting

    # def test_import_process(self):
    #     import ottermatics.process

    def test_import_patterns(self):
        import ottermatics.patterns

    def test_import_solid_materials(self):
        import ottermatics.eng.solid_materials

    def test_import_tabulation(self):
        import ottermatics.tabulation

    def test_import_thermodynamics(self):
        import ottermatics.eng.thermodynamics


if __name__ == "__main__":
    print(("\n" * 10))
    unittest.main()
