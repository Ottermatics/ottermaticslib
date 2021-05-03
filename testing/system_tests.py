
import pathlib
testpath = pathlib.Path(__file__).parent.absolute()

import logging

log = logging.getLogger('neptunya-test')

import sys
run_path = '{}/../analysis/'.format(testpath)
log.info('adding neptunya path: {}'.format(run_path))
sys.path.append(run_path)

#from matplotlib.pylab import *
import numpy as n
import unittest


import copy,traceback
import random,json
import time


log.info('testing with path: {}'.format(sys.path))
class ImportTest( unittest.TestCase ):
    '''We test the compilation of all included modules'''

    def test_import_analysis(self):
        import analysis

    def test_import_common(self):
        import common

    def test_import_components(self):
        import components

    def test_import_configuration(self):
        import configuration

    def test_import_data(self):
        import data

    def test_import_gdocs(self):
        import gdocs

    def test_import_locations(self):
        import locations

    def test_import_logging(self):
        import logging

    def test_import_plotting(self):
        import plotting

    def test_import_process(self):
        import process

    def test_import_patterns(self):
        import patterns

    def test_import_solid_materials(self):
        import solid_materials

    def test_import_tabulation(self):
        import tabulation

    def test_import_thermodynamics(self):
        import thermodynamics


if __name__ == '__main__':
    print(('\n'*10))
    unittest.main()
