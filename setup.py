#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:38:11 2019

@author: kevinrussell
"""

from setuptools import setup

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements('requirements.txt')

setup(name='Ottermatics Lib',
      version='0.1',
      description='The Ottermatic\'s Python Lib!',
      url='',
      author='Olly',
      author_email='olly@ottermatics.com',
      license='MIT',
      install_requires = install_reqs,
      zip_safe=False)