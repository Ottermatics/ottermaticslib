#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:38:11 2019

@author: kevinrussell
"""
from pip.req import parse_requirements
from setuptools import setup

install_reqs = parse_requirements(<requirements_path>)

setup(name='Ottermatics Lib',
      version='0.1',
      description='The Ottermatic\'s Python Lib!',
      url='',
      author='Olly',
      author_email='olly@ottermatics.com',
      license='MIT',
      install_requires = install_reqs,
      zip_safe=False)