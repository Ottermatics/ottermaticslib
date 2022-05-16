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

setup(name='ottermatics',
      version='0.6.1',
      description='The Ottermatic\'s Python Lib!',
      url='https://github.com/SoundsSerious/ottermaticslib',
      author='kevin russell',
      packages=["ottermatics"],
      author_email='kevin@ottermatics.com',
      license='MIT',
      entry_points={
            'console_scripts': [
                # command = package.module:function
                'condaenvset=ottermatics.common:main_cli',
                'ollymakes=ottermatics.locations:main_cli',
                'otterdrive=ottermatics.gdocs:main_cli'
            ]
      },
      install_requires = install_reqs,
      zip_safe=False)