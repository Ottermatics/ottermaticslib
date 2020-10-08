#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:28:07 2019

@author: kevinrussell
"""

print('Starting Ottermatics Enviornment')

from matplotlib.pylab import *

import os,shutil,copy,traceback,collections,logging

from .configuration import *
from .locations import *
import logging
LOG = logging.getLogger()
LOG.setLevel( logging.INFO)

#Constants
gravity = 9.81 #m/s
