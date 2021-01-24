#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:28:07 2019

@author: kevinrussell
"""


from matplotlib.pylab import *

import os,shutil,copy,traceback,collections,logging

from ottermatics.configuration import *
from ottermatics.locations import *
from ottermatics.logging import *

log.info('Starting Ottermatics Enviornment')
load_from_env('./.creds/','env.sh')

#Constants
gravity = 9.81 #m/s


