#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:28:07 2019

@author: kevinrussell
"""
global PORT, PASS, USER, HOST, DB_NAME, CLIENT_G_DRIVE,CLIENT_GDRIVE_SYNC,CLIENT_GMAIL,CLIENT_NAME,SLACK_WEBHOOK_NOTIFICATION

from matplotlib.pylab import *

import os,shutil,copy,traceback,collections,logging, subprocess

from typing import Callable, Iterator, Union, Optional, List
import sys

import logging
log = logging.getLogger('otterlib-init')
log.setLevel(logging.WARNING)
log.info('Starting Ottermatics Enviornment')

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def bool_from_env(bool_env_canidate):
    if bool_env_canidate.lower() in ('yes','true','y','1'):
        return True
    if bool_env_canidate.lower() in ('no','false','n','0'):
        return False
    return None

def load_from_env(creds_path='./.creds/',env_file='env.sh',set_env=True):
    '''extracts export statements from bash file and aplies them to the python env'''
    creds_path = os.path.join(creds_path,env_file)
    log.info("checking {} for creds".format(creds_path))
    if not 'OTTER_CREDS_SET' in os.environ or not bool_from_env(os.environ['OTTER_CREDS_SET']):        
        if os.path.exists(creds_path):
            log.info('creds found')
            with open(creds_path,'r') as fp:
                txt = fp.read()

            lines = txt.split('\n')
            for line in lines:
                if line.startswith('export'):
                    key,val = line.replace('export','').split('=')
                    log.info('setting {}'.format(key))
                    os.environ[key.strip()]=val
                    
        
        if set_env: os.environ['OTTER_CREDS_SET'] = 'yes'
    else:
        log.info('credientials already set') 


def set_conda_from_env(creds_path='./.creds/',env_file='env.sh',set_env=True):
    '''extracts export statements from bash file and aplies them to the python env'''
    creds_path = os.path.join(creds_path,env_file)
    log.info("checking {} for creds".format(creds_path))
    if os.path.exists(creds_path):
        log.info('creds found')
        with open(creds_path,'r') as fp:
            txt = fp.read()

        lines = txt.split('\n')
        for line in lines:
            if line.startswith('export'):
                key,val = line.replace('export','').split('=')
                log.info('setting {}'.format(key))
                subprocess.run(['conda','env','config','vars','set',f'{key.strip()}={val}'])

#load_from_env('./.creds/','env.sh',set_env=False)

CLIENT_G_DRIVE,CLIENT_GDRIVE_SYNC,CLIENT_GMAIL,CLIENT_NAME,SLACK_WEBHOOK_NOTIFICATION = None,None,None,None,None

#Backwards Compatability
PORT = 5432
HOST = 'localhost'
USER = 'postgres'
PASS = 'dumbpass'
DB_NAME = 'dumbdb'

DEFAULT_GDRIVE = "shared:OTTERBOX"

if 'CLIENT_GDRIVE_PATH' in os.environ:
    log.info('got CLIENT_GDRIVE_PATH')
    CLIENT_G_DRIVE = os.environ['CLIENT_GDRIVE_PATH']
else:
    CLIENT_G_DRIVE = DEFAULT_GDRIVE

if 'CLIENT_GDRIVE_SYNC' in os.environ:
    log.info('got CLIENT_GDRIVE_SYNC')
    CLIENT_GDRIVE_SYNC = bool_from_env(os.environ['CLIENT_GDRIVE_SYNC'])

if 'CLIENT_GMAIL' in os.environ:
    log.info('got CLIENT_GMAIL')
    CLIENT_GMAIL = os.environ['CLIENT_GMAIL']

if 'CLIENT_NAME' in os.environ:
    log.info('got CLIENT_NAME')
    CLIENT_NAME = os.environ['CLIENT_NAME'] 

if 'DB_NAME' in os.environ:
    DB_NAME = os.environ['DB_NAME']
    log.info("Getting ENV DB_NAME")
    
if 'DB_CONNECTION' in os.environ:
    HOST = os.environ['DB_CONNECTION']
    log.info("Getting ENV DB_CONNECTION")
    
if 'DB_USER' in os.environ:
    USER = os.environ['DB_USER']
    log.info("Getting ENV DB_USER")
    
if 'DB_PASS' in os.environ:
    PASS = os.environ['DB_PASS']
    log.info("Getting ENV DB_PASS")
    
if 'DB_PORT' in os.environ:
    PORT = os.environ['DB_PORT']
    log.info("Getting ENV DB_PORT")

if SLACK_WEBHOOK_NOTIFICATION is None and 'SLACK_WEBHOOK_NOTIFICATION' in os.environ:
    log.info('getting slack webhook')
    SLACK_WEBHOOK_NOTIFICATION = os.environ['SLACK_WEBHOOK_NOTIFICATION']    

#Constants
gravity = 9.81 #m/s
G_grav_constant = 6.67430E-11 #m3/kgs
speed_of_light = 299792458 #m/s
u_planck = 6.62607015E-34 #Js
R_univ_gas = 8.314462618#J/molkg
mass_electron = 9.1093837015E-31#kg
mass_proton = 1.67262192369E-27
mass_neutron = 1.67492749804E-27
r_electron = 2.8179403262E-15

#Common Modules
from ottermatics.configuration import *
from ottermatics.logging import *
from ottermatics.locations import *

def main_cli():
    import argparse

    parser = argparse.ArgumentParser('Set Conda Envs')
    parser.add_argument('-env',help='path to env.sh file')

    args = parser.parse_args()

    set_conda_from_env(os.path.dirname(args.env),os.path.basename(args.env))

if __name__ == '__main__':
    '''CLI To set Enviornmental Variables in Conda'''
    main_cli()
