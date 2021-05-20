import logging


import datetime
import logging

import traceback

import json
import sys,os
import uuid
#import graypy
import socket
from urllib.request import urlopen

import requests

BASIC_LOG_FMT = "[%(name)-24s]%(message)s"

LOG_LEVEL = logging.INFO

global SLACK_WEBHOOK_NOTIFICATION
SLACK_WEBHOOK_NOTIFICATION = None

try:
    #This should always work unless we don't have privideges (rare assumed)
    HOSTNAME = socket.gethostname().upper()
except:
    HOSTNAME = 'MASTER'

log = logging.getLogger('')
logging.basicConfig(format=BASIC_LOG_FMT,level=LOG_LEVEL)

FILE = os.path.dirname(__file__)
CREDS = os.path.join(FILE, 'creds')
credfile = lambda fil: os.path.join(CREDS,fil)

def is_ec2_instance():
    """Check if an instance is running on AWS."""
    result = False
    meta = 'http://169.254.169.254/latest/meta-data/public-ipv4'
    try:
        result = urlopen(meta,timeout=5.0).status == 200
        return True
    except:
        return False
    return False   


try:
    logging.getLogger('parso.cache').disabled=True
    logging.getLogger('parso.cache.pickle').disabled=True
    logging.getLogger('parso.python.diff').disabled=True

except Exception as e:
    log.warning(f'could not diable parso {e}')
# def installGELFLogger():
#     '''Installs GELF Logger'''
#     # self.gelf = graypy.GELFTLSHandler(GELF_HOST,GELF_PORT, validate=True,\
#     #                         ca_certs=credfile('graylog-clients-ca.crt'),\
#     #                         certfile = credfile('test-client.crt'),
#     #                         keyfile = credfile('test-client.key')
#     #                         )
#     log = logging.getLogger('')
#     gelf = graypy.GELFUDPHandler(host=GELF_HOST,port=12203, extra_fields=True)
#     log.addHandler(gelf)


def installSTDLogger(fmt = BASIC_LOG_FMT):
    '''We only want std logging to start'''
    log = logging.getLogger('')
    sh = logging.StreamHandler(sys.stdout)
    peerlog = logging.Formatter()
    sh.setFormatter(peerlog)
    log.addHandler( sh )    


def set_all_loggers_to(level,set_stdout=False,all_loggers=False):
    global LOG_LEVEL
    LOG_LEVEL = level
    
    if set_stdout: installSTDLogger()

    logging.basicConfig(level = LOG_LEVEL) #basic config
    #log = logging.getLogger()
    #log.setLevel(LOG_LEVEL)# Set Root Logger

    #log.setLevel(level) #root
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if logger.__class__.__name__.lower().startswith('otterlog'):
            logger.log(LOG_LEVEL,'setting log level: {}'.format(LOG_LEVEL))
            logger.setLevel(LOG_LEVEL)
        elif all_loggers:
            logger.log(LOG_LEVEL,'setting log level: {}'.format(LOG_LEVEL))
            logger.setLevel(LOG_LEVEL)



class LoggingMixin(logging.Filter):
    '''Class to include easy formatting in subclasses'''
    
    log_level = 30
    _log = None

    log_on = True
    gelf = None

    log_fmt = "[%(name)-24s]%(message)s"
    log_silo = False

    @property
    def logger(self):
        global LOG_LEVEL
        if self._log is None:
            self._log = logging.getLogger('otterlog_' +self.identity)
            self._log.setLevel(level = LOG_LEVEL)
            
            #Apply Filter Info
            self._log.addFilter(self)

            #Eliminate Outside Logging Interaction
            if self.log_silo:
                self._log.handlers = []
                self._log.propagate = False
                self.installSTDLogger()
                #self.installGELFLogger()

        return self._log

    # def installGELFLogger(self):
    #     '''Installs GELF Logger'''
    #     gelf = graypy.GELFUDPHandler(host=GELF_HOST,port=12203, extra_fields=True)
    #     self._log.addHandler(gelf)

    def resetLog(self):
        self._log = None

    def installSTDLogger(self):
        '''We only want std logging to start'''
        sh = logging.StreamHandler(sys.stdout)
        peerlog = logging.Formatter(self.log_fmt)
        sh.setFormatter(peerlog)
        self._log.addHandler( sh )            
      
       
    def add_fields(self, record):
        '''Overwrite this to modify logging fields'''
        pass

    def filter(self, record):
        '''This acts as the interface for `logging.Filter`
        Don't overwrite this, use `add_fields` instead.'''
        record.name = self.identity.lower()
        self.add_fields(record)
        return True

    def msg(self,*args):
        '''Writes to log... this should be for raw data or something... least priorty'''
        if self.log_on:
            self.logger.log(0,self.extract_message(args))

    def debug(self,*args):
        '''Writes at a low level to the log file... usually this should
        be detailed messages about what exactly is going on'''
        if self.log_on:
            self.logger.debug( self.extract_message(args))

    def info(self,*args):
        '''Writes to log but with info category, these are important typically
        and inform about progress of process in general'''
        if self.log_on:
            self.logger.info( self.extract_message(args))

    def warning(self,*args):
        '''Writes to log as a warning'''
        self.logger.warning(self.extract_message(args))

    def error(self,error,msg=''):
        '''Writes to log as a error'''
        fmt = '{msg!r}|{err!r}'
        tb = ''.join(traceback.format_exception(etype=type(error), value=error, tb=error.__traceback__))
        self.logger.exception( fmt.format(msg=msg,err=tb))

    def critical(self,*args):
        '''A routine to communicate to the root of the server network that there is an issue'''
        global SLACK_WEBHOOK_NOTIFICATION
        msg = self.extract_message(args)
        self.logger.critical(msg)

        self.slack_notification(self.identity.title(),msg)        

    def slack_notification(self, category, message, stage=HOSTNAME):
        global SLACK_WEBHOOK_NOTIFICATION
        if SLACK_WEBHOOK_NOTIFICATION is None and 'SLACK_WEBHOOK_NOTIFICATION' in os.environ:
            self.info('getting slack webhook')
            SLACK_WEBHOOK_NOTIFICATION = os.environ['SLACK_WEBHOOK_NOTIFICATION']
        headers = {'Content-type': 'application/json'}
        data = {'text':"{category} on {stage}:\n```{message}```".format(\
                            category=category.upper(),\
                            stage=stage,\
                            message=message)}
        self.info(f'Slack Notification : {SLACK_WEBHOOK_NOTIFICATION}:{category},{message}')
        slack_note = requests.post(SLACK_WEBHOOK_NOTIFICATION,data= json.dumps(data).encode('ascii'),headers=headers) 

    def extract_message(self,args):
        for arg in args:
            if type(arg) is str:
                return arg
        return ''

    @property
    def identity(self):
        return type(self).__name__
