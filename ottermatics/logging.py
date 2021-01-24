import logging


from twisted.python import log

import datetime
import logging

import traceback

import json
import sys,os
import uuid
import graypy

BASIC_LOG_FMT = "[%(name)-24s]%(message)s"

log = logging.getLogger('')
logging.basicConfig(format=BASIC_LOG_FMT,level=logging.INFO)

FILE = os.path.dirname(__file__)
CREDS = os.path.join(FILE, 'creds')
credfile = lambda fil: os.path.join(CREDS,fil)


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
    log.setLevel(level=logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    peerlog = logging.Formatter()
    sh.setFormatter(peerlog)
    log.addHandler( sh )    

class LoggingMixin(logging.Filter):
    '''Class to include easy formatting in subclasses'''
    
    log_level = 30
    _log = None

    log_on = True
    gelf = None

    log_fmt = "[%(namespace)-24s]%(message)s"
    log_silo = False

    @property
    def logger(self):
        if self._log is None:
            self._log = logging.getLogger(self.identity)
            #self._log.setLevel(level = logging.INFO)
            #Eliminate Outside Logging Interaction
            
            self._log.handlers = []

            #We have our own methods since it doesn't interact
            if self.log_silo:
                self._log.propagate = False
                self.installSTDLogger()
                #self.installGELFLogger()

            #Apply Filter Info
            self._log.addFilter(self)

        return self._log

    # def installGELFLogger(self):
    #     '''Installs GELF Logger'''
    #     gelf = graypy.GELFUDPHandler(host=GELF_HOST,port=12203, extra_fields=True)
    #     self._log.addHandler(gelf)

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
        record.namespace = self.identity.lower()
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
        msg = self.extract_message(args)
        self.logger.critical(msg)

    def extract_message(self,args):
        for arg in args:
            if type(arg) is str:
                return arg
        return ''

    @property
    def identity(self):
        return type(self).__name__
