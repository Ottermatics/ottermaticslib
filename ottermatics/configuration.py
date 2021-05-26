from contextlib import contextmanager
import attr

from ottermatics.logging import LoggingMixin, log
from ottermatics.locations import *

import numpy
import functools
import itertools
import datetime
import pandas
import os
import inspect
import pathlib

import matplotlib.pyplot as plt


#Not sure where to put this, this will vecorize a class fcuntion
class inst_vectorize(numpy.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]   

#Decorators
'''Ok get ready for some fancy bullshit, this represents alot of meta functionality to get nice
python syntax througout our application code. The concept of the 'otterize' decorator and the 
Configuration class at the moment is sacred, they are nessicary together because the otterize
handles runtime meta operatoins, while Configuration is more like a normal class. All you need to know
how to do is use the 3rd party attr's libray. Sorry if you dont like it!

This should allow alot more functionalty and standardization options than just pure subclassing as otterize can be significantly expanded
'''


#Class Definition Wrapper Methods
def property_changed(instance, variable, value):
    instance._anything_changed = True
    return value

#This one should wrap all configuraitons to track changes, and special methods
#TODO: Make this accept arguments in appication
def otterize(cls,*args,**kwargs):
    '''Wrap all Configurations with this decorator
    
    It might be nice to use all the attrs magic methods but hash requires frozen for automation, not clear what the best config is'''
    acls = attr.s(cls, on_setattr= property_changed, eq=False,repr=False,hash=False,frozen=False, *args,**kwargs)
    return acls                                        

    

def meta(title,desc=None,**kwargs):
    '''a convienience wrapper to add metadata to attr.ib
    :param title: a title that gets formatted for column headers
    :param desc: a description of the property'''
    out = {'label':title.replace('_',' ').replace('-',' ').title(),
            'desc':None,
            **kwargs}
    return out


@otterize
class Configuration(LoggingMixin):
    '''Configuration is a pattern for storing attributes that might change frequently, and proivdes the core functionality for a host of different applications.

    Configuration is able to go through itself and its objects and map all included Configurations, just to a specific level.

    Common functionality includes an __on_init__ wrapper for attrs post-init method
    '''
    _temp_vars = None

    name = attr.ib(default='default',validator= attr.validators.instance_of(str),kw_only=True)

    log_fmt = "[%(identity)-24s]%(message)s"
    log_silo = True

    _created_datetime = None

    #Our Special Init Methodology
    def __on_init__(self):
        '''Override this when creating your special init functionality, you must use attrs for input variables'''
        pass

    def __attrs_post_init__(self):
        '''This is called after __init__ by attr's functionality, we expose __oninit__ for you to use!'''
        #Store abs path On Creation, in case we change
        self._created_datetime = datetime.datetime.utcnow()
        self.__on_init__()
    
    #Identity & locatoin Methods
    @property
    def filename(self):
        '''A nice to have, good to override'''
        fil = self.identity.replace(' ','_').replace('-','_').replace(':','').replace('|','_').title()
        filename = "".join([c for c in fil if c.isalpha() or c.isdigit() or c=='_' or c=='-']).rstrip()
        return filename

    @property
    def displayname(self):
        dn = self.identity.replace('_',' ').replace('|',' ').replace('-',' ').replace('  ',' ').title()
        #dispname = "".join([c for c in dn if c.isalpha() or c.isdigit() or c=='_' or c=='-']).rstrip()
        return dn

    @property
    def identity(self):
        '''A customizeable property that will be in the log by default'''
        if not self.name:
            return self.classname.lower()
        return f'{self.classname}-{self.name}'.lower()

    @property
    def classname(self):
        '''Shorthand for the classname'''
        return str( type(self).__name__ )

    def add_fields(self, record):
        '''Overwrite this to modify logging fields, change log_fmt in your class to use the value set here.'''
        record.identity = self.identity      

    #Configuration Information
    @property
    def internal_configurations(self):
        '''go through all attributes determining which are configuration objects
        we skip any configuration that start with an underscore (private variable)'''
        return {k:v for k,v in self.store.items() \
                if isinstance(v,Configuration) and not k.startswith('_')}


    def go_through_configurations(self,level = 0,levels_to_descend = -1, parent_level=0):
        '''A generator that will go through all internal configurations up to a certain level
        if levels_to_descend is less than 0 ie(-1) it will go down, if it 0, None, or False it will
        only go through this configuration
        
        :return: level,config'''

        should_yield_level = lambda level: all([level>=parent_level, \
                                              any([levels_to_descend < 0, level <= levels_to_descend])])

        if should_yield_level(level):
            yield level,self

        level += 1
        for config in self.internal_configurations.values():
            for level,iconf in config.go_through_configurations(level,levels_to_descend,parent_level):
                yield level,iconf


    #May Depriciate - Dirty laundry below
    @contextmanager
    def difference(self,**kwargs):
        '''may want to consider using attr.evolve instead.... a context manager that will allow you 
        to dynamically change any information, then will change it back in a fail safe way. 
        
        with self.difference(name='new_name', value = new_value) as new_config:
            #do stuff with config, ok to fail

        you may not access any "private" variable that starts with an `_` as in _whatever
        
        difference is useful for saving slight differences in configuration in conjunction with solve
        you might create wrappers for evaluate, or implement a strategy pattern.

        only attributes may be changed.
        '''
        _temp_vars = {}
            
        _temp_vars.update({arg: getattr(self,arg) for arg in kwargs.keys() \
                                if hasattr(self,arg) if not arg.startswith('_')})

        bad_vars = set.difference(set(kwargs.keys()),set(_temp_vars.keys()))
        if bad_vars:
            self.warning('Could Not Change {}'.format( ','.join(list(bad_vars ) )))

        try: #Change Variables To Input
            for arg,var in kwargs.items():
                setattr(self,arg,var)
            yield self
        finally:
            for arg in kwargs.keys():
                var = _temp_vars[arg]
                setattr(self,arg,var)        

    #Ehhhh not a great look
    @property
    def store(self):
        '''lets pretend we're not playing with fire'''
        return self.__dict__




