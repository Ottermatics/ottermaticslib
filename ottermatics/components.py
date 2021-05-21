from contextlib import contextmanager
import attr

from ottermatics.logging import LoggingMixin, log
from ottermatics.tabulation import TabulationMixin
from ottermatics.configuration import otterize, Configuration
from ottermatics.patterns import SingletonMeta, flatten


import numpy
import functools
import itertools
import datetime
import pandas
import os,sys
import inspect
import pathlib
import random
import matplotlib.pyplot as plt



@otterize
class Component(TabulationMixin):
    '''Component is an Evaluatable configuration with tabulation and reporting functionality'''
    
    #A solver function that will be called on every configuration
    def evaluate(self,*args,**kwargs):
        '''evaluate is a fleixble method to be overriden. Oftentimes it might not be used as 
        configurations are useful stores. These must be called from the top level, or solve will call 
        
        :return: not stored in table, but is returned through solve making it possibly useful 
        to track error, or other feedback for control'''
        return None

    def reset_data(self):
        self.reset_table()
        self._stored_plots = []
        self.index = 0
        for config in self.internal_configurations.values():
            config.reset_data()                   

    #Plotting & Report Methods:
    @property
    def saved_plots(self):
        return self._stored_plots

    # BUG: This breaks cloudpickle due to inspect+property recursion, lets find a different answer
    # @property
    # def plotting_methods(self):
    #     return {fname:func for fname,func in inspect.getmembers(self, predicate=inspect.ismethod) \
    #                                       if fname.startswith('plot_')}

    @contextmanager
    def subplots(self,plot_tile,save=True,*args,**kwargs):
        '''context manager for matplotlib subplots, which will save the plot if no failures occured
        using a context manager makes sense so we can record the plots made, and then upload them in 
        the post processing steps.
        
        it makes sense to always save images, but to override them.
        plot id should be identity+plot_title and these should be stored by date in the report path'''
        fig,maxes = plt.subplots(*args,**kwargs)

        try:
            yield fig,maxes
            
            if save:
                #determine file name
                filename = '{}_{}'.format(self.filename,plot_tile)                
                supported_filetypes = plt.gcf().canvas.get_supported_filetypes()
                if not any([filename.endswith(ext) for ext in supported_filetypes.keys()]):
                    if '.' in filename:
                        filename = '.'.join(filename.split('.')[:-1])+self._default_image_format
                    else:
                        filename += self._default_image_format

                filepath = os.path.join(self.config_path_daily,filename)

                self.info('saving plot {}'.format(filename))
                fig.savefig(filepath)

                self._stored_plots.append( filepath )

        except Exception as e:
            self.error(e,'issue plotting {}'.format(plot_tile))

    @classmethod
    def component_subclasses(cls):
        #We find any components in ottermatics and will exclude them
        OTTER_ITEMS = set(list(flatten([inspect.getmembers( mod, inspect.isclass ) for mkey,mod in sys.modules.items() if 'ottermatics' in mkey])))
        COMP_CLASS = [modcls for mkey,modcls in OTTER_ITEMS if type(modcls) is type and issubclass(modcls, Component)]
        return {scls.__name__:scls for scls in cls.__subclasses__() if scls not in COMP_CLASS}


    #Configuration Information
    @property
    def internal_components(self):
        '''go through all attributes determining which are configuration objects
        we skip any configuration that start with an underscore (private variable)'''
        return {k:v for k,v in self.store.items() \
                if isinstance(v,Component) and not k.startswith('_')}


    def go_through_components(self,level = 0,levels_to_descend = -1, parent_level=0):
        '''A generator that will go through all internal configurations up to a certain level
        if levels_to_descend is less than 0 ie(-1) it will go down, if it 0, None, or False it will
        only go through this configuration
        
        :return: level,config'''

        should_yield_level = lambda level: all([level>=parent_level, \
                                              any([levels_to_descend < 0, level <= levels_to_descend])])

        if should_yield_level(level):
            yield level,self

        level += 1
        for comp in self.internal_components.values():
            for level,icomp in comp.go_through_components(level,levels_to_descend,parent_level):
                yield level,icomp

    @property
    def all_internal_components(self):
        return list([comp for lvl, comp in self.go_through_components() if not self is comp ])

    @property
    def unique_internal_components_classes(self):

        return list(set([ comp.__class__ for lvl, comp in self.go_through_components() \
                                                        if not self.__class__ is comp.__class__ ]))



@otterize
class ComponentIterator(Component):
    '''An object to loop through a list of components as the system is evaluated,
    
    iterates through each component and pipes data_row and data_label to this objects table'''

    _iterated_component_type = None #provides interface for tabulation & data reflection
    _components = []
    shuffle_mode = False
    _shuffled = None


    @property
    def component_list(self):
        return self._components

    @component_list.setter
    def component_list(self, new_components):
        if all([isinstance(item,Component) for item in new_components]):
            self._components = new_components
        else:
            self.warning('Input Components Were Not All Of Type Component')

    @property
    def current_component(self) -> Component:
        out = self[ self.index ]
        if out is None:
            if self.index == 0:
                return self[ 0 ]
            return self[ -1 ]
        return out



    #Wrappers for current component
    @property
    def data_dict(self):
        base = super(ComponentIterator,self).data_dict
        base.update(self.current_component.data_dict)
        return base


    @property
    def data_row(self):
        return super(ComponentIterator,self).data_row + self.current_component.data_row

    @property
    def data_label(self):
        return super(ComponentIterator,self).data_label + self.current_component.data_label

    @property
    def plot_variables(self):    
        return super(ComponentIterator,self).plot_variables + self.current_component.plot_variables

    @classmethod
    def cls_all_property_labels(cls):
        these_properties =  [obj.label.lower() for k,obj in cls.__dict__.items() if isinstance(obj,table_property)]
        if cls._iterated_component_type is not None and issubclass(cls._iterated_component_type,Component ):
            iterated_properties = cls._iterated_component_type.cls_all_property_labels()
            these_properties = list(set( iterated_properties + these_properties))
        return these_properties
    
    @classmethod
    def cls_all_attrs_fields(cls):
        these_properties =  attr.fields_dict(cls)
        if cls._iterated_component_type is not None and issubclass(cls._iterated_component_type,Component ):
            iterated_properties = attr.fields_dict(cls._iterated_component_type)
            these_properties = list(set( iterated_properties + these_properties))
        return these_properties        


    #Magicz
    #@functools.cached_property
    def _item_generator(self):
        if self.shuffle_mode:
            if self._shuffled is None:
                self._shuffled = random.sample(self.component_list,len(self.component_list))
            return self._shuffled
        return self.component_list

    def __getitem__(self,index):
        if index >= 0 and index < len(self.component_list):
            return self.component_list[index]
        if index < 0 and index > -len(self.component_list):
            return self.component_list[index]
        
    def __iter__(self):
        #TODO: Add shuffle mode!
        for item in self._item_generator():
            self._anything_changed = True
            yield item
            self._anything_changed = True
