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

@otterize
class ComponentIterator(Component):
    '''An object to loop through a list of components as the system is evaluated,
    
    iterates through each component and pipes data_row and data_label to this objects table'''

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
    def data_row(self):
        return super(ComponentIterator,self).data_row + self.current_component.data_row

    @property
    def data_label(self):
        return super(ComponentIterator,self).data_label + self.current_component.data_label

    @property
    def plot_variables(self):    
        return super(ComponentIterator,self).plot_variables + self.current_component.plot_variables

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
