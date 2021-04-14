from contextlib import contextmanager
import attr

from ottermatics.logging import LoggingMixin, log
from ottermatics.tabulation import TabulationMixin

import numpy
import functools
import itertools
import datetime
import pandas
import os
import inspect
import pathlib

import matplotlib.pyplot as plt


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
