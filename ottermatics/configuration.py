from contextlib import contextmanager
import attr

from ottermatics.logging import LoggingMixin, log
from ottermatics.locations import *
from ottermatics.gdocs import OtterDrive

import numpy
import functools
import itertools
import datetime
import pandas
import os
import inspect
import pathlib
import marshal, types

import matplotlib.pyplot as plt

pandas.set_option('use_inf_as_na', True)

#Type Checking
NUMERIC_TYPES = (float,int)

def NUMERIC_VALIDATOR():
    return attr.validators.instance_of(NUMERIC_TYPES)

#Decorators
'''Ok get ready for some fancy bullshit, this represents alot of meta functionality to get nice
python syntax througout our application code. The concept of the 'otter_class' decorator and the 
Configuration class at the moment is sacred, they are nessicary together because the otter_class
handles runtime meta operatoins, while Configuration is more like a normal class. All you need to know
how to do is use the 3rd party attr's libray. Sorry if you dont like it!
'''


#Class Definition Wrapper Methods
def property_changed(instance, variable, value):
    instance._anything_changed = True
    return value

#This one should wrap all configuraitons to track changes, and special methods
def otter_class(cls,*args,**kwargs):
    '''Wrap all Configurations with this decorator'''
    acls = attr.s(cls, on_setattr= property_changed, repr=False, *args,**kwargs)
    
    #Register Tabular Properties
    acls._val_tab_funtions = {}
    acls._stat_tab_functions = {}
    acls._avg_tab_functions = {}

    #Go through all in class and assign functions appropriately
    for methodname in dir(acls):
        method = getattr(acls, methodname)
        if hasattr(method, '_val_prop'):
            this = method._val_prop
            this['func'] = method.fget
            acls._val_tab_funtions.update(
                { methodname: this })
        if hasattr(method, '_stat_prop'):
            this = method._stat_prop
            this['func'] = method.fget            
            acls._stat_tab_functions.update(
                { methodname: this }) 
        if hasattr(method, '_avg_prop'):
            this = method._avg_prop
            this['func'] = method.fget               
            acls._avg_tab_functions.update(
                { methodname: this })                                          

    return acls

def meta(title,desc=None,**kwargs):
    '''a convienience wrapper to add metadata to attr.ib
    :param title: a title that gets formatted for column headers
    :param desc: a description of the property'''
    out = {'title':title.replace('_',' ').replace('-',' ').title(),
            'desc':None,
            **kwargs}
    return out


class BaseProperty:
    "Emulate PyProperty_Type() in Objects/descrobject.c"

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)    

#Properties to register certian class functions
class OtterCacheProp(functools.cached_property):
    pass
    # def __getstate__(self):
    #     attributes = self.__dict__.copy()
    #     attr_out = {}
    #     for key,att in attributes.items():
    #         #print('GS:',key,att)
    #         if key not in ('_val_props','_stat_prop','_avg_prop'):
    #             attr_out[key] = att
    #     return attr_out

class OtterProp(BaseProperty):
    '''We cant modify property code sooo we have OtterProps!'''

    stats_labels = ['min','avg','std','max']

    @classmethod
    def stats_return(cls,value):
        '''we run stats on a vectorized input, and return the value
        If the value is a numeric type we will just return it in a list'''
        # if isinstance(value,NUMERIC_TYPES):
        #     return [value]
        if isinstance(value,numpy.ndarray):
            max =  numpy.nanmax(value)
            min =  numpy.nanmin(value)
            avg =  numpy.nanmean(value)
            std =  numpy.nanstd(value)
            return [min,avg,max,std]
        else:
            log.warning('OtterProp: unknown type in stats method {}{}'.format(value,type(value)))
            return [None] #is sacreed

    @classmethod
    def create_stats_label(cls,value,title):
        # if isinstance(value,NUMERIC_TYPES):
        #     return [title]
        if isinstance(value,numpy.ndarray):
            max =  '{} max'.format(title)
            min =  '{} min'.format(title)
            avg =  '{} avg'.format(title)
            std =  '{} std'.format(title)
            return [min,avg,max,std]
        else:
            log.warning('OtterProp: unknown type in stats method {}{}'.format(value,type(value)))
            return [None] #is sacreed

    @classmethod
    def avg_return(cls,value):
        '''we run stats on a vectorized input, and return the value
        If the value is a numeric type we will just return it in a list'''
        if isinstance(value,NUMERIC_TYPES):
            return value
        elif isinstance(value,numpy.ndarray):
            avg =  numpy.nanmean(value)
            return avg
        else:
            log.warning('OtterProp: unknown type in stats method {}{}'.format(value,type(value)))
            return None #is sacreed

    @classmethod
    def create_avg_label(cls,value,title):
        if isinstance(value,NUMERIC_TYPES):
            return title
        elif isinstance(value,numpy.ndarray):
            avg =  '{} avg'.format(title)
            return avg
        else:
            log.warning('OtterProp: unknown type in stats method {}{}'.format(value,type(value)))
            return None

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attr_out = {}
        for key,att in attributes.items():
            #print('GS:',key,att)
            if key in ('_val_props','_stat_prop','_avg_prop'):
                print(f'skip get {self} {key} with {att}')
                #code_string = marshal.dumps(att['func'].__code__  )
                #att['func'] = code_string
                #attr_out[key] = att
            else:
                attr_out[key] = att
        return attr_out

    # def __setstate__(self,new_state):
    #     attr_out = {}
    #     for key,att in new_state.items():
    #         #print('GS:',key,att)
    #         if key in ('_val_props','_stat_prop','_avg_prop'):
    #             print(f'skil set {self} {key} with {att}')
    #             #code = marshal.loads(att['func'])
    #             #func = types.MethodType( code, self, key )
    #             #att['func'] = func
    #             #attr_out[key] = att
    #         else:
    #             attr_out[key] = att

    #     self.__dict__ = attr_out      

    # def __reduce__(self):
    #    return (self.__class__, 

def table_property(f=None,**meta):
    '''A decorator that wraps a function like a regular property
    The output will be tabulized for use in the ottermatics table methods
    values must be single valued (str, float, int) 
    
    _val_prop is the key variable set, otter_class looks it up
    :param random: set this to true if you walways want it a property to refreshh even if underlying data
    hasn't changed
    :param title: the name the field will have in a table     
    :param desc: some information about the field'''
    if f is not None:
        prop =  OtterProp(f)
        prop._val_prop = {'func':True}
        return prop
    else:
        def wrapper(f):
            prop =  OtterProp(f)
            prop._val_prop = {'func':True,**meta}
            return prop
        return wrapper

def table_stats_property(f=None,**meta):
    '''A decorator that wraps a function like a regular property
    The output must be a numpy array or list of number which can be 
    tabulized into columns like average, min, max and std for use in the ottermatics table methods
    
    _stat_prop is the key variable set, otter_class looks it up
    :param random: set this to true if you walways want it a property to refreshh even if underlying data
    hasn't changed
    :param title: the name the field will have in a table
    :param desc: some information about the field'''
    if f is not None:  
        prop =  OtterProp(f)
        prop._stat_prop = {'func':True}
        return prop
    else:
        def wrapper(f):
            prop =  OtterProp(f)
            prop._stat_prop = {'func':True,**meta}
            return prop
        return wrapper

def table_avg_property(f=None,**meta):
    '''A decorator that caches and wraps a function like a regular property
    The output must be a numpy array or list of number which can be 
    tabulized into columns like based on the array average for use in the ottermatics table methods
    
    _stat_prop is the key variable set, otter_class looks it up
    :param random: set this to true if you walways want it a property to refreshh even if underlying data
    hasn't changed
    :param title: the name the field will have in a table
    :param desc: some information about the field'''
    if f is not None:  
        prop =  OtterProp(f)
        prop._avg_prop = {'func':True}
        return prop
    else:
        def wrapper(f):
            prop =  OtterProp(f)
            prop._avg_prop = {'func':True,**meta}
            return prop
        return wrapper           

def table_cached_property(f=None,**meta):
    '''A decorator that caches and wraps a function like a regular property
    The output will be tabulized for use in the ottermatics table methods
    values must be single valued (str, float, int)
    
    _val_prop is the key variable set, otter_class looks it up
    :param random: set this to true if you walways want it a property to refreshh even if underlying data
    hasn't changed
    :param title: the name the field will have in a table
    :param desc: some information about the field'''
    if f is not None:
        prop =  OtterCacheProp(f)
        prop._val_prop = {'func':True}
        return prop
    else:
        def wrapper(f):
            prop =  OtterCacheProp(f)
            prop._val_prop = {'func':True,**meta}
            return prop
        return wrapper

def table_cached_stats_property(f=None,**meta):
    '''A decorator that caches and wraps a function like a regular property
    The output must be a numpy array or list of number which can be 
    tabulized into columns like average, min, max and std for use in the ottermatics table methods
    
    _stat_prop is the key variable set, otter_class looks it up
    :param random: se
    t this to true if you walways want it a property to refreshh even if underlying data
    hasn't changed
    :param title: the name the field will have in a table
    :param desc: some information about the field'''
    if f is not None:  
        prop =  OtterCacheProp(f)
        prop._avg_prop = {'func':True}
        return prop
    else:
        def wrapper(f):
            prop =  OtterCacheProp(f)
            prop._avg_prop = {'func':True,**meta}
            return prop
        return wrapper


def table_cached_avg_property(f=None,**meta):
    '''A decorator that caches and wraps a function like a regular property
    The output must be a numpy array or list of number which can be 
    tabulized into columns like based on the array average for use in the ottermatics table methods
    
    _stat_prop is the key variable set, otter_class looks it up
    :param random: set this to true if you walways want it a property to refreshh even if underlying data
    hasn't changed
    :param title: the name the field will have in a table
    :param desc: some information about the field'''
    if f is not None:  
        prop =  OtterCacheProp(f)
        prop._stat_prop = {'func':True}
        return prop
    else:
        def wrapper(f):
            prop =  OtterCacheProp(f)
            prop._stat_prop = {'func':True,**meta}
            return prop
        return wrapper              

#Not sure where to put this, this will vecorize a class fcuntion
class inst_vectorize(numpy.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]




@otter_class
class Configuration(LoggingMixin):
    '''Configuration is a pattern for storing attributes that might change frequently,
    provides a way to `solve` a configuration, but this might be a simple update calculation.
    Attributes are stored with attr.ib or as python properties. 

    Must be called with decorator `@otter_class` which is a wrapper for attr.s but registers a 
    setattr wrapper
    
    Special decorators:
    as wrappers for regular properties however the will tabulate data, like `table_property` and 
    `table_stats_data` that will store variables in a row of lists referred to as `TABLE` that 
    updates with all attributes of the configuration when solve is called. Solve by default doesn't 
    do anythign, it just adds a row to the table. data is stored with a increasing primary key index

    solve() is the wrapper for evaluate() which will be overriden, the variables will be saved in 
    TABLE after evaluating them.  

    importantly since its an attr's class we don't use __init__, instead overide __on_init__ 
    and use it the same way, this replaces __attrs_post_init__ which is called by attrs. Our implementation
    calls __on_init__ for you to use!

    Report Generation:
    you can call the wrapped subplots generator in this method which will automatically record the saved
    plots, catch errors and save files in a valid image format.
    with self.subplots(*args) as (fig,axes):
    '''
    _temp_vars = None

    name = attr.ib(default='default',validator= attr.validators.instance_of(str),kw_only=True)

    #Super Special Tabulating Index
    index = 0 #attr.ib(default=0,validator= attr.validators.instance_of(int),kw_only=True)

    log_fmt = "[%(identity)-24s]%(message)s"
    log_silo = True

    stored_path = None
    stored_client_folder = None
    stored_client_name = None

    #yours to implement
    _skip_attr = None
    
    #table property internal variables
    __store_options = ('csv','excel','gsheets')#,'db','gsheets','excel','json')
    _store_types = None
    _store_level = 0

    #plotting & post processing calls
    _stored_plots = []
    _report_path = 'reports/'
    _default_image_format = '.png'

    #Ultra private variables
    _val_tab_funtions = None
    _stat_tab_functions = None
    _avg_tab_functions = None

    _table = None
    _cls_attrs = None
    _attr_labels = None 
    _attr_keys = None
    _anything_changed = False
    _skip_table = False #This prevents config from reportin up through internal configurations
    _started_datetime = None

    def __on_init__(self):
        '''Override this when creating your special init functionality, you must use attrs for input variables'''
        pass

    def __attrs_post_init__(self):
        '''This is called after __init__ by attr's functionality, we expose __oninit__ for you to use!'''
        #Store abs path On Creation, in case we change. 
        self.report_path
        self._started_datetime = datetime.date.today()
        self.__on_init__()

    #A solver function that will be called on every configuration
    def evaluate(self,*args,**kwargs):
        '''evaluate is a fleixble method to be overriden. Oftentimes it might not be used as 
        configurations are useful stores. These must be called from the top level, or solve will call 
        
        :return: not stored in table, but is returned through solve making it possibly useful 
        to track error, or other feedback for control'''
        return None
    
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
        return '{}-{}'.format(self.classname,self.name).lower()

    @property
    def classname(self):
        return str( type(self).__name__ )

    def add_fields(self, record):
        '''Overwrite this to modify logging fields, change log_fmt in your class to use the value set here.'''
        record.identity = self.identity

    @property
    def client_name(self):
        if self.stored_client_name is None:
            if 'CLIENT_NAME' in os.environ:
                self.stored_client_folder = os.environ['CLIENT_NAME']
            elif in_client_dir(): #infer from path
                self.stored_client_name = client_dir_name()
            else:
                self.stored_client_folder  = 'Ottermatics'
                self.warning('no client info found - using{}'.format(self.stored_client_folder))
        return self.stored_client_folder
            

    @property
    def client_folder_root(self):
        if self.stored_client_folder is None:
            self.stored_client_folder = ottermatics_project(self.client_name)
        return self.stored_client_folder

    @property
    def report_path(self):
        if self.stored_path is None:
            self.stored_path =  os.path.realpath(os.path.join(self.client_folder_root,self._report_path))

        if not os.path.exists(self.stored_path):
            self.warning('report does not exist {}'.format(self.stored_path))
        return self.stored_path  

    @property
    def report_path_daily(self):
        return os.path.join(self.report_path,'{}'.format(self._started_datetime).replace('-','_'))                  

    @property
    def config_path(self):
        return os.path.join(self.report_path,self.filename)

    @property
    def config_path_daily(self):
        return os.path.join(self.report_path_daily,self.filename)


    #Sync Convenicne Functions
    def gsync_this_config(self,force = True):
        #Sync all local to google
        od = OtterDrive.instance()
        od.sync_to_client_folder(force=force,sub_path=self.config_path_daily)

    def gsync_this_report(self,force = True):
        #Sync all local to google
        od = OtterDrive.instance()
        od.sync_to_client_folder(force=force,sub_path=self.report_path_daily)

    def gsync_all_reports(self,force = True):
        #Sync all local to google
        od = OtterDrive.instance()
        od.sync_to_client_folder(force=force,sub_path=self.report_path)
        
    def gsync_client_folder(self,force = True):
        #Sync all local to google
        od = OtterDrive.instance()
        od.sync_to_client_folder(force=force,sub_path=self.client_folder_root)     




    #Clearance Methods
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

    #BUG: This breaks cloudpickle due to inspect+property recursion, lets find a different answer
    #@property
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


    #Data Tabulation - Intelligent Lookups
    def reset_table(self):
        '''Resets the table, and attrs label stores'''
        self.index = 0.0
        self._cls_attrs = None
        self._attr_labels = None
        self._table = []

    @property
    def TABLE(self):
        '''this should seem significant'''
        if self._table is None:
            self._table = []
        return self._table

    @property
    def dataframe(self):
        return pandas.DataFrame(data = self.TABLE, columns = self.data_label,copy=True)

    @property
    def variable_dataframe(self):
        vals = list(zip(*list((self.variable_data_dict.values()))))
        cols = list((self.variable_data_dict.keys()))
        if vals:
            return pandas.DataFrame(data = vals, columns=cols, copy = True)
        return None

    @property
    def static_dataframe(self):
        vals = [list((self.static_data_dict.values()))]
        cols = list((self.static_data_dict.keys()))
        if vals:
            return pandas.DataFrame(data = vals, columns=cols, copy = True)
        return None

    def split_dataframe_by_colmum(self,df,max_columns=10):
        df_cols = list(df.columns)
        if len(df_cols) < max_columns:
            return [df]
        
        col_chunks = list(chunks(df_cols,max_columns))
        dat_chunks = [df[colck].copy() for colck in col_chunks]

        return dat_chunks

    def cleanup_dataframe(self,df,badwords=('index','min','max')):
        tl_df = self.static_dataframe
        to_drop = [cl for cl in tl_df.columns if any([bw in cl.lower() for bw in badwords])]
        out_df = tl_df.drop(columns=to_drop)        
        return out_df

    def save_data(self,index=None):
        '''We'll save data for this object and any other internal configuration if 
        anything changed or if the table is empty This should result in at least one row of data, 
        or minimal number of rows to track changes
        
        This should capture all changes but save data'''
        if self.anything_changed or not self.TABLE or index is not None:
            self.TABLE.append(self.data_row)
            self.debug('saving data {}'.format(self.index))

            if index is not None: #only set this after we check if anything changed
                self.index = index
            else:
                self.index += 1
            
        for config in self.internal_configurations.values():
            config.save_data(index)
        self._anything_changed = False

    @property
    def anything_changed(self):
        '''use the on_setattr method to determine if anything changed, 
        also assume that stat_tab could change without input changes'''
        if self._anything_changed or self.has_random_properties:
            return True
        return False

    @property
    def table_iterator(self):
        '''Checks data to see what type of table it is, type 1 is a single row
        whereas type 2 & 3 have several rows, but type 2 has some values that are singular
        
        iterates table_type,values,label'''
        if not self.TABLE:
            return 0,[],[]
        if len(self.TABLE) <= 1:
            yield 1,self.TABLE[0],self.data_label
        else:
            for col,label in zip(zip(*self.TABLE),self.data_label):
                col = numpy.array(col).flatten().tolist()
                if len(set(col)) <= 1:
                    yield 2, col[0], label #Can assume that first value is equal to all
                else:
                    yield 3, col, label

    @property
    def static_data_dict(self):
        '''returns key-value pairs in table that are single valued, all in type 1, some in type 2'''
        output = {}
        for tab_type,value,label in self.table_iterator:
            if tab_type == 1:
                return {l:v for l,v in zip(label,value)}
            elif tab_type == 2:
                output[label] = value

        return output

    @property
    def variable_data_dict(self):
        '''returns a dictionary of key value pairs where values list is not the same all of type 3 and some of type 2''' 
        output = {}
        for tab_type,value,label in self.table_iterator:
            if tab_type == 3:
                output[label] = value
        return output

    def format_label(self,label):
        return label.replace('-','_')

    def output_format_label(self,label):
        return label.replace('_',' ').replace('-',' ').title()

    @property
    def data_row(self):
        '''method that returns collects valid tabiable attributes immediately from this config
        will ignore any attributes labeled in _skip_attr which is a list
        best to not override these, but you can if you really want'''

        prop_vals = [float(val) if isinstance(val,numpy.ndarray) else val \
                                for key,val in self.tab_value_properties.items()]
        
        stats_vals = [ OtterProp.stats_return( val ) for key,val in self.tab_stats_properties.items() ]
        stats_vals = list(filter(lambda v: v is not None,itertools.chain.from_iterable(stats_vals)))

        avg_vals = [ OtterProp.avg_return( val ) for key,val in self.tab_avg_properties.items() ]

        return numpy.array( self.attr_row + prop_vals + avg_vals + stats_vals ).flatten().tolist()

    @property
    def data_label(self):
        '''method that returns the lables for the tabilated attributes immediately from this config,
        will ignore any attributes labeled in _skip_attr which is a list
        best to not override these, but you can if you really want'''
        prop_labels = [key for key,meta in self.tab_value_meta.items()]

        stats_labels = [ OtterProp.create_stats_label(val,key) for key,val in self.tab_stats_properties.items()]                        
        stats_labels = list(filter(lambda v: v is not None,itertools.chain.from_iterable(stats_labels)))

        avg_labels = [ OtterProp.create_avg_label(val,key) for key,val in self.tab_avg_properties.items()]      

        return list(map( self.format_label, self.attr_labels + prop_labels + avg_labels + stats_labels ))

    @property
    def skip_attr(self) -> list: 
        if self._skip_attr is None:
            return list(self.internal_configurations.keys())
        return self._skip_attr + list(self.internal_configurations.keys())

    def get_label(self,k,v):
        if v.metadata and 'title' in v.metadata:
            return self.format_label(v.metadata['title'])
        return self.format_label(k)

    @property
    def attr_labels(self) -> list:
        '''Returns formated attr label if the value is numeric'''
        if self._attr_labels is None:
            _attr_labels = list([self.get_label(k,v) for k,v in attr.fields_dict(self.__class__).items() \
                                                if k not in self.skip_attr and k.lower() != 'index' \
                                                and isinstance(self.store[k],NUMERIC_TYPES)])
            self._attr_labels = _attr_labels                                            
        return self._attr_labels

    @property
    def attr_row(self) -> list:
        '''Returns formated attr data if the value is numeric'''
        return list([self.store[k] for k in attr.fields_dict(self.__class__).keys() \
                                            if k not in self.skip_attr and k.lower()  != 'index' \
                                            and isinstance(self.store[k],NUMERIC_TYPES)])
    @property
    def attr_raw_keys(self) -> list:
        return [k for k in attr.fields_dict(self.__class__).keys()]

    @property
    def tab_value_properties(self):
        return {(meta['title'] if 'title' in meta else fname):meta['func'](self) \
                                for fname,meta in self._val_tab_funtions.items()}

    @property
    def tab_value_meta(self):
        return {(meta['title'] if 'title' in meta else fname):meta \
                                for fname,meta in self._val_tab_funtions.items()}        

    @property
    def tab_stats_properties(self):
        return {(meta['title'] if 'title' in meta else fname):meta['func'](self) 
                                    for fname,meta in self._stat_tab_functions.items()}

    @property
    def tab_stats_meta(self):
        return { (meta['title'] if 'title' in meta else fname) :meta  \
                                    for fname,meta in self._stat_tab_functions.items()}

    @property
    def tab_avg_properties(self):
        return {(meta['title'] if 'title' in meta else fname):meta['func'](self) 
                                    for fname,meta in self._avg_tab_functions.items()}

    @property
    def tab_avg_meta(self):
        return { (meta['title'] if 'title' in meta else fname) :meta  \
                                    for fname,meta in self._avg_tab_functions.items()}                                    
            
    @property
    def has_random_properties(self):
        '''looks at stat tab function metadata for 'random=True' 
        by default it is assumed that properties do not change unless underlying attributes change'''
        for meta in self._stat_tab_functions.values():
            if 'random' in meta and meta['random'] == True:
                return True
        for meta in self._avg_tab_functions.values():
            if 'random' in meta and meta['random'] == True:
                return True                
        return False

    #Configuration Information
    @property
    def internal_configurations(self):
        '''go through all attributes determining which are configuration objects
        we skip any configuration that start with an underscore (private variable), _skip_table=True,
        or isnt in attr_keys'''
        return {k:v for k,v in self.store.items() \
                if isinstance(v,Configuration) and \
                    k in self.attr_raw_keys and \
                    not k.startswith('_') and \
                    not v._skip_table}


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

    def recursive_data_structure(self,levels_to_descend = -1, parent_level=0):
        '''Returns the static and variable data from each configuration to grab defined by the
        recursive commands input in this function
        
        data is stored like: output[level]=[{static,variable},{static,variable},...]'''

        output = {}

        for level,conf in self.go_through_configurations(0,levels_to_descend,parent_level):
            if level in output:
                output[level].append({'static':conf.static_dataframe,\
                                      'variable':conf.variable_dataframe,'conf':conf})
            else:
                output[level] = [{'static':conf.static_dataframe,\
                                  'variable':conf.variable_dataframe,\
                                  'conf':conf}]

        return output


    #Save functionality
    def save_table(self,dataframe=None,filename=None,*args,**kwargs):
        '''Header method to save the config in many different formats'''
        if dataframe is None:
            dataframe = self.dataframe.copy()
            #Make Coulmns Human Readabile
            dataframe.columns = [self.output_format_label(col) for col in dataframe.columns]
            
        for save_format in self.store_types:
            try:
                if save_format == 'csv':
                    self.save_csv(dataframe,filename,*args,**kwargs)
                elif save_format == 'excel':
                    self.save_excel(dataframe,filename,*args,**kwargs)
                elif save_format == 'gsheets':
                    self.save_gsheets(dataframe,filename,*args,**kwargs)
            except Exception as e:
                self.error(e,'Issue Saving Tables:')                      

    def save_csv(self,dataframe,filename=None,*args,**kwargs):
        if self.TABLE:
            if filename is None:
                filename = '{}.csv'.format(self.filename)
            if type(filename) is str and not filename.endswith('.csv'):
                filename += '.csv'
            filepath = os.path.join(self.config_path_daily,filename)
            dataframe.to_csv(path_or_buf=filepath,*args,**kwargs)

    def save_excel(self,dataframe,filename=None,*args,**kwargs):
        if self.TABLE:
            if filename is None:
                filename = '{}.xlsx'.format(self.filename)
            if type(filename) is str and not filename.endswith('.xlsx'):
                filename += '.xlsx'
            filepath = os.path.join(self.config_path_daily,filename)
            dataframe.to_excel(path_or_buf=filepath,*args,**kwargs)

    def save_gsheets(self,dataframe,filename=None,*args,**kwargs):
        
        od = OtterDrive.instance()
        
        
        filepath = self.config_path_daily

        self.info('saving as gsheets in dir {}'.format(filepath))

        gpath = od.sync_path(filepath)

        od.ensure_g_path_get_id(gpath)
        pth_id = od.folder_cache[gpath]

        sht = od.gsheets.create(filename,folder=pth_id)
        wk = sht.sheet1
        wk.set_dataframe(dataframe,start='A1')

        self.info('gsheet saved -> {}'.format(os.path.join(gpath,filename)))

        
    @property
    def store_level(self):
        return self._store_level

    @store_level.setter
    def store_level(self,new_level:int):
        assert isinstance(new_level,(int))
        self._store_level = new_level

    @property
    def store_types(self):
        if self._store_types is None:
            self._store_types = [] #initalization
        return self._store_types

    @store_types.setter
    def store_types(self, new_type_or_list):
        '''If you add a list or iterable, and each value is a valid output option we will assign it
        otherwise if its a value in the valid options it will be added.
        '''
        if isinstance(new_type_or_list,(list,tuple)):
            assert all([val in self.__store_options for val in new_type_or_list])
            self._store_types = list(new_type_or_list)

        elif new_type_or_list in self.__store_options:
             self._store_types.append(new_type_or_list)
        
        else:
            self.warning('store types input not valid {}'.format(new_type_or_list))


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




if __name__ == '__main__':

    import unittest
    import io
    import tempfile

    @otter_class
    class TestConfig(Configuration):

        attrs_prop = attr.ib(1.0)
        attrs_str = attr.ib('hey now')

        @table_property
        def test_one(self):
            return 1
            
        @table_stats_property
        def test_two(self):
            return numpy.random.rand(100)
            
        @table_property(title='three',desc='some words',random=True)
        def test_three(self):
            return 2
    
        @table_stats_property(title='four',desc='would make sense',random=True)
        def test_four(self):
            return numpy.random.rand(100)

    class Test(unittest.TestCase):
        test_file_name= 'test_dataframe_file'
        test_dir = '~/'

        def setUp(self):
            self.test_config = TestConfig()
            self.test_dir =  tempfile.mkdtemp()

        def tearDown(self):
            contents_local = os.listdir(self.test_dir)
            for fil in contents_local:
                if self.test_file_name in fil:
                    rfil = os.path.join(self.test_dir,fil)
                    #print('removing '+rfil)
                    #os.remove(rfil)

        def test_attrs_labels(self):
            #Default
            self.assertEqual( self.test_config.attr_labels, ['name', 'index', 'attrs_prop', 'attrs_str'])
        
        def test_attrs_vals(self):
            self.assertEqual( self.test_config.attr_row, ['default', 0, 1.0, 'hey now'])
        
        def test_table_labels(self):
            stats_ans = set(('four', 'test_two'))
            vals_ans = set(('test_one', 'three'))
            
            self.assertEqual(set(self.test_config.tab_value_properties.keys()),vals_ans)
            self.assertEqual(set(self.test_config.tab_stats_properties.keys()),stats_ans)

        def test_table_vals(self):
            vals_ans = set((1,2))
            self.assertEqual(set(self.test_config.tab_value_properties.values()),vals_ans)          
            
        def test_assemble_data(self):
            print('testing data assembly')
            #Run before test_table_to...
            self.assertFalse(self.test_config.TABLE)
            self.test_config.save_data()

            self.assertTrue(self.test_config.TABLE)
            self.assertTrue(self.test_config.static_data_dict)
            self.assertFalse(self.test_config.variable_data_dict)

            self.test_config.save_data()
            self.assertTrue(self.test_config.variable_data_dict)

        def file_in_format(self,fileextension,path=True):
            fille =  '{}.{}'.format(self.test_file_name,fileextension)
            if path:
                path = os.path.join(self.test_dir,fille)
                return path
            else:
                return fille

        def test_table_to_csv(self):
            pass
            #print('saving '+self.file_in_format('xlsx'))
            #self.test_config.save_csv(self.file_in_format('csv'))
            #self.assertIn(self.file_in_format('csv',False), os.listdir(self.test_dir))
            

        def test_table_to_excel(self):
            pass
            #print('saving '+self.file_in_format('xlsx'))
            #self.test_config.save_excel(self.file_in_format('xlsx'))
            #self.assertIn(self.file_in_format('xlsx',False),os.listdir(self.test_dir))

        def test_table_to_gsheets(self):
            pass
        
        def test_table_to_db(self):
            pass                             


    unittest.main()