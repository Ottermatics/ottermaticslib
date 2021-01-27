from contextlib import contextmanager
import attr

from ottermatics.logging import LoggingMixin, log

import numpy
import functools
import itertools
import datetime
import pandas
import os

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
    acls = attr.s(cls, on_setattr= property_changed,*args,**kwargs)
    
    #Register Tabular Properties
    acls._val_tab_funtions = {}
    acls._stat_tab_functions = {}
    #acls._vect_tab_functions = {}

    #Go through all in class and assign functions appropriately
    for methodname in dir(acls):
        method = getattr(acls, methodname)
        if hasattr(method, '_val_prop'):
            acls._val_tab_funtions.update(
                {methodname: method._val_prop})
        if hasattr(method, '_stat_prop'):
            acls._stat_tab_functions.update(
                {methodname: method._stat_prop}) 
        # if hasattr(method, '_vect_prop'):
        #     acls._stat_tab_functions.update(
        #         {methodname: method._vect_prop})                              

    return acls


#Properties to register certian class functions
class OtterProp(property):
    '''We cant modify property code sooo we have OtterProps!'''
    pass

    stats_labels = ['min','avg','std','max']

    @classmethod
    def stats_return(cls,value):
        '''we run stats on a vectorized input, and return the value
        If the value is a numeric type we will just return it in a list'''
        if isinstance(value,(float,int)):
            return [value]
        elif isinstance(value,numpy.ndarray):
            max =  numpy.nanmax(value)
            min =  numpy.nanmin(value)
            avg =  numpy.nanmean(value)
            std =  numpy.nanstd(value)
            return [min,avg,max,std]
        else:
            log.warning('OtterProp: unknown type in stats method {}{}'.format(value,type(value)))
            return None 

    @classmethod
    def create_stats_label(cls,value,title):
        if isinstance(value,(float,int)):
            return [title]
        elif isinstance(value,numpy.ndarray):
            max =  '{} max'.format(title)
            min =  '{} min'.format(title)
            avg =  '{} avg'.format(title)
            std =  '{} std'.format(title)
            return [min,avg,max,std]
        else:
            log.warning('OtterProp: unknown type in stats method {}{}'.format(value,type(value)))
            return None 



def table_property(f=None,**meta):
    '''A decorator that wraps a function like a regular property
    The output will be tabulized for use in the ottermatics table methods
    values must be single valued (str, float, int) 
    
    _val_prop is the key variable set, otter_class looks it up'''
    if f is not None:
        prop =  OtterProp(f)
        prop._val_prop = {'func':f}
        return prop
    else:
        def wrapper(f):
            prop =  OtterProp(f)
            prop._val_prop = {'func':f,**meta}
            return prop
        return wrapper



def table_stats_property(f=None,**meta):
    '''A decorator that wraps a function like a regular property
    The output must be a numpy array or list of number which can be 
    tabulized into columns like average, min, max and std for use in the ottermatics table methods
    
    _stat_prop is the key variable set, otter_class looks it up'''
    if f is not None:  
        prop =  OtterProp(f)
        prop._stat_prop = {'func':f}
        return prop
    else:
        def wrapper(f):
            prop =  OtterProp(f)
            prop._stat_prop = {'func':f,**meta}
            return prop
        return wrapper

#Not sure where to put this, this will vecorize a class fcuntion
class vectorize_inst(numpy.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)







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

    importantly since its an attr's class we don't use __init__, instead overide __attrs_post_init__ 
    and use it the same way.
    '''
    _temp_vars = None

    name = attr.ib(default='default',validator= attr.validators.instance_of(str))
    index = attr.ib(default=0,validator= attr.validators.instance_of(int))

    log_fmt = "[%(identity)-24s]%(message)s"
    log_silo = True

    #yours to implement
    _skip_attr = None

    #table property internal variables
    __store_options = ('csv','excel')#,'db','gsheets','excel','json')
    _store_types = None
    _store_level = 0

    #Ultra private variables
    _val_tab_funtions = None
    _stat_tab_functions = None    
    _table = None
    _cls_attrs = None
    _attr_labels = None 
    _anything_changed = False

    #Solver Configuration
    def evaluate(self,*args,**kwargs):
        '''evaluate is a fleixble method to be overriden. Oftentimes it might not be used as 
        configurations are useful stores
        
        :return: not stored in table, but is returned through solve making it possibly useful 
        to track error, or other feedback for control'''
        return None 
    
    @property
    def filename(self):
        '''A nice to have, good to override'''
        return self.name.replace(' ','_')

    @property
    def identity(self):
        '''A customizeable property that will be in the log by default'''
        return '{}-{}'.format(type(self).__name__,self.name).lower()

    def add_fields(self, record):
        '''Overwrite this to modify logging fields, change log_fmt in your class to use the value set here.'''
        record.identity = self.identity        

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

    def save_data(self,index=None):
        '''We'll save data for this object and any other internal configuration if 
        anything changed or if the table is empty This should result in at least one row of data, 
        or minimal number of rows to track changes
        
        This should capture all changes but save data'''
        self.debug('saving data')
        if self.anything_changed or not self.TABLE:
            if index is not None: #only set this after we check if anything changed
                self.index = index
            else:
                self.index += 1
            self.TABLE.append(self.data_row)
        for config in self.internal_configurations.values():
            config.save_data(index)
        self._anything_changed = False


    @property
    def anything_changed(self):
        '''use the on_setattr method to determine if anything changed, 
        also assume that stat_tab could change without input changes'''
        if self._anything_changed or self.has_random_stats_properties:
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
                if len(set(col)) <= 1:
                    yield 2, col[0] ,label #Can assume that first value is equal to all
                else:
                    yield 3, col,label

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

    @property
    def data_row(self):
        '''method that returns collects valid tabiable attributes immediately from this config
        will ignore any attributes labeled in _skip_attr which is a list
        best to not override these, but you can if you really want'''

        prop_vals = [val for key,val in self.tab_value_properties.items()]
        
        stats_vals = [ OtterProp.stats_return( val ) for key,val in self.tab_stats_properties.items() ]
        stats_vals = list(filter(None,itertools.chain.from_iterable(stats_vals)))

        return self.attr_row + prop_vals + stats_vals

    @property
    def data_label(self):
        '''method that returns the lables for the tabilated attributes immediately from this config,
        will ignore any attributes labeled in _skip_attr which is a list
        best to not override these, but you can if you really want'''
        prop_labels = [key for key,meta in self.tab_value_meta.items()]

        stats_labels = [ OtterProp.create_stats_label(val,key) for key,val in self.tab_stats_properties.items()]                        
        stats_labels = list(filter(None,itertools.chain.from_iterable(stats_labels)))

        return self.attr_labels + prop_labels + stats_labels

    @property
    def skip_attr(self) -> list: 
        if self._skip_attr is None:
            return list(self.internal_configurations.keys())
        return self._skip_attr + list(self.internal_configurations.keys())

    @property
    def attr_labels(self) -> list:
        if self._attr_labels is None:
            self._attr_labels = list([k for k in attr.fields_dict(self.__class__).keys() \
                                                if k not in self.skip_attr])
        return self._attr_labels

    @property
    def attr_row(self) -> list:
        return list([self.store[k] for k in self.attr_labels])

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
    def has_random_stats_properties(self):
        '''looks at stat tab function metadata for 'random=True' 
        by default it is assumed that properties do not change unless underlying attributes change'''
        for meta in self._stat_tab_functions.values():
            if 'random' in meta and meta['random'] == True:
                return True
        return False

    #Recursive Configuration Information
    @property
    def internal_configurations(self):
        '''go through all attributes determining which are configuration objects'''
        return {k:v for k,v in self.store.items() if isinstance(v,Configuration)}



    def go_through_configurations(self,level = 0,levels_to_descend = -1, parent_level=0):
        '''A generator that will go through all internal configurations up to a certain level
        if levels_to_descend is less than 0 ie(-1) it will go down, if it 0, None, or False it will
        only go through this configuration
        
        :return: level,config'''

        should_yield_level = lambda level: all([level>=parent_level, \
                                              any([levels_to_descend < 0, level <= levels_to_descend])])

        if should_yield_level(level):
            yield level,self

        for config in self.internal_configurations.values():
            for level,iconf in config.go_through_configurations(level+1,levels_to_descend,parent_level):
                yield level,iconf

    def recursive_data_structure(self,levels_to_descend = -1, parent_level=0):
        '''Returns the static and variable data from each configuration to grab defined by the
        recursive commands input in this function
        
        data is stored like: output[level]=[{static,variable},{static,variable},...]'''

        output = {}

        for level,conf in self.go_through_configurations(0,levels_to_descend,parent_level):
            if level in output:
                output[level].append({'static':conf.static_data_dict,'variable':conf.variable_data_dict})
            else:
                output[level] = [{'static':conf.static_data_dict,'variable':conf.variable_data_dict}]

        return output


    #Save functionality
    def save_table(self,filename=None,*args,**kwargs):
        for save_format in self.store_types:
            if save_format == 'csv':
                self.save_csv(filename,*args,**kwargs)
            elif save_format == 'excel':
                self.save_excel(filename,*args,**kwargs)
            #elif save_format == 'gsheets':
            #    self.save_excel(filename,*args,**kwargs)                                

    def save_csv(self,filename=None,*args,**kwargs):
        if self.TABLE:
            if filename is None:
                filename = '{}.csv'.format(self.filename)
            if type(filename) is str and not filename.endswith('.csv'):
                filename += '.csv'
            self.dataframe.to_csv(path_or_buf=filename,*args,**kwargs)

    def save_excel(self,filename=None,*args,**kwargs):
        if self.TABLE:
            if filename is None:
                filename = '{}.xlsx'.format(self.filename)
            if type(filename) is str and not filename.endswith('.xlsx'):
                filename += '.xlsx'
            self.dataframe.to_excel(path_or_buf=filename,*args,**kwargs)

    def save_gsheets(self,filename=None,*args,**kwargs):
        pass

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