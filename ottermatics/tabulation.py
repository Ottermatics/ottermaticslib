from contextlib import contextmanager
import attr

from ottermatics.configuration import Configuration, otterize, meta, chunks, inst_vectorize
from ottermatics.logging import LoggingMixin, log
from ottermatics.client import ClientInfoMixin
from ottermatics.locations import *
from ottermatics.gdocs import *

import numpy
import functools
import itertools
import datetime
import pandas
import os
import inspect
import pathlib

import matplotlib.pyplot as plt


pandas.set_option('use_inf_as_na', True)

#Type Checking
NUMERIC_TYPES = (float,int)
NUMERIC_NAN_TYPES = (float,int,type(None))
STR_TYPES = (str,)
TABLE_TYPES = (int,float,str,type(None))

def NUMERIC_VALIDATOR():
    return attr.validators.instance_of(NUMERIC_TYPES)

def NUMERIC_NAN_VALIDATOR():
    return attr.validators.instance_of(NUMERIC_NAN_TYPES)

def STR_VALIDATOR():
    return attr.validators.instance_of(STR_TYPES)

ATTR_VALIDATOR_TYPES = (attr.validators._AndValidator,attr.validators._InstanceOfValidator,attr.validators._MatchesReValidator,attr.validators._ProvidesValidator,attr.validators._OptionalValidator,attr.validators._InValidator,attr.validators._IsCallableValidator,attr.validators._DeepIterable,attr.validators._DeepMapping)

TAB_VALIDATOR_TYPE = attr.validators._InstanceOfValidator #our validators shoudl require a type i think, at least for tabulation

class table_property:
    """Emulate PyProperty_Type() in Objects/descrobject.c
    
        You can initalize just the functions, or precreate the object but with meta
        @table_property
        def function(...): < this uses __init__ to assign function
            
        @table_property(desc='really nice',label='funky function')
        def function(...):    < this uses __call__ to assign function"""
    
    desc = ''
    label = None

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, desc = None, label=None):
        '''You can initalize just the functions, or precreate the object but with meta
        @table_property
        def function(...): < this uses __init__ to assign function
            
        @table_property(desc='really nice',label='funky function')
        def function(...):    < this uses __call__ to assign function
        '''

        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        elif doc is not None:
            self.__doc__ = doc
        
        if desc is not None:
            self.desc = desc
        
        if label is not None:
            self.label = label

        elif fget is not None:
            self.label = fget.__name__

    def __call__(self,fget=None, fset=None, fdel=None, doc=None):
        '''this will be called when either label or desc is set'''
        if self.fget is None:
            self.fget = fget
        if self.fset is None:            
            self.fset = fset
        if self.fdel is None:            
            self.fdel = fdel

        if doc is None and fget is not None:
            doc = fget.__doc__
        elif doc is not None:
            self.__doc__ = doc     
        return self

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





@otterize
class TabulationMixin(Configuration,ClientInfoMixin):
    '''In which we define a class that can enable tabulation'''
    #Super Special Tabulating Index
    index = 0 #Not an attr on purpose, we want pandas to provide the index

    #yours to implement
    always_save_data = False
    skip_plot_vars = []
    _skip_attr = None
    
    
    #table property internal variables
    __store_options = ('csv','excel','gsheets')#,'db','gsheets','excel','json')
    _store_types = None
    _store_level:int = -1  

    #Ultra private variables
    _val_tab_funtions = None
    _stat_tab_functions = None
    _avg_tab_functions = None

    _table = None
    _cls_attrs = None
    _attr_labels = None 
    #_attr_keys = None
    _anything_changed = False
    _skip_table = False #This prevents config from reportin up through internal configurations    

    max_col_width_static:int = 10

    #Internal Dataframe Caches: - Reset on anything_changed, and reset_table()
    _dataframe = None
    _variable_dataframe= None
    _static_dataframe= None
    _static_data_dict= None
    _variable_data_dict= None
    _toplevel_static= None
    _otherstatic_tables= None
    _variable_tables= None
    _joined_dataframe= None

    #Data Tabulation - Intelligent Lookups
    def save_data(self,index=None,saved=None, force = False):
        '''We'll save data for this object and any other internal configuration if 
        anything changed or if the table is empty This should result in at least one row of data, 
        or minimal number of rows to track changes
        This should capture all changes but save data'''

        if saved is None:
            saved = set()
        

        if self.anything_changed or not self.TABLE or force:
            self.TABLE.append(self.data_dict)
            self.debug('saving data {}'.format(self.index))
            saved.add(self)    

            if index is not None: #only set this after we check if anything changed
                self.index = index
            else:
                self.index += 1
            
            
        for config in self.internal_components.values():
            if config not in saved:
                config.save_data(index, saved=saved)
            else:
                self.debug(f'skipping saved config {config.identity}')

        self._anything_changed = False

    @property
    def anything_changed(self):
        '''use the on_setattr method to determine if anything changed, 
        also assume that stat_tab could change without input changes'''
        if self._anything_changed or self.always_save_data:
            return True
        return False        

    def reset_table(self):
        '''Resets the table, and attrs label stores'''
        self.index = 0.0
        self._cls_attrs = None
        self._attr_labels = None
        self._table = None
        self.reset_meta()

        #Cached Dataframes
    def reset_meta(self):
        self._dataframe = None
        self._variable_dataframe = None
        self._static_dataframe = None
        self._static_data_dict = None
        self._variable_data_dict = None
        self._toplevel_static = None
        self._otherstatic_tables = None
        self._variable_tables = None
        self._joined_dataframe = None    

    @property
    def TABLE(self):
        '''this should seem significant'''
        if self._table is None:
            self._table = []
        return self._table

    @property
    def dataframe(self):
        if self._dataframe is None or self._anything_changed:
            #self._dataframe = pandas.DataFrame(data = self.TABLE, columns = self.data_label,copy=True)
            self._dataframe = pandas.DataFrame(data = self.TABLE, copy=True)

        return self._dataframe

    @property
    def variable_dataframe(self):
        if self._variable_dataframe is None or self._anything_changed:

            vals = list(zip(*list((self.variable_data_dict.values()))))
            cols = list((self.variable_data_dict.keys()))
            if vals:
                self._variable_dataframe = pandas.DataFrame(data = vals, columns=cols, copy = True)
            else:
                self._variable_dataframe = None

        return self._variable_dataframe

    @property
    def static_dataframe(self):
        if self._static_dataframe is None or self._anything_changed:

            vals = [list((self.static_data_dict.values()))]
            cols = list((self.static_data_dict.keys()))
            if vals:
                self._static_dataframe = pandas.DataFrame(data = vals, columns=cols, copy = True)
            else:
                self._static_dataframe = None

        return self._static_dataframe

    @property
    def static_data_dict(self):
        '''returns key-value pairs in table that are single valued, all in type 1, some in type 2'''
        
        if self._static_data_dict is None or self._anything_changed:
            output = {}
            for tab_type,value,label in self.table_iterator:
                if tab_type == 1:
                    return {l:v for l,v in zip(label,value)}
                elif tab_type == 2:
                    output[label] = value
            self._static_data_dict = output
        return self._static_data_dict

    @property
    def variable_data_dict(self):
        '''returns a dictionary of key value pairs where values list is not the same all of type 3 and some of type 2'''
        if self._variable_data_dict is None or self._anything_changed:
            output = {}
            for tab_type,value,label in self.table_iterator:
                if tab_type == 3:
                    output[label] = value
            self._variable_data_dict = output
        return self._variable_data_dict        

    #Multi-Component Table Lookups
    @property
    def toplevel_static(self):
        if self._toplevel_static is None or self._anything_changed:
            out_df = self.cleanup_dataframe( self.static_dataframe )
            df_list = self.split_dataframe_by_colmum(out_df,self.max_col_width_static)

            self._toplevel_static = {'conf':self,'dfs':  df_list }

        return self._toplevel_static

    @property
    def other_static_tables(self):
        if self._otherstatic_tables is None or self._anything_changed:
            rds = self.recursive_data_structure(self.store_level)
            output = []
            for index, components in rds.items():
                if index > 0:
                    for comp in components:
                        if comp['static'] is not None:
                            df_list = self.split_dataframe_by_colmum(comp['static'],self.max_col_width_static)                        
                            output.append({'conf':comp['conf'],'dfs': df_list})
            self._otherstatic_tables = output
        return self._otherstatic_tables            

    @property
    def variable_tables(self):
        '''Grabs all valid variable dataframes and puts them in a list'''
        if self._variable_tables is None or self._anything_changed:
            rds = self.recursive_data_structure(self.store_level)

            output = []
            for index, components in rds.items():
                for comp in components:
                    if comp['variable'] is not None:
                        output.append({'conf':comp['conf'],'df':comp['variable']})

            self._variable_tables = output
        return self._variable_tables

    @property
    def joined_dataframe(self):
        '''this is a high level data frame with all data that changes in the system'''
        if self._joined_dataframe is None or self._anything_changed:
            if self.variable_tables:
                self._joined_dataframe = pandas.concat([ vt['df'] for vt in list(reversed(self.variable_tables))],axis=1)
            else:
                self._joined_dataframe = None
        return self._joined_dataframe

    @property
    def complete_dataframe(self):
        '''this is a high level data frame with all data in the system'''
        #FIXME: join all dataframes
        if self._joined_dataframe is None or self._anything_changed:

            rds = self.recursive_data_structure()
            if rds:
                dataframes = []
                stat_dataframes = []
                for lvl, comps in rds.items():
                    for comp in comps:
                        dataframes.append(comp['conf'].dataframe)
                self._joined_dataframe = pandas.concat(dataframes,join='outer',axis=1)
                        
            else:
                self._joined_dataframe = None

        return self._joined_dataframe    
        

    def split_dataframe_by_colmum(self,df,max_columns=10):
        df_cols = list(df.columns)
        if len(df_cols) < max_columns:
            return [df]
        
        col_chunks = list(chunks(df_cols,max_columns))
        dat_chunks = [df[colck].copy() for colck in col_chunks]

        return dat_chunks

    def cleanup_dataframe(self,df,badwords=('min','max')):
        tl_df = self.static_dataframe
        to_drop = [cl for cl in tl_df.columns if any([bw in cl.lower() for bw in badwords])]
        out_df = tl_df.drop(columns=to_drop)        
        return out_df



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

            for label in set.union(*[set(tuple(row.keys())) for row in self.TABLE]):
                #col = numpy.array(col).flatten().tolist()
                col = [row[label] if label in row else None for row in self.TABLE]

                if all([isinstance(cvar,TABLE_TYPES) for cvar in col]): #do a type check
                    if len(set(col)) <= 1: #All Values Are Similar
                        yield 2, col[0], label #Can assume that first value is equal to all
                    else:
                        yield 3, col, label

    #Properties & Attribues
    #TODO: Switch to dict based recording, that way the data frame can sort out sparsity of data for intermittent vars
    @property
    def data_dict(self):
        out = self.attr_dict
        out.update(self.table_dict )
        out['index'] = self.index
        return {k.lower():v if v is not None else numpy.nan for k,v in out.items() if isinstance(v,TABLE_TYPES)}

    @property
    def data_row(self):
        '''method that returns collects valid tabiable attributes immediately from this config
        will ignore any attributes labeled in _skip_attr which is a list
        best to not override these, but you can if you really want'''
        return self.attr_row + self.table_properties

    @property
    def data_label(self):
        '''method that returns the lables for the tabilated attributes immediately from this config,
        will ignore any attributes labeled in _skip_attr which is a list
        best to not override these, but you can if you really want'''
        return list(map(self.format_label,self.attr_labels + self.table_properties_labels))

    @property
    def plot_variables(self):
        '''Checks columns for ones that only contain numeric types or haven't been explicitly skipped'''
        if self.joined_dataframe is not None:
            check_type = lambda key: all([ isinstance(v, NUMERIC_TYPES) for v in self.joined_dataframe[key] ])
            return [ var.lower() for var in self.joined_dataframe.columns 
                                 if var.lower() not in self.skip_plot_vars and check_type(var)]
        return []

    @property
    def skip_attr(self) -> list: 
        if self._skip_attr is None:
            return list(self.internal_configurations.keys())
        return self._skip_attr + list(self.internal_configurations.keys())

    def format_label_attr(self,k,attr_prop):
        if attr_prop.metadata and 'label' in attr_prop.metadata:
            return self.format_label(attr_prop.metadata['label'])
        return self.format_label(attr_prop)
    
    def format_label(self,label):
        return label.replace('_',' ').replace('-',' ').title()

    @property
    def attr_labels(self) -> list:
        '''Returns formated attr label if the value is numeric'''
        if self._attr_labels is None:
            attr_labels = list([k.lower() for k,v in attr.fields_dict(self.__class__).items() if k not in self.skip_attr])
            self._attr_labels = attr_labels                                            
        return self._attr_labels

    @property
    def attr_row(self) -> list:
        '''Returns formated attr data if the value is numeric'''
        return list([self.store[k] for k in self.attr_raw_keys if k not in self.skip_attr])

    @property
    def attr_raw_keys(self) -> list:
        return [k for k in attr.fields_dict(self.__class__).keys()]

    @property
    def attr_dict(self) -> list:
        '''Returns formated attr data if the value is numeric'''
        return {k.lower(): self.store[k] for k in self.attr_raw_keys if k not in self.skip_attr }

    @property
    def table_dict(self):
        class_dict = self.__class__.__dict__
        #We use __get__ to emulate the property, we could call regularly from self but this is more straightforward
        return { k.lower(): obj.__get__(self) for k,obj in class_dict.items() if isinstance(obj,table_property)}

    @property
    def table_properties(self):
        class_dict = self.__class__.__dict__
        #We use __get__ to emulate the property, we could call regularly from self but this is more straightforward
        tabulated_properties = [obj.__get__(self) for k,obj in class_dict.items() if isinstance(obj,table_property)]
        return tabulated_properties

    @property
    def table_properties_labels(self):
        class_dict = self.__class__.__dict__
        tabulated_properties = [obj.label.lower() for k,obj in class_dict.items() if isinstance(obj,table_property)]
        return tabulated_properties   

    @property
    def table_properties_keys(self):
        class_dict = self.__class__.__dict__
        tabulated_properties = [k for k,obj in class_dict.items() if isinstance(obj,table_property)]
        return tabulated_properties           

    @property
    def table_properties_description(self):
        class_dict = self.__class__.__dict__
        tabulated_properties = [obj.desc for k,obj in class_dict.items() if isinstance(obj,table_property)]
        return tabulated_properties       

    @classmethod
    def cls_all_property_labels(cls):
        return [obj.label for k,obj in cls.__dict__.items() if isinstance(obj,table_property)]

    @classmethod
    def cls_all_property_keys(cls):
        return [k for k,obj in cls.__dict__.items() if isinstance(obj,table_property)]                       
    
    @classmethod
    def cls_all_attrs_fields(cls):
        return attr.fields_dict(cls)

    #Clearance Methods
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

    #Multi-Component Table Combination Methods


    #Saving & Data Acces Methods
    def get_field_from_table(self,field,check_type=None,check_value:Callable = None):
        '''Converts Pandas To Numpy Array By Key, also handles poorly formated fields
        :param check_type: use a type or tuple of types to validate if the field is of type table
        :param check_value: use a function to check each value to ensure its valid for return, check type take priority'''
        if self.joined_dataframe is None:
            return numpy.array([])
        elif field in self.joined_dataframe:                                              
            table = self.joined_dataframe[field]
        elif field.title() in self.joined_dataframe:
            table = self.joined_dataframe[field.title()]
        else:
            raise Exception('No Field Named {}'.format(field))
        
        #Remove Infinity
        table = table.replace([numpy.inf, -numpy.inf], numpy.nan)

        if check_type is not None:    
            if all([isinstance(v,check_type) for v in table]):
                return table.to_numpy(copy=True)
            return None            
        elif check_value is not None:
            if all([check_value(v) for v in table]):
                return table.to_numpy(copy=True)
            return None                       

        return table.to_numpy(dtype=float,copy=True)

        



    def save_to_worksheet(self,worksheet:pygsheets.Worksheet):
        '''Saves to a gsheets via pygsheets adds static and regular data'''

        title = self.identity.replace('_',' ').title()

        self.info('saving worksheet as {}'.format(title))
        wksh = worksheet

        wksh.clear()

        #Static data
        start = pygsheets.Address((2,2))

        tld = self.toplevel_static
        sdf = tld['dfs']

        cur_index = start + (1,0)
        for i,df in enumerate(sdf):
            self.debug('saving dataframe {}'.format(df))
            wksh.update_value(start.label,self.identity)
            wksh.cell(start.label).set_text_format('bold',True)
            wksh.set_dataframe(df,cur_index.label , extend=True)
            cur_index += (2,0)

        cur_index += (3,0)

        var_index = pygsheets.Address(cur_index.label)

        max_row = 0

        vrt = self.variable_tables
        self.info('saving {} other static tables'.format(len(vrt)))

        for dfpack in vrt:
            conf = dfpack['conf']
            df = dfpack['df']
            self.debug('saving dataframe {}'.format(df))

            (num_rows,num_cols) = df.shape
            max_row = max(max_row,num_rows)
            
            wksh.update_value((var_index-(1,0)).label,conf.classname)
            wksh.cell((var_index-(1,0)).label).set_text_format('bold',True)    

            wksh.set_dataframe(df,start=var_index.label,extend=True)
            
            var_index += (0,num_cols)
            
        cur_index += (3+max_row,0) 
            
        ost = self.other_static_tables

        self.info('saving {} other static tables'.format(len(ost)))

        for dfpack in ost:
            conf = dfpack['conf']
            sdf = dfpack['dfs']

            wksh.update_value((cur_index-(1,0)).label,conf.identity)
            wksh.cell((cur_index-(1,0)).label).set_text_format('bold',True)            

            for i,df in enumerate(sdf):
                
                self.debug('saving {} dataframe {}'.format(conf.identity, df))
                wksh.set_dataframe(df,start=cur_index.label ,extend=True)
                cur_index += (2,0)
            
            cur_index += (3,0)   

    #Save functionality
    def save_table(self,dataframe=None,filename=None,meta_tags=None,*args,**kwargs):
        '''Header method to save the config in many different formats
        :param meta_tags: a dictionary with headers being column names, and the value as the item to fill that column'''
        if dataframe is None:
            dataframe = self.dataframe
        
        self.info('saving gsheets...')

        if meta_tags is not None and type(meta_tags) is dict:
            for tag,value in meta_tags.items():
                dataframe[tag] = value


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
            dataframe.to_csv(path_or_buf=filepath,index=False,*args,**kwargs)

    def save_excel(self,dataframe,filename=None,*args,**kwargs):
        if self.TABLE:
            if filename is None:
                filename = '{}.xlsx'.format(self.filename)
            if type(filename) is str and not filename.endswith('.xlsx'):
                filename += '.xlsx'
            filepath = os.path.join(self.config_path_daily,filename)
            dataframe.to_excel(path_or_buf=filepath,*args,**kwargs)

    def save_gsheets(self,dataframe,filename=None,index=False,*args,**kwargs):
        '''A function to save the table to google sheets
        :param filename: the filepath on google drive - be careful!
        '''
        with self.drive.context(filepath_root=self.local_sync_path, sync_root=self.cloud_sync_path) as gdrive:
            with gdrive.rate_limit_manager( self.save_gsheets,6,dataframe,filename=filename,*args,**kwargs) as tdrive:
                
                old_sleep = tdrive._sleep_time
                tdrive.reset_sleep_time( max(old_sleep,2.5) )
                
                gpath = tdrive.sync_path(self.local_sync_path)
                self.info(f'saving as gsheets in dir {self.local_sync_path} -> {gpath}')
                parent_id = gdrive.get_gpath_id(gpath)
                #TODO: delete old file if exists
                tdrive.sleep(2*(1+tdrive.time_fuzz*random.random()))
                if tdrive and tdrive.gsheets:
                    sht = tdrive.gsheets.create(filename,folder=parent_id)
                    
                    tdrive.sleep(2*(1+tdrive.time_fuzz*random.random()))
                    tdrive.cache_directory(parent_id)

                    wk = sht.sheet1

                    wk.rows = dataframe.shape[0]
                    gdrive.sleep(2*(1+tdrive.time_fuzz*random.random()))

                    wk.set_dataframe(dataframe,start='A1',fit=True)
                    gdrive.sleep(2*(1+tdrive.time_fuzz*random.random()))

                    #TODO: add in dataframe dict with schema sheename: {dataframe,**other_args}
                    self.info('gsheet saved -> {}'.format(os.path.join(gpath,filename)))

                tdrive.reset_sleep_time( old_sleep )

            
        
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
        

if __name__ == '__main__':

    import unittest
    import io
    import tempfile

    @otterize
    class TestConfig(TabulationMixin):

        attrs_prop = attr.ib(1.0)
        attrs_str = attr.ib('hey now')
        always_save_data = True

        @table_property
        def test_one(self):
            return 1
            
        @table_property
        def test_two(self):
            return numpy.random.rand(100)
            
        @table_property(label='three',desc='some words')
        def test_three(self):
            return 2
    
        @table_property(label='four',desc='would make sense')
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
            self.assertEqual( self.test_config.attr_labels, ['Name',  'Attrs Prop', 'Attrs Str'])
        
        def test_attrs_vals(self):
            self.assertEqual( self.test_config.attr_row, ['default', 1.0, 'hey now'])
        
        def test_property_labels(self):
            ans = set(('four', 'test_two','test_one', 'three'))
            self.assertEqual(set(self.test_config.table_properties_labels),ans)

        def test_table_vals(self):
            vals_ans = (1,2)
            for val in vals_ans:
                self.assertTrue(any([val==valcan if type(valcan) is int else False for valcan in self.test_config.table_properties]))
            
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


