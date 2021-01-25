from contextlib import contextmanager
import attr

from .logging import LoggingMixin

import numpy
import functools


class inst_vectorize(numpy.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

def tabled_property(function):
    '''A decorator that wraps a function like a regular property
    The output will be tabulized for use in the ottermatics table methods
    values must be single valued (str, float, int) '''
    pass

def tabled_stats_property(function):
    '''A decorator that wraps a function like a regular property
    The output must be a numpy array or list of number which can be 
    tabulized into columns like average, min, max and std for use in the ottermatics table methods'''    
    pass


@attr.s
class ConfigurationAnalysis(Configuration):
    '''A type of configuration that will reach down among all attribues of a configuration,
    and interact with their solvers and stored data
    
    data will be stored a colmn row type arrangement in csv, database table (via sqlalchemy), and gsheets table,
    a saving strategy will be chosen using _____________
    
    configuration analysis's own table will be structure mostly around primary variables for this problem,
    they could be created by hand for your table, reaching into other configuraitons, or they could be the default.

    We will keep track of how deep we reach into the configuration to store tables, `store_level` starting at zero will only
    save the table of this configuration
    '''

    store_type = attr.ib(default=None, validator=attr.validators.in_(['csv','db','gsheets',None]))
    store_level = attr.ib(default=1, validator=attr.validators.instance_of(int))

    def go_through_configurations(self,parent_level):
        pass

    #Table Saving functionality
    def save_file(self):
        pass
    
    def remove_file(file):
        pass

    @property
    def rel_store_path(self):
        return 'analysis/{}'.format(datetime.date.today()).replace('-','_')

    @property
    def filename(self):
        return '{}_{}'.format(self.identity,self.name,self.date).replace(' ','_')



@attr.s
class Configuration(LoggingMixin):
    '''Configuration is a pattern for storing attributes that might change frequently,
    provides a way to `solve` a configuration, but this might be a simple update calculation.
    Attributes are stored with attr.ib or as python properties. 
    
    Special types of decorators are provided 
    as wrappers for regular properties however the will tabulate data, like `tabled_property` 
    that will store variables in a row of dictionaries referred to as `TABLE` that updates with all 
    attributes of the configuration when solve is called. Solve by default doesn't do anythign, it just
    adds a row to the table. data is stored with a increasing primary key index

    solve() is the wrapper for evaluate() which will be overriden, the variables will be saved in TABLE
    after evaluating them. 

    importantly since its an attr's class we don't use __init__, instead overide __attrs_post_init__ and
    use it the same way.
    '''
    _temp_vars = None

    name = attr.ib(default='default',validator= attr.validators.instance_of(str))
    index = attr.ib(default=0,validator= attr.validators.instance_of(int))

    log_fmt = "[%(identity)-24s]%(message)s"
    log_silo = True

    #yours to implement
    _skip_attr = None

    #Ultra private variables
    _table = None
    _cls_attrs = None
    _attr_labels = None    
    

    __attrs_post_init__(self,*args,**kwargs):
        self.reset_table()


    def evaluate(*args,**kwargs):
        '''evaluate is a fleixble method to be overriden. Oftentimes it might not be as configurations are 
        useful stores
        
        :return: isn't stored in table, but is returned through solve making it possibly useful to track error'''
        return None

    def solve(self,*args,**kwargs):
        #dont override this
        output = self.evaluate(*args,**kwargs)
        self.save_data()
        self.solve_configurations(*args,**kwargs)
        return output

    def solve_configurations(self,*args,**kwargs):
        for config in self.internal_configurations:
            self.config.solve(*args,**kwargs)

    def save_data(self):
        if self.anything_changed:
            self.table.append(self.data_row)

    def reset_table(self):
        '''Resets the table, and attrs label stores'''
        self.index = 0.0
        self._cls_attrs = None
        self._attr_labels = None
        return self._table = []

    @property
    def TABLE(self):
        '''this should seem significant'''
        return self._table

    @property
    def internal_configurations(self):
        '''go through all attributes determining which are configuration objects'''
        return {k:v for v in for k,v in self.store.items() if isinstance(v,Configuration)}

    @property
    def data_row(self):
        '''method that returns collects valid tabiable attributes immediately from this config
        will ignore any attributes labeled in _skip_attr which is a list
        best to not override these, but you can if you really want'''
        return self.attr_row

    @property
    def data_label(self):
        '''method that returns the lables for the tabilated attributes immediately from this config,
        will ignore any attributes labeled in _skip_attr which is a list
        best to not override these, but you can if you really want'''
        return self.attr_labels

    @contextmanager
    def difference(self,**kwargs):
        '''may want to consider using attr.evolve instead.... a context manager that will allow you to dynamically change any information,
        then will change it back in a fail safe way. 
        
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
            print('Could Not Change {}'.format( ','.join(list(bad_vars ) )))

        try: #Change Variables To Input
            for arg,var in kwargs.items():
                setattr(self,arg,var)
            yield self
        finally:
            for arg in kwargs.keys():
                var = _temp_vars[arg]
                setattr(self,arg,var)

    def add_fields(self, record):
        '''Overwrite this to modify logging fields, change log_fmt in your class to use the value set here.'''
        record.identity = self.identity

    @property
    def filename(self):
        '''A nice to have, good to override'''
        return self.name.replace(' ','_')

    @property
    def identity(self):
        '''A customizeable property that will be in the log by default'''
        return '{}-{}'.format(type(self).__name__,self.name).lower()

    #Data Tabulation - Intelligent Lookups
    @property
    def skip_attr(self) -> list: 
        return self._skip_attr

    @property
    def attr_labels(self) -> list:
        if self._attr_labels is None:
            self._attr_labels = list([k for k in attr.fields_dict(self.__class__).keys() if k not in self.skip_attr])
        return self._attr_labels

    @property
    def attr_row(self) -> list:
        return list([self.store[k] for k in self.attr_labels])

    @property
    def store(self):
        #lets pretend we're not playing with fire
        return self.__dict__