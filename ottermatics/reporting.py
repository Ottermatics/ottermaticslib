
import logging

from ottermatics.data import *
from ottermatics.tabulation import *
from ottermatics.configuration import *
from ottermatics.components import *
#from ottermatics.patterns import SingletonMeta

from ottermatics.analysis import *

import attr
import random

from sqlalchemy import Integer, ForeignKey, String, Column, Table,Boolean, MetaData, Unicode, UnicodeText
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.associationproxy import association_proxy

from collections import OrderedDict

DEFAULT_STRING_LENGTH = 256

'''A Module in which we reflect changing schema in components or other tabulationmixin instances
when solved by an an analysis. The goal is to preserve information from expensive analysis to be 
analized later in google data studio.

There are two major components to this module, that follow from the working plan:
1) Investigate the existing database and create the tables nessicary
2) upload the data to the database in the respective tables
3) If nessicary update or create a view for each component and analysis (hard part!)

At the time of creation attributes with ottermatics validators will be created as columns in the component table, where as only numeric key value pairs will be added to the components Vertical Attribute Mapping table (VAM)

'''

log = logging.getLogger('otter-reporting')


class MappedDictMixin:
    """Adds obj[key] access to a mapped class.

    This class basically proxies dictionary access to an attribute
    called ``_proxied``.  The class which inherits this class
    should have an attribute called ``_proxied`` which points to a dictionary.

    """

    def __len__(self):
        return len(self._proxied)

    def __iter__(self):
        return iter(self._proxied)

    def __getitem__(self, key):
        return self._proxied[key]

    def __contains__(self, key):
        return key in self._proxied

    def __setitem__(self, key, value):
        self._proxied[key] = value

    def __delitem__(self, key):
        del self._proxied[key]


class DBRegistry(DataBase):
    '''DBRegistry represents an abstract interface to register tables of a component type'''

    __abstract__ = True

    id = Column( Integer, primary_key = True)
    tablename = Column(String(DEFAULT_STRING_LENGTH))
    attrtable = Column(String(DEFAULT_STRING_LENGTH))
    classname = Column(String(DEFAULT_STRING_LENGTH))
    mro_inheritance = Column(String(1200))
    active = Column(Boolean(), default=True )

    _cls_tabble_mapping = None
    registry_component = None

    default_args = { 'keep_existing':True , 'autoload': False }
    attr_table_prefix = ''
    table_prefix = ''

    def __init__(self,mapped_class):
        self.classname == mapped_class.__name__
        
        name = mapped_class.__name__.lower()
        self.tablename = f'{self.table_prefix}{name}'
        self.attrtable = f'{self.attr_table_prefix}{name}'     

        distinct =mapped_class.mro()
        if mapped_class in distinct:
            distinct.remove(mapped_class)
        comp_inx = distinct.index(Component)

        self.mro_inheritance = '.'.join([cls.__name__ for cls in distinct[0:comp_inx+1]])


    @classmethod
    def mapping(cls):
        return cls._cls_tabble_mapping

    @classmethod
    def db_columns(cls):
        if '__table__' in cls.__dict__ and cls.__table__ is not None:
            return set(list(cls.__table__.c.keys()))

    @classmethod
    def all_fields(cls):
        all_table_fields = set([ attr.lower() for attr in cls._mapped_component.cls_all_property_labels() ])
        all_attr_fields = set([ attr.lower() for attr in cls._mapped_component.cls_all_attrs_fields().keys() ])
        all_fields = set.union(all_table_fields,all_attr_fields)
        return set(list(all_fields))

    @classmethod
    def dynamic_columns(cls):
        if cls._mapped_component is None:
            return []
        return set.difference(cls.all_fields(), cls.db_columns())

    @classmethod
    def filter_by_validators(cls,attr_obj):
        if not attr_obj.validator:
            return False
        valtype = type(attr_obj.validator.type)
        if not valtype:
            return False
        if valtype is tuple and any([gtype in attr_obj.validator.type for gtype in (str,float,int)]):
            return True
        if any([ gtype is attr_obj.validator.type for gtype in (str,float,int) ]):
            return True        
        return False

    @classmethod
    def validator_to_column(cls,attr_obj):
        if type(attr_obj.validator.type) is tuple and any([ gtype in attr_obj.validator.type for gtype in (float,int)]):
            return Column(Numeric, default=attr_obj.default, nullable=True)
        if type(attr_obj.validator.type) is tuple and str in attr_obj.validator.type:
            return Column(String(DEFAULT_STRING_LENGTH),default=attr_obj.default, nullable=True)        
        if str is attr_obj.validator.type:
            return Column(String(DEFAULT_STRING_LENGTH),default=attr_obj.default, nullable=True)
        return Column(Numeric,default=attr_obj.default, nullable=True)   


    @classmethod
    def dict_attr(this_class,comp_cls, target_table):
        log.debug(f'dict_attr: {this_class}|{comp_cls} {target_table}')
        name = comp_cls.__name__.lower()
        dict_attr = {'__tablename__': f'{this_class.attr_table_prefix}{name}' ,
                        'id': Column(Integer, primary_key=True),
                        'component_id': Column(Integer,ForeignKey( f'{target_table}.id')),
                        'key': Column(Unicode(64), primary_key=True), 
                        'value': Column(Numeric),
                        '__table_args__': this_class.default_args,
                        '__abstract__': False
                        }
        return dict_attr

    @classmethod
    def comp_attr(this_class,comp_cls,analysis_tablename=None):
        log.debug(f'comp_attr: {this_class}|{comp_cls} {analysis_tablename}')
        name = comp_cls.__name__.lower()
        tblname = f'{this_class.table_prefix}{name}'

        comp_attr = {'__tablename__': tblname ,
                        'id' :  Column(Integer, primary_key=True),
                        '__table_args__': this_class.default_args,
                        '__abstract__': False,
                        '_mapped_component': comp_cls
                        }

        if isinstance(analysis_tablename,str): #its a compnent
            log.debug(f'comp_attr - adding FK {this_class}|{comp_cls} {analysis_tablename}.id')
            comp_attr['result_id'] = Column(Integer, ForeignKey(f'{analysis_tablename}.id'))

        return comp_attr

    @classmethod
    def component_attrs(this_class,comp_cls):
        # #TODO: assign key values as attribute dict
        # #TODO: check if class has a __tablename__ and if it does use that with, ensure component_table_ on front
        log.debug(f'component_attrs: {this_class}|{comp_cls}')

        component_attrs = { key.lower(): this_class.validator_to_column(field) for key,field in attr.fields_dict(comp_cls).items() if this_class.filter_by_validators(field) and not key.lower() == 'name'}

        component_attrs.update( this_class.comp_attr(comp_cls) )
        return component_attrs

    @classmethod
    def db_component_dict(this_class,comp_cls,dict_type = False):
        '''returns the nessicary table types to make for reflection of input component class
        
        this method represents the dynamic creation of SQLA typing and attributes via __dict__'''

        log.debug(f'db_component_dict: {this_class}|{comp_cls} dict: {dict_type}')
        assert isinstance(comp_cls, type)
        assert Component in comp_cls.mro()

        comp_attr = this_class.component_attrs(comp_cls)
        if not dict_type:
            return {comp_attr['__tablename__']: type(f'DB_{comp_cls.__name__}',(this_class,), comp_attr) }

        else:
            dict_attr = this_class.dict_attr(comp_cls, f"{comp_attr['__tablename__']}.id" )
            dict_name = f'DB_{comp_cls.__name__}_Dict'
            dict_db = type(dict_name,(DataBase,), dict_attr)

            dict_col = attribute_mapped_collection("key")
            comp_attr['dict_values']  = relationship( dict_db.__name__, collection_class = dict_col )
            comp_attr['_proxied']  = association_proxy( "dict_values", "value",
                                                        creator=lambda key, value: dict_db(key=key, value=value))
            comp_name = f'DB_{comp_cls.__name__}'
            comp_db = type(comp_name,(MappedDictMixin,DataBase), comp_attr)

            globals()[comp_name] = comp_db
            globals()[dict_name] = dict_db

            out = OrderedDict()
            out[comp_attr['__tablename__']] = comp_db
            out[dict_attr['__tablename__']] = dict_db

            return out
                    

    @classmethod
    def registered_component_classes(cls):
        if cls.registry_component is not None and issubclass(cls.registry_component, Component):
            return cls.registry_component.component_subclasses()

        elif not type(cls.registry_component) is type:
            log.warning(f'Registered Component Class {cls.registry_component} not a class')

        elif not issubclass(cls.registry_component, Component):
            log.warning(f'Registered Component Class {cls.registry_component} not a component')

        else:
            log.warning(f'Invalid Registered Component Class {cls.registry_component}, ensure its a Component class')

    @classmethod
    def ensure_analysis_tables(this_cls,db):
        log.debug(f'ensure_analysis_tables: {this_cls}|{db}')
        for cls_name, cls in this_cls.registered_component_classes().items():
            to_create = []
            for table_name, compdb in this_cls.db_component_dict( cls, dict_type=True ).items():
                if not db.engine.has_table(compdb.__tablename__):
                    log.info(f'creating tables {table_name}')
                    to_create.append(compdb.__table__)
                    compdb.__table__.create(db.engine)
                    
                else:
                    log.info(f'table exists already {table_name}')         

                this_cls._cls_tabble_mapping[cls] = compdb.__table__
            
            #this_cls.metadata.create_all(db.engine,tables = to_create)
            with db.session_scope() as sesh:
                for tbl in to_create:
                    #tbl = this_cls(cls)
                    sesh.add(tbl)

class AnalysisRegistry( DBRegistry  ):

    __tablename__ = 'analysis_registry'
    table_prefix = 'analysis_table_'
    attr_table_prefix = 'analysis_attr_'
    registry_component = Analysis
    _cls_tabble_mapping = {} #this will be class based

class ComponentRegistry( DBRegistry ):

    __tablename__ = 'component_registry'
    table_prefix = 'component_table_'
    attr_table_prefix = 'component_attr_'
    registry_component = Component
    _cls_tabble_mapping = {}#this will be class based

@otterize
class ResultsRegistry( Configuration ):

    analysis_registry = AnalysisRegistry
    component_registry = ComponentRegistry

    components = None
    analyses = None

    initalized = False
    
    #inputs
    db = attr.ib( ) #required input is a database connector

    def initalize(self):
        #1) Ensure & Load Config & Analysis Tables

        #2) Use ComponentRegistry, and AnalysisRegistry to gather subclasses of each. Analysis will also have results tables since they are components, these will be referenced in the analysis table. 
        self.analysis_registry.__table__.create(self.db.engine, checkfirst=True)
        self.component_registry.__table__.create(self.db.engine, checkfirst=True)

        #self.analysis_registry.ensure_analysis_tables(self.db)
        #self.component_registry.ensure_component_tables(self.db)

        #3) For each component and analysis map create tables or map them if they dont exist.
        # - use a registry approach (singleton?) to map the {component: db_class} pairs

        #4) When an an an analysis is solved, if configured for reporting, on post processing the results will be added to the database 

    def register_analysis(self,analysis):
        if self.initalized:
            if analysis._solved and isinstance(analysis, Analysis):
                pass
                #1) Identify & Ensure The Analysis Table
                #2) Ensure Component Tables For The Analysis
                #3) For each result, add it to the DB

            if not analysis._solved:
                self.warning(f"Warning analysis {analysis} not solved")

            if not isinstance(analysis, Analysis):
                self.warning(f"Warning analysis {analysis} is not type Analysis {type(analysis)}")
                
        else:
                self.warning(f"Results Registry Not Initalized")