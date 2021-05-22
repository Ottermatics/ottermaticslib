
import logging

from ottermatics.data import *
from ottermatics.tabulation import *
from ottermatics.configuration import *
from ottermatics.components import *
#from ottermatics.patterns import SingletonMeta

from ottermatics.analysis import *

import attr
import random

from sqlalchemy import  ForeignKey,  Column, Table, MetaData, Unicode, UnicodeText
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.sql.sqltypes import BOOLEAN, NUMERIC, VARCHAR, INTEGER, INT, Integer,String,Boolean,Numeric

from sqlalchemy.sql import func

from threading import Thread

from sqlalchemy.sql.expression import Insert
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.compiler import compiles

#TODO: add schema info
ReportBase = declarative_base( )

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

log  = logging.getLogger('otterlib-report')


#ignore duplicates
@compiles(Insert, 'postgresql')
def ignore_duplicates(insert, compiler, **kw):
    s = compiler.visit_insert(insert, **kw)
    ignore = insert.kwargs.get('postgresql_ignore_duplicates', False)
    return s if not ignore else s + ' ON CONFLICT DO NOTHING'

Insert.argument_for('postgresql', 'ignore_duplicates', None)

#TODO: make new sqalchemy base, dont mixin other bases into our special fancy db

#These Mixins & Base Table Classes Help Us By Defining Common Bases For tables

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

#TODO: Bases for AttrTable, CompTable, AnalysisTable

class AttrBase(ReportBase):
    __abstract__ = True
    #id = Column(Integer(), primary_key=True)
    key = Column(Unicode(64), nullable=False, primary_key=True)
    value = Column(Numeric)

    def __init__(self,result_id_in,key=None,value=None):
        #log.debug(f'creating attr rid:{result_id_in} {self}, {key}={value}')
        
        self.result_id = result_id_in

        if key is not None and value is not None:
            if isinstance(value,(float,int)):
                self.key = key
                self.value = float(value)


    @property
    def unique_keys(self):
        rr = ResultsRegistry()
        with rr.db.session_scope() as sesh:
            results = list(set( sesh.query(self.__class__.key).all()))
        return results          

    @classmethod
    def items(cls):
        rr = ResultsRegistry()
        with rr.db.session_scope() as sesh:
            results = list( sesh.query(cls).all())
        return results  


class TableBase( MappedDictMixin, ReportBase ):
    __abstract__ = True

    check_types = {
                    String: (str,),
                    Integer: (int,),
                    Numeric: (float,int),
                    Boolean: (bool,)               
                  }

    store_type = {
                    String:  str,
                    Integer: int,
                    Numeric: float,
                    Boolean: bool
                  }                  

    ignore_keys = ('id','index')

    attr_class = None

    attr_store = None

    def __init__(self,result_id_in,**kwargs):
        
        self.attr_store = {}
        if 'result_id_in' in self.__dict__:
            self.result_id_in = result_id_in

        log.debug(f'creating TB id:{result_id_in} obj:{self}')
        table = self.__class__.__table__
        cols = self.dbi_columns
        
        
        for key, val in kwargs.items():
            
            key = key.lower()
            vtype = type(val)            
            
            if key not in self.ignore_keys: #we will try to store the data
                
                if key in cols: #we will map the data to the column
                    
                    col = table.columns[key]                
                    ctype = type(col.type)

                    #log.debug(f'{self} setting {key}|{col}|{ctype}|{vtype}')

                    in_stypes = any([issubclass(ctype,check_type) for check_type in self.check_types])

                    if in_stypes:
                        
                        use_ctype = ctype
                        if not ctype in self.check_types:
                            valids = list(filter( lambda ct: issubclass(ctype,ct), list(self.check_types.keys()) ))

                            if valids:
                                if len(valids) > 1:
                                    self.warning('more than one type matched for!!! {ctype}->{valids} ')          
                                use_ctype = valids[0]
                            #log.debug(f'getting registered type {use_ctype} for {ctype}')

                        in_type = isinstance(val, self.check_types[ use_ctype ])

                    else:
                        in_type = False

                    if in_stypes and in_type:

                        setattr(self, key, self.store_type[use_ctype](val) )
                    else:
                        log.debug(f'{key}={val} {vtype}=>{use_ctype}? col-valid:{in_stypes} valid-type:{in_type} col:{col}')
                        
                elif isinstance(val,(float,int)): #dynamic mapping
                    if not numpy.isnan(val):
                        atobj = self.attr_class( result_id_in, key=key, value=val)
                        self.attr_store[key] = atobj
                        #self[key] = atobj #dont use orm for upload!

                #else:
                #    log.debug(f'{key} not found for columns {cols}')           

    #these are breaking the analysis table creation since we can't make the attr_class before initiation when @declared_attr is made
    # @declared_attr
    # def dict_values(cls):
    #     if cls.attr_class:
    #         return relationship( cls.attr_class.__name__, collection_class = attribute_mapped_collection("key") )

    # @declared_attr
    # def _proxied(cls):
    #     return association_proxy( "dict_values", "value")       

    @property
    def dbi_columns(self):
        return list(self.__class__.__table__.columns.keys())

    @classmethod
    def db_columns(cls):
        if '__table__' in cls.__dict__ and cls.__table__ is not None:
            return set(list(cls.__table__.c.keys()))

    @classmethod
    def all_fields(cls):
        if cls._mapped_component is None:
            return []        
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
    def items(cls):
        rr = ResultsRegistry()
        with rr.db.session_scope() as sesh:
            results = list( sesh.query(cls).all())
        return results













class ReportingMixin:
    '''Add to an analysis component'''

    _report_db = None

    @property
    def report_db(self):
        if self._report_db is not None:
            return self._report_db
        else:
            self.warning('no report db connection set!')

        #else: #this is alot, lets be more careful
        #    self.warning('autogenerating database...')
        #    report_db = DBConnection(database_name='reports')
        #    report_db.ensure_database_exists()

    @report_db.setter
    def report_db(self,inputval:DBConnection):
        if isinstance(inputval, DBConnection):
            self._report_db = inputval
        else:
            self.warning(f'Got bad value {inputval} of type {type(inputval)}')

    def report_data(self, use_thread = False):

        if self.report_db and self.solved:
            try:
                assert isinstance( self, Analysis )
                rr = ResultsRegistry( self.report_db )
                rr.ensure_analysis( self )
                return rr.upload_analysis( self, use_thread = use_thread )

            except Exception as e:
                self.error(e, 'Issue Reporting Data')

        elif not self.solved:
            self.warning('Analysis Not Solved, Cannot Upload')

        elif not self.report_db:
            self.warning('No Report Database Initated')    




@otterize
class ResultsRegistry(Configuration,metaclass = SingletonMeta):
    '''An instance to manage the various database mappings to this live code repository'''
    
    default_args = {'keep_existing':True, 'autoload': False}
    
    db = attr.ib()
    name = attr.ib(default='')
    base = attr.ib(default=ReportBase) #,factory=declarative_base) #slot for sqlalchemy base

    _component_cls_table_mapping = None
    
    schema = 'report'
    
    analysis_table_abrv = ''
    analysis_attr_abrv = 'anly_attr_'
    component_table_abrv = 'comp_tbl_'
    component_attr_abrv = 'comp_attr_'    

    def __on_init__(self):
        self.info('creating registry and ensuring db exists!')
        self.db.ensure_database_exists(create_meta = False)

        self.base.metadata.bind = self.db.engine
        self.base.metadata.reflect( self.db.engine, autoload=True, keep_existing = True )

        self._component_cls_table_mapping = {}
        self.initalize()

    def initalize(self):
        AnalysisRegistry.__table__.create(self.db.engine, checkfirst=True)
        ComponentRegistry.__table__.create(self.db.engine, checkfirst=True)       

    @property
    def db_classes(self):
        return list(set([val[0] for val in self.mapped_tables.values()]))

    @property
    def component_classes(self):
        return list(set([val[1] for val in self.mapped_tables.values()]))

    @property
    def mapped_classes(self):
        '''schema class : db class'''
        return { val[1] : val[0] for val in self.mapped_tables.values() }

    @property
    def mapped_tables(self):
        return self._component_cls_table_mapping

    @property
    def analyses(self):
        return Analysis.component_subclasses()

    @property
    def components(self):
        comps = Component.component_subclasses()
        comp_subclasses = set(comps)
        analy_subclasses = set(Analysis.component_subclasses())
        comp_keys = set.difference(comp_subclasses,analy_subclasses)
        
        return {ckey:comps[ckey] for ckey in comp_keys}        

    def filter_by_validators(self,attr_obj):
        if not attr_obj.validator:
            return False
        
        if not isinstance(attr_obj.validator,TAB_VALIDATOR_TYPE):
            return False

        valtype = type(attr_obj.validator.type)
        if not valtype:
            return False
        if valtype is tuple and any([gtype in attr_obj.validator.type for gtype in (str,float,int)]):
            return True
        if any([ gtype is attr_obj.validator.type for gtype in (str,float,int) ]):
            return True        
        return False

    def validator_to_column(self,attr_obj):
        if type(attr_obj.validator.type) is tuple and any([ gtype in attr_obj.validator.type for gtype in (float,int)]):
            return Column(Numeric, default=attr_obj.default, nullable=True)
        if type(attr_obj.validator.type) is tuple and str in attr_obj.validator.type:
            return Column(String(DEFAULT_STRING_LENGTH),default=attr_obj.default, nullable=True)        
        if str is attr_obj.validator.type:
            return Column(String(DEFAULT_STRING_LENGTH),default=attr_obj.default, nullable=True)
        return Column(Numeric,default=attr_obj.default, nullable=True)   


    def dict_attr(self,comp_cls, target_table_id, results_table = None, is_analysis=False):
        '''we make the default entries for a components attribues table'''

        #this modulelates names for componets / analysis tables
        table_abrv = self.component_attr_abrv
        if issubclass(comp_cls, Analysis): table_abrv = self.analysis_attr_abrv
        name = comp_cls.__name__.lower()

        default_attr = {'__tablename__': f'{table_abrv}{name}' ,
                        #'comp_id': Column(Integer,ForeignKey( target_table_id )),
                        '__table_args__': self.default_args,
                        '__abstract__': False
                        }
        
        if results_table is not None and not is_analysis: #analysis won't be mapped
            default_attr['__tablename__'] = f'{results_table}_{table_abrv}{name}'
            root_obj,_ = self.mapped_tables[results_table]#analysis won't be mapped
            backref_name = f'db_comp_attr_{name}'
            default_attr['result_id'] = Column(Integer, ForeignKey(f'{results_table}.id'),primary_key=True)
            default_attr['analysis'] = relationship(root_obj.__name__,backref = backref(backref_name,lazy='noload'))
        
        elif issubclass(comp_cls, Analysis) and is_analysis:
            default_attr['__tablename__'] = f'{table_abrv}{name}'
            backref_name = f'db_comp_attr_{name}'
            default_attr['result_id'] = Column(Integer, ForeignKey(target_table_id),primary_key=True)
            #default_attr['analysis'] = relationship(comp_cls.__name__,backref = backref(backref_name,lazy='noload'))
                    
        return default_attr

    def default_attr(self,comp_cls, results_table = None):
        '''we make the default entries for a components primary table'''

        table_abrv = self.component_table_abrv
        if issubclass(comp_cls, Analysis): table_abrv = self.analysis_table_abrv

        name = comp_cls.__name__.lower()
        #TODO: Add mapping __init__ method
        default_attr = {'__tablename__': f'{table_abrv}{name}' ,
                        #'id': Column(Integer, primary_key=True),
                        '__table_args__': self.default_args,
                        '__abstract__': False,
                        '_mapped_component': comp_cls,
                        'attr_store': None
                        }

        if results_table is not None:
            tbl_name = f'{results_table}_{table_abrv}{name}'
            backref_name = f'db_comp_{name}'
            root_obj,_ = self.mapped_tables[results_table]
            default_attr['__tablename__'] = f'{results_table}_{table_abrv}{name}' 
            default_attr['result_id'] = Column(Integer, ForeignKey(f'{results_table}.id') ,primary_key=True)
            default_attr['analysis'] = relationship(root_obj.__name__,backref = backref(backref_name,lazy='noload'))

        else: #its an analysis
            default_attr['created'] = Column(DateTime, server_default=func.now())
            default_attr['active'] = Column(Boolean(), server_default = 't')
            default_attr['run_id'] = Column(String(36)) #corresponds to uuid set in analysis
            default_attr['id']: Column(Integer, primary_key=True)
            #default_attr['components'] = relationship(TableBase, back_populates= "analysis")

        return default_attr

    #These use keys of table_property or attr and lower() so no spaces or capitals
    def all_possible_component_fields(self,cls):
        all_table_fields = set([ attr.lower() for attr in cls.cls_all_property_keys() ])
        all_attr_fields = set([ attr.lower() for attr in cls.cls_all_attrs_fields().keys() ])
        all_fields = set.union(all_table_fields,all_attr_fields)
        return set(list(all_fields))

    def component_attr(self,comp_cls, results_table = None):
        # #TODO: assign key values as attribute dict
        # #TODO: check if class has a __tablename__ and if it does use that with, ensure component_table_ on front

        component_attr = { key.lower(): self.validator_to_column(field) for key,field in comp_cls.cls_all_attrs_fields().items() if self.filter_by_validators(field)}

        component_attr.update({ k.lower(): Column(k.lower(), Numeric(),nullable=True) \
                                for k in comp_cls.cls_all_property_keys() })

        component_attr.update(self.default_attr(comp_cls,results_table=results_table))
        return component_attr

    def db_component_dict(self,comp_cls,results_table=None,is_analysis=False):
        '''returns the nessicary table types to make for reflection of input component class
        
        this method represents the dynamic creation of SQLA typing and attributes via __dict__'''

        assert isinstance(comp_cls, type)
        assert Component in comp_cls.mro()

        comp_attr = self.component_attr(comp_cls,results_table=results_table)
        tablename = comp_attr['__tablename__']

        if results_table:
            comp_cls_name = f'DB_{results_table}_{comp_cls.__name__}'

        else:
            comp_cls_name = f'DB_{comp_cls.__name__}'

        if is_analysis: results_table = comp_attr['__tablename__'] #pass reference for id!

        if is_analysis: #We create the analysis first
            if 'id' not in comp_attr:
                comp_attr['id']= Column(Integer, primary_key=True)

            comp_db_tup = (comp_cls_name,(TableBase,), comp_attr)
            comp_db = self.check_or_create_db_type(comp_attr['__tablename__'], comp_db_tup)

            dict_attr = self.dict_attr(comp_cls, f"{comp_attr['__tablename__']}.id" , results_table=results_table,is_analysis=is_analysis)
            dict_name = f'{comp_cls_name}_Dict'
            dict_db_tup = (dict_name,(AttrBase,), dict_attr)
            dict_db = self.check_or_create_db_type(dict_attr['__tablename__'], dict_db_tup)
            
            comp_db.attr_class = dict_db #This delayed monkey patch is fine since we wont be able to try any init's with this class until 

        else:
            dict_attr = self.dict_attr(comp_cls, f"{comp_attr['__tablename__']}.id" , results_table=results_table)
            dict_name = f'{comp_cls_name}_Dict'
            dict_db_tup = (dict_name,(AttrBase,), dict_attr)
            dict_db = self.check_or_create_db_type(dict_attr['__tablename__'], dict_db_tup)
            
            comp_attr['attr_class'] = dict_db
            comp_db_tup = (comp_cls_name,(TableBase,), comp_attr)
            comp_db = self.check_or_create_db_type(comp_attr['__tablename__'], comp_db_tup)

        return {comp_attr['__tablename__']:comp_db, dict_attr['__tablename__']: dict_db}

    def check_or_create_db_type(self,tablename,type_tuple):
        cls_name,ttype,attr_dict = type_tuple
        
        if self.db.engine.has_table(tablename):                
            
            remove = set(['__table_args__'])
            for key,item in attr_dict.items():
                if issubclass( type(item), Column ):
                    remove.add(key)
            
            for rkey in remove:
                if rkey in attr_dict:
                    #self.debug(f'removing {rkey}')
                    attr_dict.pop(rkey)
                else:
                    self.debug(f'couldnt find {rkey}')

            attr_dict['__table__'] = self.base.metadata.tables[tablename]
            attr_dict['__table_args__'] = {'autoload':True}

            self.debug(f'making type: {cls_name,self.base,attr_dict}')
            cls_db = type(cls_name,ttype,attr_dict)
            
            self.debug(f'mapped existing table to type {tablename} -> {cls_db.__name__}')
            setattr(self,cls_db.__name__,cls_db)
            
            return cls_db    

        else:
            self.info(f'creating class {cls_name} with properties {attr_dict}')
            cls_db = type(*type_tuple)
            self.debug(f'created table type {cls_db.__name__} -> {cls_db.__tablename__}')
            

            setattr(self,cls_db.__name__,cls_db)
            return cls_db    

    def map_component(self,dbcomponent,component):
        self.debug(f'mapping {dbcomponent},{component} -> {dbcomponent.__tablename__}')
        #add to internal mapping
        self._component_cls_table_mapping[dbcomponent.__tablename__] = (dbcomponent,component)
        
        tablenames = self.db.engine.table_names()
        atables = AnalysisRegistry.tablenames(self.db)
        ctables = ComponentRegistry.tablenames(self.db)

        self.info(f'comparing {tablenames} | {atables} | {ctables}')
        #create record of components and analyses
        if isinstance(component,Analysis) or issubclass(component,Analysis):
            if isinstance(component,Analysis): component = component.__class__
            if dbcomponent.__tablename__ not in atables:
                self.info(f'adding analysis record { dbcomponent.__tablename__}')
                with self.db.session_scope() as sesh:
                    rec = AnalysisRegistry(dbcomponent,component)
                    sesh.add( rec )

        elif isinstance(component,Component) or issubclass(component,Component):
            if isinstance(component,Component): component = component.__class__
            if dbcomponent.__tablename__ not in ctables:
                self.info(f'adding comonent record { dbcomponent.__tablename__}')
                with self.db.session_scope() as sesh:
                    rec = ComponentRegistry(dbcomponent,component)
                    sesh.add( rec )  

    def ensure_all_tables(self):
        '''creates a set of component tables linked to each analysis'''
        
        for cls_name, cls in  self.analyses.items():
            self.debug(f'ensure-all-tables: {cls}')
            self.ensure_analysis(cls)

    def ensure_component_tables(self,results_table=None):
        for cls_name, cls in self.components.items():
            self.debug(f'ensure-comp-tables: {cls}')
            self.ensure_component_table(cls, results_table=results_table)

    def ensure_component_tables_in_analysis(self,analysis,results_table=None):
        assert isinstance(analysis,Analysis)
        for cls in analysis.unique_internal_components_classes:
            self.debug(f'ensure-analyinst-tables: {cls}')
            self.ensure_component_table(cls, results_table=results_table)                



    def ensure_component_table(self, component_cls, results_table=None):
        self.debug(f'ensure-comp-table: {component_cls}')

        is_analysis = False
        if issubclass(component_cls, Analysis): is_analysis = True
        
        self.info(f'ensuring component {component_cls} analysis?:{is_analysis}')
        db_component_type_dict = self.db_component_dict( component_cls, results_table=results_table ,is_analysis=is_analysis )
        for comp_tbl, compdb in db_component_type_dict.items():

            if not  self.db.engine.has_table(comp_tbl ):
                self.info(f'creating tables {comp_tbl}')
                compdb.__table__.create(self.db.engine, checkfirst=True)

            else:
                self.debug(f'table exists already {comp_tbl}')

        for comp_tbl, compdb in db_component_type_dict.items():
            if not compdb.__name__.endswith('Dict'): self.map_component(compdb, component_cls ) 

        return db_component_type_dict

    def ensure_analysis(self,analysis):
        '''pass a class or instance to create the tables, passing an instance also ensures its internal components are
        tabulated'''
        self.info(f'ensure-analysis: {analysis}')
        analysis_cls = self.get_analysis_class( analysis )

        db_component_type_dict = self.ensure_component_table(analysis_cls)    

        #return primary table name
        out = [dbckey for dbckey in db_component_type_dict.keys() if dbckey.startswith(self.analysis_table_abrv)]
        if out:
            analysis_table = out[0]
            if analysis is analysis_cls:
                self.ensure_component_tables(results_table=analysis_table)
            else: #this is for the instance which has runtime components that we'll check
                self.ensure_component_tables_in_analysis(analysis,results_table=analysis_table)
        else:
            self.warning('no analysis table found')

    def get_analysis_class(self,analysis):
        if isinstance(analysis, Analysis):
            analysis_cls = analysis.__class__
        elif issubclass(analysis, Analysis):
            analysis_cls = analysis
        else:
            self.warning('ensure-analysis: analysis not mapped, using direct input')
            analysis_cls = analysis
        return analysis_cls                 

    def upload_analysis(self,analysis, use_thread = False):
        '''a wrapper for upload analysis with a thread selection context,
        :returns: a thread reference, which you can join after setting other work.'''

        if use_thread:
            thread = Thread(target = self._upload_analysis, args=(analysis,) )
            thread.start()
            return thread

        return self._upload_analysis(analysis)

    def _upload_analysis(self,analysis):
        '''the general method here we're using is to precompute the analysis row id, then batch the inserts after
        forcing the correct primary key'''
        self.info(f'upload analysis: {analysis}')

        class AvoidDuplicateAbortUpload(Exception): pass

        try:        
            def gen(table):
                for item in table:
                    yield item
            
            #analysis must be solved!
            assert isinstance(analysis,Analysis)
            assert analysis.solved
            
            acmpcls = analysis.__class__
            if acmpcls not in self.mapped_classes:
                self.ensure_analysis(analysis)

            adbcls = self.mapped_classes[acmpcls]

            rds = analysis.recursive_data_structure()
            data_gens = {}
            
            main_gen = gen(analysis.TABLE)

            for lvl, comps in rds.items():
                if lvl > 0:
                    for comp in comps:
                        cmp = comp['conf']
                        data_gens[self.mapped_classes[cmp.__class__]] = gen(cmp.TABLE)

            #with self.db.scoped_session() as sesh:
            any_succeeded = True
            last_values = {}
            inx = 0
            main_result = None

            while any_succeeded and inx < analysis.index:
                any_succeeded = False #guilty until proven innocent!!

                try:
                    data_dict = next(main_gen)
                    last_values[adbcls] = data_dict
                    main_result = adbcls(None,**data_dict)

                except StopIteration:
                    data_dict = last_values[adbcls]
                    main_result = adbcls(None,**data_dict)  

                else:
                    any_succeeded = True #here's ur fricken evidence ur honor

                with self.db.session_scope() as sesh:
                    sesh.add(main_result)
                    
                    sesh.commit()
                    result_id = int(main_result.id)

                    main_attrs = list(main_result.attr_store.values())
                    for item in main_attrs:
                        item.result_id = result_id
                    
                
                others = []
                for dbclass, generator in data_gens.items():
                    try:
                        data_dict = next(generator)
                        last_values[dbclass] = data_dict
                        data_dict['result_id'] = result_id
                        cmp_result = dbclass(result_id,**data_dict)
                        others.append(cmp_result)
                    
                    except StopIteration:
                        data_dict = last_values[dbclass] 
                        
                        data_dict['result_id'] = result_id
                        cmp_result = dbclass(result_id,**data_dict)
                        others.append(cmp_result)

                    else:
                        any_succeeded = True #here's ur fricken evidence ur honor
                
                #the scoped session rolls back anything created this time :)
                if not any_succeeded: 
                    raise AvoidDuplicateAbortUpload() 
                
                with self.db.session_scope() as sesh:

                    add_list = main_attrs + others + flatten([list(item.attr_store.values()) for item in others ])

                    sesh.add_all(add_list)

                
                inx += 1 #index += 1 is done at end of analysis so we should model that

        except AvoidDuplicateAbortUpload:
            pass #this is fine

        except Exception as e:
            self.error(e,'Issue Uploading Data')



#These items are registry related for the dynamic mapping

class MappedItem(ReportBase):
    __abstract__ = True

    _mapped_component = None

    id = Column( Integer, primary_key = True)
    tablename = Column(String(DEFAULT_STRING_LENGTH))
    attrtable = Column(String(DEFAULT_STRING_LENGTH))
    classname = Column(String(DEFAULT_STRING_LENGTH))
    mro_inheritance = Column(String(1200))

    active = Column(Boolean(), default=True )

    def __init__(self,tablerep, mapped_class):
        self.classname = mapped_class.__name__
        self.tablename = tablerep.__tablename__

        if issubclass(mapped_class,Analysis):
            self.attrtable = f'anly_attr_{tablerep.__tablename__}'
        else:
            self.attrtable = tablerep.__tablename__.replace('tbl','attr')
        
        self._mapped_component = mapped_class

        distinct =mapped_class.mro()
        if mapped_class in distinct:
            distinct.remove(mapped_class)
        comp_inx = distinct.index(Component)
        self.mro_inheritance = '.'.join([cls.__name__ for cls in distinct[0:comp_inx+1]])    

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
    def tablenames(cls,db):
        #rr = ResultsRegistry()
        with db.session_scope() as sesh:
            results = list( sesh.query(cls.tablename).all())
        return flatten(results)

class AnalysisRegistry( MappedItem ):

    __tablename__ = 'analysis_registry'


class ComponentRegistry( MappedItem ):

    __tablename__ = 'component_registry'




