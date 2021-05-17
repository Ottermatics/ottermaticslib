
from ottermatics.reporting import *
from ottermatics.configuration import otterize
from ottermatics.tabulation import table_property
from ottermatics.components import Component
from ottermatics.analysis import Analysis

#tests scripts

@otterize
class TestComponent(Component):
    blank = attr.ib(default=None, validator=None) #no validator

    name = attr.ib(default='test', validator=STR_VALIDATOR())
    word = attr.ib(default='bird', validator=STR_VALIDATOR())
    val = attr.ib(default=10, validator=NUMERIC_VALIDATOR())
    #ones = attr.ib(default=1111, validator=NUMERIC_VALIDATOR())
    #wacky = attr.ib(default='one hundred and 111', validator=STR_VALIDATOR())
    #more = attr.ib(default='other stuff', validator=STR_VALIDATOR())

    has_random_properties = True

    @table_property
    def synthetic_val(self):
        if word == 'schfiftyfive':
            return 55
        return random.random() * self.val

@otterize
class TestAnalysis(Analysis):

    internal_component = attr.ib(factory=TestComponent)
    some_random_value = attr.ib(default='hey now',validator=STR_VALIDATOR())
    other_rand_val = attr.ib(default=1E6,validator=NUMERIC_VALIDATOR())
    
    has_random_properties = True

    @table_property
    def internal_val(self):
        return self.internal_component.val

    @table_property
    def rand_val(self):
        return random.random() * self.internal_component.val


#1) Ensure exists Reflect The Database
db = DBConnection('reports',host='localhost',user='postgres',passd='***REMOVED***')
db.ensure_database_exists()
#db.engine.echo = True

DataBase.metadata.bind = db.engine
DataBase.metadata.drop_all()
DataBase.metadata.reflect( db.engine, autoload=True, keep_existing = True )

#2) Use ComponentRegistry, and AnalysisRegistry to gather subclasses of each. Analysis will also have results tables since they are components, these will be referenced in the analysis table. 
AnalysisRegistry.__table__.create(db.engine, checkfirst=True)
ComponentRegistry.__table__.create(db.engine, checkfirst=True)

AnalysisRegistry.ensure_analysis_tables(db)
ComponentRegistry.ensure_component_tables(db)


#3) For each component and analysis map create tables or map them if they dont exist.
# - use a registry approach (singleton?) to map the {component: db_class} pairs






#4) When an an an analysis is solved, if configured for reporting, on post processing the results will be added to the database 