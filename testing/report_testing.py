
from ottermatics.reporting import *
from ottermatics.configuration import otterize
from ottermatics.tabulation import table_property
from ottermatics.components import Component
from ottermatics.analysis import Analysis
from logging import *
#tests scripts

@otterize
class TestComponent(Component):
    blank = attr.ib(default=None, validator=None) #no validator

    name = attr.ib(default='test', validator=STR_VALIDATOR())
    word = attr.ib(default='bird', validator=STR_VALIDATOR())
    val = attr.ib(default=10, validator=NUMERIC_VALIDATOR())


    has_random_properties = True

    @table_property
    def synthetic_val(self):
        if self.word == 'schfiftyfive':
            return 55
        return random.random() * self.val

    @table_property
    def surpise_val(self):
        return 13

@otterize
class OtherComponent(Component):

    arg1 = attr.ib(default='mmmm', validator=STR_VALIDATOR())
    arg2 = attr.ib(default='word', validator=STR_VALIDATOR())
    argnum = attr.ib(default=10, validator=NUMERIC_VALIDATOR())
    ones = attr.ib(default=1111, validator=NUMERIC_VALIDATOR())
    wacky = attr.ib(default='one hundred and 111', validator=STR_VALIDATOR())
    more = attr.ib(default='other stuff', validator=STR_VALIDATOR())

    has_random_properties = True

    @table_property
    def random_val(self):
        return self.argnum * random.random() / random.random()

    @table_property
    def new_val(self):
        return self.argnum * random.random() 

@otterize
class SupriseComponent(Component):

    arg3 = attr.ib(default='mmmm', validator=STR_VALIDATOR())
    arg4 = attr.ib(default='word', validator=STR_VALIDATOR())
    argque = attr.ib(default=-999, validator=NUMERIC_VALIDATOR())
    #ones = attr.ib(default=1111, validator=NUMERIC_VALIDATOR())
    #wacky = attr.ib(default='one hundred and 111', validator=STR_VALIDATOR())
    #more = attr.ib(default='other stuff', validator=STR_VALIDATOR())

    has_random_properties = True

    @table_property
    def negative_random_val(self):
        return self.argque * random.random() / random.random()        


@otterize
class TestAnalysis(Analysis):

    surprise = attr.ib(factory=SupriseComponent)
    other_cocomponent = attr.ib(factory=OtherComponent)
    internal_component = attr.ib(factory=TestComponent)
    some_random_value = attr.ib(default=10,validator=NUMERIC_VALIDATOR())
    other_rand_val = attr.ib(default=1E6,validator=NUMERIC_VALIDATOR())
    
    mode = 'iterator'
    iterator = attr.ib(factory = lambda: list(range(10)))
    has_random_properties = True

    @table_property
    def internal_val(self):
        return self.internal_component.val

    @table_property
    def rand_val(self):
        return random.random() * self.internal_component.val

    def evaluate(self, item ):
        self.some_random_value = item

    @table_property
    def random_none(self):
        val = random.random()
        if val > 0.6:
            return None
        return val


if __name__ == '__main__':
    log = logging.getLogger()
    DEBUG = True
    if DEBUG:
        log.setLevel(logging.DEBUG)
        logging.basicConfig( level = logging.DEBUG)
        set_all_loggers_to( logging.DEBUG)

    analysis = TestAnalysis()
    tc = analysis.internal_component


    #1) Ensure exists Reflect The Database
    db = DBConnection('reports',host='localhost',user='postgres',passd='dumbpass')
    db.ensure_database_exists()
    #db.engine.echo = True

    rr = ResultsRegistry(db=db)
    # rr.ensure_all_tables()
    # rr.base.metadata.drop_all()
    # rr.initalize()

    #2) Use ComponentRegistry, and AnalysisRegistry to gather subclasses of each. Analysis will also have results tables since they are components, these will be referenced in the analysis table. 
    analysis.solve()
    rr.ensure_analysis(analysis)
    thread = rr.upload_analysis(analysis, use_thread=True)
    thread.join()

    #3) For each component and analysis map create tables or map them if they dont exist.
    # - use a registry approach (singleton?) to map the {component: db_class} pairs

    #4) When an an an analysis is solved, if configured for reporting, on post processing the results will be added to the database 