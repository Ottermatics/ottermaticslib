from ottermatics.datastores.reporting import *
from ottermatics.configuration import otterize
from ottermatics.tabulation import system_property
from ottermatics.components import Component
from ottermatics.analysis import Analysis
from logging import *

import numpy

# tests scripts

# suprise the db and add more columns to test dynamic creation
suprise = False


@otterize
class TestComponent(Component):
    blank = attr.ib(default=None, validator=None)  # no validator

    name = attr.ib(default="test", validator=STR_VALIDATOR())
    word = attr.ib(default="bird", validator=STR_VALIDATOR())
    val = attr.ib(default=10)

    always_save_data = True

    @system_property
    def synthetic_val(self):
        if self.word == "schfiftyfive":
            return 55
        return random.random() * self.val

    @system_property
    def nan_val(self):
        return numpy.nan

    @system_property
    def inf_val(self):
        return numpy.inf

    if suprise:

        @system_property
        def surpise_val(self):
            return 13


@otterize
class OtherComponent(Component):
    arg1 = attr.ib(default="mmmm", validator=STR_VALIDATOR())
    arg2 = attr.ib(default="word", validator=STR_VALIDATOR())
    argnum = attr.ib(default=10)

    if suprise:
        ones = attr.ib(default=1111)
        wacky = attr.ib(
            default="one hundred and 111", validator=STR_VALIDATOR()
        )
        more = attr.ib(default="other stuff", validator=STR_VALIDATOR())

    always_save_data = True

    @system_property
    def random_val(self):
        return self.argnum * random.random() / random.random()

    if suprise:

        @system_property
        def new_val(self):
            return self.argnum * random.random()


if suprise:

    @otterize
    class SupriseComponent(Component):
        arg3 = attr.ib(default="mmmm", validator=STR_VALIDATOR())
        arg4 = attr.ib(default="word", validator=STR_VALIDATOR())
        argque = attr.ib(default=-999)
        # ones = attr.ib(default=1111)
        # wacky = attr.ib(default='one hundred and 111', validator=STR_VALIDATOR())
        # more = attr.ib(default='other stuff', validator=STR_VALIDATOR())

        always_save_data = True

        @system_property
        def negative_random_val(self):
            return self.argque * random.random() / random.random()


@otterize
class TestAnalysis(Analysis):
    if suprise:
        surprise = attr.ib(factory=SupriseComponent)

    other_cocomponent = attr.ib(factory=OtherComponent)
    internal_component = attr.ib(factory=TestComponent)
    some_random_value = attr.ib(default=10)
    other_rand_val = attr.ib(default=1e6)

    mode = "iterator"
    iterator = attr.ib(factory=lambda: list(range(10)))
    always_save_data = True

    @system_property
    def internal_val(self):
        return self.internal_component.val

    @system_property
    def rand_val(self):
        return random.random() * self.internal_component.val

    def evaluate(self, item):
        self.some_random_value = item

    @system_property
    def random_none(self):
        val = random.random()
        if val > 0.6:
            return None
        return val

    if suprise:

        @system_property
        def suprise(self):
            return random.random() * self.internal_component.val


if __name__ == "__main__":
    log = logging.getLogger()
    DEBUG = True
    if DEBUG:
        log.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        set_all_loggers_to(logging.DEBUG)

    analysis = TestAnalysis()
    tc = analysis.internal_component

    # 1) Ensure exists Reflect The Database
    db = DBConnection(
        "reports", host="localhost", user="postgres", passd="dumbpass"
    )
    db.ensure_database_exists()
    # db.engine.echo = True

    rr = ResultsRegistry(db=db)
    # rr.ensure_all_tables()
    # rr.base.metadata.drop_all()
    # rr.initalize()

    # 2) Use ComponentRegistry, and AnalysisRegistry to gather subclasses of each. Analysis will also have results tables since they are components, these will be referenced in the analysis table.
    analysis.run()
    rr.ensure_analysis(analysis)
    thread = rr.upload_analysis(analysis, use_thread=True)
    thread.join()

    # 3) For each component and analysis map create tables or map them if they dont exist.
    # - use a registry approach (singleton?) to map the {component: db_class} pairs

    # 4) When an an an analysis is solved, if configured for reporting, on post processing the results will be added to the database
