"""

"""
import pandas, attr, numpy
from ottermatics.properties import *

# import matplotlib.pyplot as plt
pandas.set_option("use_inf_as_na", True)

# Type Checking
NUMERIC_TYPES = (float, int)
NUMERIC_NAN_TYPES = (float, int, type(None))
STR_TYPES = (str,)
TABLE_TYPES = (int, float, str, type(None))


# TODO: add min / max args & attrs boilerplate
def NUMERIC_VALIDATOR():
    return attr.validators.instance_of(NUMERIC_TYPES)


def NUMERIC_NAN_VALIDATOR():
    return attr.validators.instance_of(NUMERIC_NAN_TYPES)


def STR_VALIDATOR():
    return attr.validators.instance_of(STR_TYPES)


ATTR_VALIDATOR_TYPES = (
    attr.validators._AndValidator,
    attr.validators._InstanceOfValidator,
    attr.validators._MatchesReValidator,
    attr.validators._ProvidesValidator,
    attr.validators._OptionalValidator,
    attr.validators._InValidator,
    attr.validators._IsCallableValidator,
    attr.validators._DeepIterable,
    attr.validators._DeepMapping,
)

TAB_VALIDATOR_TYPE = (
    attr.validators._InstanceOfValidator
)  # our validators should require a type i think, at least for tabulation
