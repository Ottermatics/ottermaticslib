"""

"""
import pandas, attr, numpy
from ottermatics.properties import *
import attrs

# import matplotlib.pyplot as plt
pandas.set_option("use_inf_as_na", True)

# Type Checking
NUMERIC_TYPES = (float, int, numpy.int64, numpy.float64)
NUMERIC_NAN_TYPES = (float, int, type(None), numpy.int64, numpy.float64)
STR_TYPES = (str,numpy.string_)
TABLE_TYPES = (int, float, str, type(None), numpy.int64, numpy.float64)

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


# Improved Attrs Creation Fields
def Options(*choices, **kwargs):
    """creates an attrs field with validated choices on init and setattr
    :param choices: a list of choices that are validated on input, the first becoming the default unless it is passed in kwargs
    :param kwargs: keyword args passed to attrs field"""
    assert choices, f"must have some choices!"
    assert "type" not in kwargs, "options type set is str"
    assert set([type(c) for c in choices]) == set((str,)), "choices must be str"
    assert "on_setattr" not in kwargs

    validators = [attrs.validators.in_(choices)]

    # Merge Validators
    if "validators" in kwargs:
        in_validators = kwargs.pop("validators")
        if isinstance(in_validators, list):
            validators.extend(in_validators)
        elif isinstance(in_validators, attr.validators._ValidatorType):
            validators.append(in_validators)
        else:
            raise ValueError(f"bad validator {in_validators}")

    # Default
    if "default" in kwargs:
        default = kwargs.pop("default")
        assert type(default) is str
    else:
        default = choices[0]

    on_setattr = [attrs.setters.validate]

    # Create The Attr!
    a = attrs.field(
        default=default,
        type=str,
        validator=validators,
        on_setattr=on_setattr,
        **kwargs,
    )
    return a


# def Numeric #TODO with min/max that is enforced in solver!
