"""Dataframe Module:

Store data in dataframes and provide a simple interface to manipulate it.
"""

from contextlib import contextmanager
import attr

from engforge.common import inst_vectorize, chunks
from engforge.properties import engforge_prop
# from engforge.configuration import Configuration, forge
from engforge.logging import LoggingMixin
from engforge.typing import *
from engforge.properties import *
from typing import Callable

import numpy
import pandas
import os
import collections
import uuid


class DataFrameLog(LoggingMixin):
    pass


log = DataFrameLog()


# Dataframe interrogation functions
def is_uniform(s: pandas.Series):
    a = s.to_numpy()  # s.values (pandas<0.24)
    if (a[0] == a).all():
        return True
    try:
        if not numpy.isfinite(a).any():
            return True
    except:
        pass
    return False


# key_func = lambda kv: len(kv[0].split('.'))*len(kv[1])
# length of matches / length of key
key_func = lambda kv: len(kv[1]) / len(kv[0].split("."))
# key_func = lambda kv: len(kv[1])


# TODO: remove duplicate columns
# mtches = collections.defaultdict(set)
# dfv = ecs.dataframe_variants[0]
# for v1,v2 in itertools.combinations(dfv.columns,2):
#     if numpy.all(dfv[v1]==dfv[v2]):
#
#         mtches[v1].add(v2)
#         mtches[v2].add(v1)

#TODO: integrate statistical output of dataframe, if at all in problem domain
#1. stats_mode: mean, median,min,max mode, std, var, skew, kurtosis
#2. min_mode: mean,median,std,min,max
#3. sub_mode: store the dataframe completely separately
class dataframe_property(engforge_prop):
    pass


#aliases
dataframe_prop = dataframe_property
df_prop = dataframe_property


def determine_split(raw, top: int = 1, key_f=key_func):
    parents = {}

    for rw in raw:
        grp = rw.split(".")
        for i in range(len(grp)):
            tkn = ".".join(grp[0 : i + 1])
            parents[tkn] = set()

    for rw in raw:
        for par in parents:
            if rw.startswith(par):
                parents[par].add(rw)

    grps = sorted(parents.items(), key=key_f, reverse=True)[:top]
    return [g[0] for g in grps]


def split_dataframe(df: pandas.DataFrame) -> tuple:
    """split dataframe into a dictionary of invariants and a dataframe of variable values

    :returns tuple: constants,dataframe
    """
    uniform = {}
    for s in df:
        c = df[s]
        if is_uniform(c):
            uniform[s] = c[0]

    df_unique = df.copy().drop(columns=list(uniform))
    return uniform, df_unique if len(df_unique) > 0 else df_unique


class DataframeMixin:
    dataframe: pandas.DataFrame

    _split_dataframe_func = split_dataframe
    _determine_split_func = determine_split

    def smart_split_dataframe(self, df=None, split_groups=0, key_f=key_func):
        """splits dataframe between constant values and variants"""
        if df is None:
            df = self.dataframe
        out = {}
        const, vardf = split_dataframe(df)
        out["constants"] = const
        columns = set(vardf.columns)
        split_groups = min(split_groups, len(columns) - 1)
        if split_groups == 0:
            out["variants"] = vardf
        else:
            nconst = {}
            cgrp = determine_split(const, min(split_groups, len(const) - 1))
            for i, grp in enumerate(sorted(cgrp, reverse=True)):
                columns = set(const)
                bad_columns = [c for c in columns if not c.startswith(grp)]
                good_columns = [c for c in columns if c.startswith(grp)]
                nconst[grp] = {c: const[c] for c in good_columns}
                for c in good_columns:
                    if c in columns:
                        columns.remove(c)
            out["constants"] = nconst

            raw = sorted(set(df.columns))
            grps = determine_split(raw, split_groups, key_f=key_f)

            for i, grp in enumerate(sorted(grps, reverse=True)):
                columns = set(vardf.columns)
                bad_columns = [c for c in columns if not c.startswith(grp)]
                good_columns = [c for c in columns if c.startswith(grp)]
                out[grp] = vardf.copy().drop(columns=bad_columns)
                # remove columns from vardf
                vardf = vardf.drop(columns=good_columns)
            if vardf.size > 0:
                out["misc"] = vardf
        return out

    @solver_cached
    def _split_dataframe(self):
        """splits dataframe between constant values and variants"""
        return split_dataframe(self.dataframe)

    @property
    def dataframe_constants(self):
        return self._split_dataframe[0]

    @property
    def dataframe_variants(self):
        o = self._split_dataframe[1:]
        if isinstance(o, list):
            if len(o) == 1:
                return o[0]
        return o

    def format_columns(self, dataframe: pandas.DataFrame):
        dataframe.rename(
            lambda x: x.replace(".", "_").lower(), axis="columns", inplace=True
        )

    # Plotting Interface
    @property
    def skip_plot_vars(self) -> list:
        """accesses '_skip_plot_vars' if it exists, otherwise returns empty list"""
        if hasattr(self, "_skip_plot_vars"):
            return [var.lower() for var in self._skip_plot_vars]
        return []

    @property
    def dataframe(self):
        """returns the dataframe"""
        raise NotImplementedError("must be implemented in subclass")
