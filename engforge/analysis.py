import attr
from engforge.configuration import forge, Configuration
from engforge.components import Component
from engforge.tabulation import TabulationMixin, DataframeMixin
from engforge.system import System
from engforge.typing import *
from engforge.reporting import *
from engforge.attr_plotting import PlottingMixin


# import datetime
import os
from uuid import uuid4

import random
import attrs

from contextlib import contextmanager
import inspect

import matplotlib.pylab as pylab

list_check = attrs.validators.instance_of(list)


def make_reporter_check(type_to_check):
    def reporter_type_check(inst, attr, value):
        if not isinstance(value, list):
            raise ValueError("must be a list!")

        if not all([isinstance(v, type_to_check) for v in value]):
            raise ValueError("must be a list of type {value.__name__}!")

    return reporter_type_check


@forge
class Analysis(Configuration, TabulationMixin, PlottingMixin, DataframeMixin):
    """Analysis takes a system and many reporters, runs the system, adds its own system properties to the dataframe and post processes the results

    make_plots() makes plots from the analysis, and stores figure
    post_process()  but can be overriden
    report_results() writes to reporters
    """

    # TODO: generate pdf with tables ect.

    system: System = attrs.field()
    table_reporters: list = attrs.field(
        factory=list, validator=make_reporter_check(TableReporter)
    )
    plot_reporters: list = attrs.field(
        factory=list, validator=make_reporter_check(PlotReporter)
    )

    _stored_plots: dict = attrs.field(factory=dict)
    _uploaded = False

    show_plots: bool = attrs.field(default=True)

    @property
    def uploaded(self):
        return self._uploaded
    

    def run(self, *args, **kwargs):
        """Analysis.run() passes inputs to the assigned system and saves data via the system.run(cb=callback), once complete `Analysis.post_process()` is run also being passed input arguments, then plots & reports are made"""
        self.info(
            f"running analysis {self.identity} with input {args} {kwargs}"
        )
        cb = lambda *args, **kw: self.system.last_context.save_data(force=True)
        out = self.system.run(*args, **kwargs, cb=cb)
        self.post_process(*args, **kwargs)

        self._stored_plots = {}
        self.make_plots(analysis=self, store_figures=self._stored_plots)
        self.report_results()

    def post_process(self, *args, **kwargs):
        """A user customizeable function"""
        pass

    def report_results(self):
        self.info(f"report results")

        for tbl_reporter in self.table_reporters:
            try:
                tbl_reporter.upload(self)
            except Exception as e:
                self.error(e, "issue in {tbl_reporter}")

        for plt_reporter in self.plot_reporters:
            try:
                plt_reporter.upload(self)
            except Exception as e:
                self.error(e, "issue in {plt_reporter}")

        if self.show_plots:
            self.info(f"showing plots {len(self.stored_plots)}")
            for figkey, fig in self.stored_plots.items():
                self.info(f"showing {figkey}")
                try:
                    fig.show()
                except Exception as e:
                    self.error(e, f"issue showing {figkey}")

        self._uploaded = True

    @property
    def dataframe(self):
        # TODO: join with analysis dataframe
        return self.system.dataframe
































    # Plotting & Report Methods:


#     @property
#     def _report_path(self):
#         """Add some name options that work into ClientInfoMixin"""
#
#         if (
#             self.namepath_mode == "both"
#             and self.mode == "iterator"
#             and isinstance(self.component_iterator, Component)
#         ):
#             return os.path.join(
#                 self.namepath_root,
#                 f"{self.name}",
#                 f"{self.component_iterator.name}",
#             )
#         elif (
#             self.namepath_mode == "iterator"
#             and self.mode == "iterator"
#             and isinstance(self.component_iterator, Component)
#         ):
#             return os.path.join(
#                 self.namepath_root, f"{self.component_iterator.name}"
#             )
#         else:
#             if self.name != "default":
#                 return os.path.join(
#                     self.namepath_root, f"{self.classname.lower()}_{self.name}"
#                 )
#             return os.path.join(self.namepath_root, f"{self.classname.lower()}")

# @property
# def component_iterator(self) -> ComponentIterator:
#     """Override me!"""
#     return self.iterator

# def post_process(self):
#     """override me!"""
#     pass
#
#     def reset_analysis(self):
#         self.reset_data()
#         self._solved = False
#         self.run_id = None

#     def gsync_results(self, filename="Analysis", meta_tags=None):
#         """Syncs All Variable Tables To The Cloud"""
#         with self.drive.context(
#             filepath_root=self.local_sync_path, sync_root=self.cloud_sync_path
#         ) as gdrive:
#             with self.drive.rate_limit_manager(
#                 self.gsync_results, 6, filename=filename, meta_tags=meta_tags
#             ):
#                 old_sleep = gdrive._sleep_time
#                 gdrive.reset_sleep_time(max(old_sleep, 1.0))
#
#                 gpath = gdrive.sync_path(self.local_sync_path)
#
#                 self.debug(f"saving as gsheets {gpath}")
#                 parent_id = gdrive.get_gpath_id(gpath)
#                 # TODO: delete old file if exists
#
#                 gdrive.sleep(12 * random.random())
#
#                 gdrive.cache_directory(parent_id)
#                 gdrive.sleep()
#
#                 # Remove items with same name in parent dir
#                 parent = gdrive.item_nodes[parent_id]
#                 parent.remove_contents_with_title(filename)
#
#                 df = self.joined_dataframe
#
#                 # Make the new sheet
#                 sht = gdrive.gsheets.create(filename, folder=parent_id)
#                 gdrive.sleep(2 * (1 + gdrive.time_fuzz * random.random()))
#
#                 wk = sht.add_worksheet(filename)
#                 gdrive.sleep(2 * (1 + gdrive.time_fuzz * random.random()))
#
#                 wk.rows = df.shape[0]
#                 gdrive.sleep(2 * (1 + gdrive.time_fuzz * random.random()))
#
#                 wk.set_dataframe(df, start="A1", fit=True)
#                 gdrive.sleep(2 * (1 + gdrive.time_fuzz * random.random()))
#
#                 for df_result in self.variable_tables:
#                     df = df_result["df"]
#                     conf = df_result["conf"]
#
#                     if meta_tags is not None and type(meta_tags) is dict:
#                         for tag, value in meta_tags.items():
#                             df[tag] = value
#
#                     gdrive.sleep(2 * (1 + gdrive.time_fuzz * random.random()))
#                     wk = sht.add_worksheet(conf.displayname)
#                     gdrive.sleep(2 * (1 + gdrive.time_fuzz * random.random()))
#
#                     wk.rows = df.shape[0]
#                     gdrive.sleep(2 * (1 + gdrive.time_fuzz * random.random()))
#
#                     wk.set_dataframe(df, start="A1", fit=True)
#                     gdrive.sleep(2 * (1 + gdrive.time_fuzz * random.random()))
#
#                 sht.del_worksheet(sht.sheet1)
#                 gdrive.sleep(2 * (1 + gdrive.time_fuzz * random.random()))
#
#                 # TODO: add in dataframe dict with schema sheename: {dataframe,**other_args}
#                 self.info(
#                     "gsheet saved -> {}".format(os.path.join(gpath, filename))
#                 )
#
#                 gdrive.reset_sleep_time(old_sleep)

#     @property
#     def columns(self):
#         if self.solved:
#             return list(self.joined_dataframe)
#         else:
#             return []
#
#     def plot(self, x, y, kind="line", **kwargs):
#         """
#         A wrapper for pandas dataframe.plot
#         :param grid: set True if input is not False
#         """
#
#         # TODO: Add a default x iterator for what is iterated in analysis!
#
#         if "grid" not in kwargs:
#             kwargs["grid"] = True
#
#         if isinstance(y, (list, tuple)):
#             old_y = set(y)
#             y = list([yval for yval in y if yval in self.dataframe.columns])
#             rmv_y = set.difference(old_y, set(y))
#             if rmv_y:
#                 self.warning(f"vars not found: {rmv_y}")
#
#         if self.solved and y:
#             df = self.dataframe
#             return df.plot(x=x, y=y, kind=kind, **kwargs)
#         elif y:
#             self.warning("not solved yet!")
#         elif self.solved:
#             self.warning("bad input!")
