import attr
from ottermatics.configuration import otterize, Configuration
from ottermatics.components import Component
from ottermatics.tabulation import TabulationMixin
from ottermatics.system import System
from ottermatics.typing import *
from ottermatics.reporting import *

# import datetime
import os
from uuid import uuid4

import random
import attrs

from contextlib import contextmanager
import inspect


list_check = attrs.validators.instance_of(list)


def make_reporter_check(type_to_check):
    def reporter_type_check(inst, attr, value):
        if not isinstance(value, list):
            raise ValueError("must be a list!")

        if not all([isinstance(v, type_to_check) for v in value]):
            raise ValueError("must be a list of type {value.__name__}!")

    return reporter_type_check


@otterize
class Analysis(TabulationMixin):
    """Analysis takes a system and many reporters, runs the system, adds its own system properties to the dataframe and post processes the results

    post_process() typically creates plots, but can be overriden
    """

    system: System = attrs.field()
    table_reporters: list = attrs.field(
        factory=list, validator=make_reporter_check(TableReporter)
    )
    plot_reporters: list = attrs.field(
        factory=list, validator=make_reporter_check(PlotReporter)
    )

    _stored_plots:list = attrs.field(factory=list)
    _uploaded = False

    @property
    def uploaded(self):
        return self._uploaded

    def run(self, *args, **kwargs):
        self.info(f"running results with input {args} {kwargs}")
        out = self.system.run(*args, **kwargs, cb=self.save_data)
        self.post_process(*args, **kwargs)
        self.make_plots()
        self.report_results()

    def post_process(self, *args, **kwargs):
        """A user customizeable function"""
        pass

    def make_plots(self):
        """creates user PLOTs and TRACESs (for transients only)"""
        # TODO:  define PLOT attribute
        # TODO: define TRACE attribute
        pass

    def report_results(self):
        self.info(f"report results")

        for tbl_reporter in self.table_reporters:
            tbl_reporter.upload(self)

        for plt_reporter in self.plot_reporters:
            plt_reporter.upload(self)

        self._uploaded = True

#     def go_through_components(
#         self, level=0, levels_to_descend=-1, parent_level=0
#     ):
#         """A generator that will go through all internal configurations up to a certain level
#         if levels_to_descend is less than 0 ie(-1) it will go down, if it 0, None, or False it will
#         only go through this configuration
# 
#         :return: level,config"""
# 
#         should_yield_level = lambda level: all(
#             [
#                 level >= parent_level,
#                 any([levels_to_descend < 0, level <= levels_to_descend]),
#             ]
#         )
# 
#         if should_yield_level(level):
#             yield level, self
# 
#         level += 1
#         for comp in self.internal_components.values():
#             for level, icomp in comp.go_through_components(
#                 level, levels_to_descend, parent_level
#             ):
#                 yield level, icomp
# 
#     @property
#     def all_internal_components(self):
#         return list(
#             [
#                 comp
#                 for lvl, comp in self.go_through_components()
#                 if not self is comp
#             ]
#         )
# 
#     @property
#     def unique_internal_components_classes(self):
#         return list(
#             set(
#                 [
#                     comp.__class__
#                     for lvl, comp in self.go_through_components()
#                     if not self.__class__ is comp.__class__
#                 ]
#             )
#         )

    # Plotting & Report Methods:
    @property
    def saved_plots(self):
        return self._stored_plots

    @property
    def plotting_methods(self):
        return {
            fname: func
            for fname, func in inspect.getmembers(
                self, predicate=inspect.ismethod
            )
            if fname.startswith("plot_")
        }

    @contextmanager
    def subplots(self, plot_tile, save=True, *args, **kwargs):
        """context manager for matplotlib subplots, which will save the plot if no failures occured
        using a context manager makes sense so we can record the plots made, and then upload them in
        the post processing steps.

        it makes sense to always save images, but to override them.
        plot id should be identity+plot_title and these should be stored by date in the report path
        """
        fig, maxes = plt.subplots(*args, **kwargs)

        try:
            yield fig, maxes

            if save:
                # determine file name
                filename = "{}_{}".format(self.filename, plot_tile)
                supported_filetypes = plt.gcf().canvas.get_supported_filetypes()
                if not any(
                    [
                        filename.endswith(ext)
                        for ext in supported_filetypes.keys()
                    ]
                ):
                    if "." in filename:
                        filename = (
                            ".".join(filename.split(".")[:-1])
                            + self._default_image_format
                        )
                    else:
                        filename += self._default_image_format

                filepath = os.path.join(self.config_path_daily, filename)

                self.info("saving plot {}".format(filename))
                fig.savefig(filepath)

                self._stored_plots.append(filepath)

        except Exception as e:
            self.error(e, "issue plotting {}".format(plot_tile))


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
#                 self.warning(f"parameters not found: {rmv_y}")
#
#         if self.solved and y:
#             df = self.dataframe
#             return df.plot(x=x, y=y, kind=kind, **kwargs)
#         elif y:
#             self.warning("not solved yet!")
#         elif self.solved:
#             self.warning("bad input!")
