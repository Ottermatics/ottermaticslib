"""This module defines PLOT and TRACE methods that allow the plotting of Statistical & Transient relationships of data in each system
"""


import attrs
import uuid
import numpy as np
import seaborn as sns
import matplotlib.pylab as pylab
from matplotlib.font_manager import get_font_names
import inspect

import typing
from ottermatics.configuration import otterize
from ottermatics.properties import *
from ottermatics.env_var import EnvVariable


class PlotLog(LoggingMixin):
    pass


log = PlotLog()


# Seaborn Config Options
SEABORN_CONTEXTS = ["paper", "talk", "poster", "notebook"]
SEABORN_THEMES = ["darkgrid", "whitegrid", "dark", "white", "ticks"]


def conv_ctx(ctx):
    if ctx.lower() not in SEABORN_CONTEXTS:
        raise ValueError(f"theme must be one of {SEABORN_CONTEXTS}")
    return ctx.lower()


def conv_theme(theme):
    if theme.lower() not in SEABORN_THEMES:
        raise ValueError(f"theme must be one of {SEABORN_THEMES}")
    return theme.lower()


# Color Maps
SEABORN_COLORMAPS = ["deep", "musted", "bright", "pastel", "dark", "colorblind"]
SEABORN_COLORMAPS += list(pylab.colormaps.keys())
SEABORN_COLORMAPS += ["husl", "hls"]
# TODO: handle other seaboorn options


def conv_maps(map):
    if map.lower() not in SEABORN_COLORMAPS:
        raise ValueError(f"theme must be one of {SEABORN_COLORMAPS}")
    return str(map.lower())


# SEABORN_FONTS = get_font_names()
# FONT_INX_LC = {k.lower():k for k in SEABORN_FONTS}
# def conv_font(fnt):
#     if fnt.lower() not in FONT_INX_LC:
#         raise ValueError(f'theme must be one of {SEABORN_FONTS}')
#     act = FONT_INX_LC[str(fnt).lower()]
#     return act


# Seaborn Config Via Env Var
_def_opts = {"obscure": False}
SEABORN_CONTEXT = EnvVariable(
    "SEABORN_CONTEXT",
    default="paper",
    type_conv=conv_ctx,
    desc=f"choose one of: {SEABORN_CONTEXTS}",
    **_def_opts,
)
SEABORN_THEME = EnvVariable(
    "SEABORN_THEME",
    default="darkgrid",
    type_conv=conv_theme,
    desc=f"choose one of: {SEABORN_THEMES}",
    **_def_opts,
)
SEABORN_PALETTE = EnvVariable(
    "SEABORN_PALETTE",
    default="deep",
    type_conv=conv_maps,
    desc=f"choose one of: {SEABORN_COLORMAPS}",
    **_def_opts,
)

#Figure Saving Config
#FIGURE_SAVE = EnvVariable(
#    ""
#)

# TODO: fonts are platform dependent :(
# SEABORN_FONT = EnvVariable('SEABORN_FONT',default='sans-serif',**_def_opts)

# Staistical View Through Seaborn:
# takes a dataframe and displays it via several types of statisical views, the default is `relplot.scatterplot` to show first order trends.
PLOT_KINDS = {
    "displot": ("histplot", "kdeplot", "ecdfplot", "rugplot"),
    "relplot": ("scatterplot", "lineplot"),
    "catplot": (
        "stripplot",
        "swarmplot",
        "boxplot",
        "violinplot",
        "pointplot",
        "barplot",
    ),
}


# Install Seaborn Themes
def install_seaborn(rc_override=None, **kwargs):
    default = dict(
        context=SEABORN_CONTEXT.secret,
        style=SEABORN_THEME.secret,
        palette=SEABORN_PALETTE.secret,
    )
    default.update(**kwargs)

    if rc_override:
        sns.set_theme(**default, rc=rc_override)
    else:
        sns.set_theme(**default)


install_seaborn()

class PlottingMixin:
    """Inherited by Systems and Analyses to provide common interface for plotting"""

    _stored_plots: dict

    @instance_cached
    def plots(self):
        return {k: getattr(self, k) for k in self.plot_attributes()}
    
    @property
    def stored_plots(self):
        if not hasattr(self,'_stored_plots'):
            self._stored_plots = {}
        return self._stored_plots

    @instance_cached
    def traces(self):
        if not self.transients_attributes():
            return {} #not a transient system
        return {k: getattr(self, k) for k in self.trace_attributes()}
    
    def make_plots(self,analysis:"Analysis"=None,store_figures:bool=True,pre=None):
        """makes plots and traces of all on this instance, and if a system is 
        subsystems. Analysis should call make plots however it can be called on a system as well
        :param analysis: the analysis that has triggered this plot
        :param store_figure: a boolean or dict, if neither a dictionary will be created and returend from this function
        :returns: the dictionary from store_figures logic
        """
        if not pre:
            pre = f'{self.classname}'
        else:
            pre =  f'{pre}.{self.classname}'

        if analysis and store_figures is True:
            imgstore = analysis._stored_plots
        elif store_figures == True:
            imgstore = {} #gotta store somewhere
        elif isinstance(store_figures,dict):
            imgstore = store_figures
        else:
            imgstore = None


        #Announce
        log.info(f'Plotting {pre}')
        
        #Traces
        for plotnm,plot in self.plots.items():
            try:
                log.info(f'{self.identity} plotting {pre}.{plotnm} | {plot}')
                fig,ax = plot()
                if isinstance(fig,pylab.Figure):
                    pylab.close(fig)
                if imgstore is not None:
                    imgstore[f'{pre}.{plotnm}']=fig
            except Exception as e:
                log.error(e,f'issue in plot {plot}')

        #Traces
        for plotnm,plot in self.traces.items():
            try:
                log.info(f'{self.identity} tracing {pre}.{plotnm} | {plot}')
                fig,ax = plot()
                if isinstance(fig,pylab.Figure):
                    pylab.close(fig)
                if imgstore is not None:
                    imgstore[f'{pre}.{plotnm}']=fig
            except Exception as e:
                log.error(e,f'issue in trace {plot}')

        #Sub Systems
        for confnm, conf in self.internal_configurations.items():
            if isinstance(conf,PlottingMixin):
                log.info(f'{self.identity} system plotting {confnm} | {conf}')
                conf.make_plots(analysis,store_figures=store_figures,pre=pre)

        return imgstore

class PlotInstance:
    """combine plotclass parms with system info"""

    plot_cls: "PLOT"
    system: "System"

    refs = None

    def __init__(self, system: "System", plot_cls: "PLOT"):
        self.plot_cls = plot_cls
        self.system = system

        #
        sys_ref = set(self.system.all_references.keys())

        diff = set()
        parms = set()
        for k,vers in self.plot_cls.plot_parms().items():
            if isinstance(vers,list):
                for v in vers:
                    if v not in sys_ref:
                        diff.add(v)
                    else:
                        parms.add(v)
            elif vers not in sys_ref:
                diff.add(vers)
            else:
                parms.add(vers)

        if self.system.log_level < 10:
            log.debug(f"system references: {sys_ref}")
            if diff:
                log.debug(f"has diff {diff}")

        if diff:
            raise KeyError(f"has system diff: {diff}")

        self.refs = {k: self.system.all_references[k] for k in parms}

    def plot(self, **kwargs):
        """applies the system dataframe to the plot"""
        return self(**kwargs)

    def __call__(self, **override_kw):
        """
        method allowing a similar type.kind(**override_kw,**default) (ie. relplot.scatterplot(x=different parameter))
        #TODO: override strategy
        """
        if not self.system.solved:
            raise ValueError(f"not solved yet!")

        PC = self.plot_cls
        f = self.plot_cls.plot_func

        # Defaults
        args = PC.plot_parms()
        if hasattr(PC, "kind"):
            args["kind"] = kind = PC.kind.replace("plot", "")

        # these go in plot
        extra = PC.plot_extra()

        # Parse title
        title = self.plot_cls.title
        if "title" in override_kw:
            title = override_kw.pop("title")

        # Announce Override
        if override_kw:
            log.debug(f"overriding parms {override_kw}")
            args.update(**override_kw)

        log.info(
            f"plotting {self.system.identity}| {self.identity} with {args}"
        )
        fig = ax = f(data=self.system.dataframe, **args, **extra)

        return self.process_fig(fig, title)

    def process_fig(self, fig, title):
        if isinstance(fig, pylab.Axes):
            ax = fig
            fig = fig.fig
        elif isinstance(fig, sns.FacetGrid):
            ax = fig
            fig = fig.fig
        else:
            ax = fig

        # Polish Fig Args
        if title:
            fig.subplots_adjust(top=0.9)
            fig.suptitle(title)

        return fig,ax

    @property
    def identity(self) -> str:
        return f"{self.plot_cls.type}.{self.plot_cls.kind}"


class PLOT_ATTR(attrs.Attribute):
    """base class for plot attributes"""

    name: str
    on_system: "System"
    title: str = None
    kind: str
    cls_parms = {"x", "y"}

    @classmethod
    def create_instance(cls, system: "System"):
        raise NotImplemented("need to implement on subclass")

    @classmethod
    def configure_for_system(cls, name, system):
        """add the system class, and check the dependent and independent values"""
        cls.name = name
        cls.on_system = system

    @classmethod
    def plot_parms(cls) -> dict:
        """gathers seaborn plot parms that will scope from system.dataframe"""
        p = {}
        p["x"] = cls.x
        p["y"] = cls.y
        if cls.hue:
            p["hue"] = cls.hue
        if cls.col:
            p["col"] = cls.col
        if cls.row:
            p["row"] = cls.row

        # Add the options
        parm_opts = cls.type_parm_options[cls.type]
        for k, arg in cls.plot_args.items():
            if k in parm_opts:
                p[k] = arg
        return p

    @classmethod
    def plot_extra(cls):
        plot_parms = cls.plot_parms()
        out = {}
        for k, arg in cls.plot_args.items():
            if k not in plot_parms:
                out[k] = arg
        return out

    @classmethod
    def validate_plot_args(cls, system: "System"):
        """Checks system.system_references that cls.plot_parms exists"""
        log.info(f'validating: {system}')
        sys_ref = system.system_references
        attr_keys = set(sys_ref["attributes"].keys())
        prop_keys = set(sys_ref["properties"].keys())
        valid = attr_keys.union(prop_keys)

        diff = set()
        for k,vers in cls.plot_parms().items():
            if isinstance(vers,list):
                for v in vers:
                    if v not in valid:
                        diff.add(v)
            elif vers not in valid:
                diff.add(vers)
            
        if log.log_level <= 10:
            log.debug(
                f"{cls.__name__} has parms: {attr_keys} and bad input: {diff}"
            )

        if diff:
            raise KeyError(
                f"bad plot parms: {diff} do not exist in system: {valid}"
            )

    # TODO: there's a pattern here with these attrs.Attributes
    @classmethod
    def make_plot_factory(cls):
        return attrs.Factory(cls.create_instance, takes_self=True)

    @classmethod
    def create_instance(cls, system: "System") -> PlotInstance:
        cls.validate_plot_args(system)
        return PlotInstance(system, cls)


trace_type = typing.Union[str, list]


class TraceInstance(PlotInstance):
    @classmethod
    def plot_extra(cls) -> dict:
        return cls.plot_cls.extra_args

    def __call__(self, **override_kw):
        """
        method allowing a similar type.kind(**override_kw,**default) (ie. relplot.scatterplot(x=different parameter))
        #TODO: override strategy
        """
        if not self.system.solved:
            raise ValueError(f"not solved yet!")

        PC = self.plot_cls

        type = PC.type
        types = self.plot_cls.types
        if 'type' in override_kw and override_kw['type'] in types:
            type = override_kw.pop('type')
        elif 'type' in override_kw:
            raise KeyError(f'invalid trace type, must be in {types}')

        if type == 'scatter':
            f = lambda ax,*args,**kwargs: ax.scatter(*args,**kwargs)
        elif type == 'line':
            f = lambda ax,*args,**kwargs: ax.plot(*args,**kwargs)

        # Defaults
        args = PC.plot_args.copy()

        # these go in plot
        extra = PC.plot_extra()

        # Parse title
        title = PC.title
        if "title" in override_kw:
            title = override_kw.pop("title")

        # Announce Override
        if override_kw:
            log.debug(f"overriding parms {override_kw}")
            args.update(**override_kw)

        log.info(
            f"plotting {self.system.identity}| {self.identity} with {args}"
        )

        #PLOTTING
        # Make the axes and plot
        #Get The MVPS
        x =args.pop('x')
        if 'x' in override_kw:
            x = override_kw['x']
        y=args.pop('y')
        if 'y' in override_kw:
            y = override_kw['y']
        if not isinstance(y,list): y = list(y)
        
        #secondary plot
        if "y2" in args and args["y2"]:
            y2=args.pop('y2')
            if 'y2' in override_kw:
                y2 = override_kw['y2']
            if y2 is None: 
                y2 = []
            elif not isinstance(y2,list): 
                y2 = list(y2)
        else:
            y2 = []

        #TODO: insert marker, color ect per group, ensure no overlap
        fig, axes = pylab.subplots(2 if y2 else 1, 1)
        if y2:
            ax,ax2=axes[0],axes[1]
        else:
            ax = axes
            ax2=None


        #Loop over y1
        yleg = []
        for y in y:
            f(ax,x,y,data=self.system.dataframe,label=y,**args,**extra)
        else:
            # The only specificity of the code is when plotting the legend
            h, l = np.hstack([l.get_legend_handles_labels()
                            for l in ax.figure.axes
                            if l.bbox.bounds == ax.bbox.bounds]).tolist()
            ax.legend(handles=h, labels=l, loc='upper right')               

        #Loop over y2
        for y in y2:
            f(ax2,x,y,data=self.system.dataframe,label=y,**args,**extra)     
        else:
            # The only specificity of the code is when plotting the legend
            h, l = np.hstack([l.get_legend_handles_labels()
                            for l in ax2.figure.axes
                            if l.bbox.bounds == ax2.bbox.bounds]).tolist()
            ax2.legend(handles=h, labels=l, loc='upper right')                        

        return self.process_fig(fig, title)


class TRACE(PLOT_ATTR):
    """trace is a plot for transients, with y and y2 axes which can have multiple parameters each"""

    types = ["scatter", "line"]
    type = "scatter"

    # mainem
    y2: trace_type
    y: trace_type
    x: str

    plot_args: dict
    extra_args: dict

    # Extended parameters per option
    type_parm_options = {
        "scatter": ("size", "color"),
        "line": ("color", "marker", "linestyle"),
    }

    type_options = {
        "scatter": ("vmax", "vmin", "marker", "alpha", "cmap"),
        "line": ("linewidth",),
    }
    all_options = (
        "xlabel",
        "ylabel",
        "y2label",
        "title",
        "xticks",
        "yticks",
        "alpha",
    )

    always = ("x", "y", "y2")

    @classmethod
    def define(
        cls, x="time", y: trace_type = None, y2=None, kind="line", **kwargs
    ):
        """Defines a plot that will be matplotlib, with validation happening as much as possible in the define method

        #Plot Choice
        :param kind: specify the kind of type of plot scatter or line, with the default being line

        # Dependents & Independents:
        :param x: the x parameter for each plot, by default 'time'
        :param y: the y parameter for each plot, required
        :param y2: the y2 parameter for each plot, optional

        # Additional Parameters:
        :param title: this title will be applied to the figure.suptitle()
        :param xlabel: the x label, by default the capitalized parameter
        :param ylabel: the x label, by default the capitalized parameter
        :param y2label: the x label, by default the capitalized parameter
        """

        if x is None or y is None:
            raise ValueError(f"x and y must both be input")

        if not isinstance(x,str):
            raise ValueError(f'x must be string')
        
        if not isinstance(y,(list,str)):
            raise ValueError(f'y must be string or list')

        if not any([ y2 is None, isinstance(y2,(list,str)) ]):
            raise ValueError(f'y2 must be string or list')

        # Validate Plot
        assert kind in cls.types, f"invalid kind not in {cls.types}"
        if kind == "line":
            pfunc = pylab.plot
        elif kind == "scatter":
            pfunc = pylab.scatter
        else:
            raise KeyError(f"bad plot kind: {kind}")

        # Remove special args
        title = None
        if "title" in kwargs:
            title = kwargs.pop("title")

        # Remove special args
        non_parm_args = cls.valid_non_parms(kind)
        parm_args = cls.valid_parms_options(kind)
        extra = {}

        for k in list(kwargs.keys()):
            if k in non_parm_args:
                extra[k] = kwargs.pop(k)

        # Validate Args
        assert set(kwargs).issubset(parm_args), f"invalid plot args {kwargs}"
        plot_args = kwargs
        plot_args['x'] = x
        plot_args['y'] = y
        
        if 'y2':
            plot_args['y2'] = y2

        log.info(f"adding TRACE|{kind} {x},{y},{y2},{kwargs}")

        # Create A New Signals Class
        new_name = f"TRACE_x_{x}_y_{y}_{str(uuid.uuid4())}".replace(
            ".", "_"
        ).replace("-", "")
        new_dict = dict(
            name=new_name,
            x=x,
            y=y,
            y2=y2,
            plot_func=pfunc,
            plot_args=plot_args,
            extra_args=extra,
            title=title,
            kind=kind,
        )
        new_plot = type(new_name, (cls,), new_dict)

        return new_plot

    @classmethod
    def valid_non_parms(cls, type) -> set:
        s = set(cls.all_options)
        s = s.union(set(cls.type_options[type]))
        return s

    @classmethod
    def valid_parms_options(cls, type) -> set:
        s = set(cls.always)
        s = s.union(set(cls.type_parm_options[type]))
        return s

    @classmethod
    def plot_parms(cls) -> set:
        pa = cls.plot_args.copy()
        
        y1 = pa.pop("y")
        c = set()
        if isinstance(y1, list):
            for y in y1:
                c.add(y)
        else:
            c.add(y1)
        pa['y'] = list(c)

        if "y2" in c:
            c = set()
            y2 = pa.pop("y2")
            if isinstance(y2, list):
                for y in y2:
                    c.add(y)
            else:
                c.add(y2)

            pa['y2'] = list(c)
        return pa

    @classmethod
    def create_instance(cls, system: "System") -> TraceInstance:
        cls.validate_plot_args(system)
        return TraceInstance(system, cls)


class PLOT(PLOT_ATTR):
    """Plot is a conveinence method"""

    types: tuple = ("displot", "relplot", "catplot")
    std_fields: tuple = ("x", "y", "col", "hue", "row")

    # These options must be filled out
    type: str
    kind: str
    x: str
    y: str

    # optional, must be parm
    hue: str
    col: str
    row: str

    # Extended parameters per option
    type_parm_options = {
        "displot": (),
        "relplot": ("style", "shape"),
        "catplot": (),
    }

    # These capture the function and extra keywords
    plot_func: None
    plot_args: dict

    @classmethod
    def define(
        cls,
        x,
        y,
        _type="relplot",
        kind="scatterplot",
        row=None,
        col=None,
        hue=None,
        **kwargs,
    ):
        """Defines a plot that will be rendered in seaborn, with validation happening as much as possible in the define method

        #Plot Choice
        :param _type: the type of seaborn plot (relplot,displot,catplot)
        :param kind: specify the kind of type of plot (ie. scatterplot of relplot)

        # Dependents & Independents:
        :param x: the x parameter for each plot
        :param y: the y parameter for each plot

        # Additional Parameters:
        :param row: create a grid of data with row parameter
        :param col: create a grid of data with column parameter
        :param hue: provide an additional dimension of color based on this parameter
        :param title: this title will be applied to the figure.suptitle()
        """

        # Validate Plot
        assert (
            _type in PLOT_KINDS
        ), f"type {_type} must be in {PLOT_KINDS.keys()}"
        kinds = PLOT_KINDS[_type]
        assert kind in kinds, f"plot kind {kind} not in {kinds}"

        # Remove special args
        title = None
        if "title" in kwargs:
            title = kwargs.pop("title")

        # Validate Args
        pfunc = getattr(sns, _type)
        kfunc = getattr(sns, kind)
        args = set(inspect.signature(kfunc).parameters.keys())
        assert set(kwargs).issubset(args), f"only {args} allowed for kw"
        plot_args = kwargs

        log.info(
            f"adding PLOT|{_type}.{kind}({x},{y},hue={hue},c:[{col}],r:[{row}],{kwargs}"
        )

        # Create A New Signals Class
        new_name = f"PLOT_x_{x}_y_{y}_{str(uuid.uuid4())}".replace(
            ".", "_"
        ).replace("-", "")
        new_dict = dict(
            name=new_name,
            x=x,
            y=y,
            hue=hue,
            row=row,
            col=col,
            type=_type,
            kind=kind,
            plot_func=pfunc,
            plot_args=plot_args,
            title=title,
        )
        new_plot = type(new_name, (cls,), new_dict)

        return new_plot

    @classmethod
    def plot_parms(cls) -> dict:
        """gathers seaborn plot parms that will scope from system.dataframe"""
        p = {}
        p["x"] = cls.x
        p["y"] = cls.y
        if cls.hue:
            p["hue"] = cls.hue
        if cls.col:
            p["col"] = cls.col
        if cls.row:
            p["row"] = cls.row

        # Add the options
        parm_opts = cls.type_parm_options[cls.type]
        for k, arg in cls.plot_args.items():
            if k in parm_opts:
                p[k] = arg
        return p
