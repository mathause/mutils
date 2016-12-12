#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Hauser
# Date:

import cartopy.crs as ccrs
import collections
# import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mplcolors
import numpy as np
import string

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime


def get_months(day_of_month=15, extend=False):
    """
    one date for every month of the year

    Parameters
    ----------
    day_of_month : integer
        on which day of the month the date should be
    extend : bool
        if True prepends and appends one month

    Returns
    -------
    dates : ndarray
        one entry per day
    """


    months = range(1, 13)

    year = [2]*12

    if extend:
        months = [12] + months + [1]
        year = [1] + year + [3]

    dates = np.zeros_like(months)

    for i, month in enumerate(months):
        date = (year[i], month, day_of_month)
        dates[i] = mdates.date2num(datetime(*date))

    return dates

# =============================================================================

def get_doy(dt=1., begday=(2, 1, 1), endday=(2, 12, 31)):
    """
    day of the year, for year 2

    Parameters
    ----------
    dt : float
        delta time in days, default: 1
    begday : (year, month, day)
        first day of range 
    endday : (year, month, day)
        last day of range 

    Returns
    -------
    days : ndarray
        numbered from begday to endday

    """

    begday = mdates.date2num(datetime(*begday))
    endday = mdates.date2num(datetime(*endday))

    days = np.arange(begday, endday + 1, dt)
    return days


# =============================================================================

def format_ax_months(ax=None, xlim=(366, 365*2), date_format='%b', xlabel=True):
    """
    month name at the 15th, xtick line at the 1st

    Parameters
    ----------
    ax : ax : matplotlib Axes
        Axes object to format, otherwise uses current axes.
    xlim : (xmin, xmax)
        extend of the x axis
    date_format : strftime formatter
        format of the xtick labels

    """

    if ax is None:
        ax = plt.gca()

    # format x axis
    ax.xaxis.set_major_locator(mdates.DayLocator(1))
    ax.xaxis.set_minor_locator(mdates.DayLocator(15))
    
    if xlabel:
        hfmt = mdates.DateFormatter(date_format)
        ax.xaxis.set_minor_formatter(hfmt)

    ax.tick_params(which='minor', length=0)

    ax.set_xticklabels([])

    ax.set_xlim(xlim)


# =============================================================================

def equal_axes(axes, xlim=True, ylim=True):
    """
    adjust xlim and ylim to the min/ max of all axes

    Parameters
    ----------
    axes: list
        axes to adjust
    xlim : bool, optional
        If true, adjust xlim.
    ylim : bool, optional
        If true, adjust ylim.

    """
    axes = np.array(axes).flatten()

    if xlim:
        mn = min([ax.get_xlim()[0] for ax in axes])
        mx = max([ax.get_xlim()[1] for ax in axes])
        [ax.set_xlim(mn, mx) for ax in axes]

    if ylim:
        mn = min([ax.get_ylim()[0] for ax in axes])
        mx = max([ax.get_ylim()[1] for ax in axes])
        [ax.set_ylim(mn, mx) for ax in axes]



# =============================================================================

def geoaxis_simple(gs, projection=ccrs.PlateCarree(), coastlines=True):

    ax = plt.subplot(gs, projection=projection, axisbg=[1, 0, 0])
    # ax.set_extent([-122, -115, 30, 50], ccrs.Geodetic())
    if coastlines:
        ax.coastlines(linewidth=0.75, color='0.25')
    return ax


# =============================================================================

def get_subplots(x, y, geoaxis=True, flatten=True, 
                 projection=ccrs.PlateCarree(), coastlines=True, 
                 **kwargs):

    gs = gridspec.GridSpec(x, y, **kwargs)

    nplot = x*y

    if flatten:
        axes = np.ones(shape=nplot, dtype=np.object)
    else:
        axes = np.ones(shape=(x, y), dtype=np.object)

    if np.iterable(geoaxis):
        geoaxis = np.array(geoaxis).flatten()
        if nplot != axes.shape[0]:
            raise RuntimeError("Wrong Number of items in geoaxis.")
    else:
        geoaxis = [geoaxis] * nplot

    for i, g in enumerate(gs):
        if geoaxis[i]:
            axes[i] = geoaxis_simple(g, projection, coastlines)
        else:
            axes[i] = plt.subplot(g)

    f = plt.gcf()    

    return f, axes


# =============================================================================

def legend(*args, **kwargs):
    """
    add legend and format it properly

    Parameters
    ----------
    *args : non-keyword args
        see matplotlib legend held
    ax : matplotlib axis
        axis to add legend to
    **kwargs ; keyword args
        see matplotlib legend help
    """

    ax = kwargs.pop('ax', plt.gca())

    facecolor = kwargs.pop('facecolor', '0.95')
    linewidth = kwargs.pop('linewidth', 0.0)

    kwargs['frameon'] = kwargs.pop('frameon', True)

    legend = ax.legend(*args, **kwargs)
    rect = legend.get_frame()

    rect.set_facecolor(facecolor)
    rect.set_linewidth(linewidth)


# ----------------------------------------------------------------------

def set_font_size(obj, font_size=8):
    """
    set font size of axis or all axes of a figure

    Parameters
    ----------
    obj : matplotlib axis of figure object
        sets font size of the axis or all axes of figure
    font_size : int
        font size
    """

    axes = _get_ax_iterable(obj)

    for ax in axes:

        leg = []
        if ax.legend_ is not None:
            leg = ax.legend_.texts + [ax.legend_.get_title()]

        suptitle = []
        if ax.get_figure()._suptitle is not None:
            suptitle = [ax.get_figure()._suptitle]

        for item in ([ax.title, ax._left_title, ax._right_title] +
                     [ax.xaxis.label, ax.yaxis.label] +
                      ax.get_xticklabels() + ax.get_yticklabels() +
                      ax.xaxis.get_minorticklabels() + ax.yaxis.get_minorticklabels() + 
                      ax.texts + leg + suptitle):

            item.set_fontsize(font_size)


# ----------------------------------------------------------------------


def map_ylabel(s, x=-0.07, y=0.55, ax=None, **kwargs):
    """
    add ylabel to cartopy plot

    Parameters
    ----------
    s : string
        text to display
    x : float
        x position
    y : float
        y position
    ax : matplotlib axis
        axis to add the label
    **kwargs : keyword arguments
        see matplotlib text help

    Returns
    -------
    h : handle
        text handle of the created text field

    ..note::
    http://stackoverflow.com/questions/35479508/cartopy-set-xlabel-set-ylabel-not-ticklabels

    """
    if ax is None:
        ax = plt.gca()
    
    va = kwargs.pop('va', 'bottom')
    ha = kwargs.pop('ha', 'center')
    rotation = kwargs.pop('rotation', 'vertical')
    rotation_mode = kwargs.pop('rotation_mode', 'anchor')
    
    transform = kwargs.pop('transform', ax.transAxes)

    return ax.text(x, y, s, va=va, ha=ha, rotation=rotation, 
                   rotation_mode=rotation_mode,
                   transform=transform, **kwargs)

# ----------------------------------------------------------------------


def map_xlabel(s, x=-0.07, y=0.55, ax=None, **kwargs):
    """
    add xlabel to cartopy plot

    Parameters
    ----------
    s : string
        text to display
    x : float
        x position
    y : float
        y position
    ax : matplotlib axis
        axis to add the label
    **kwargs : keyword arguments
        see matplotlib text help

    Returns
    -------
    h : handle
        text handle of the created text field

    ..note::
    http://stackoverflow.com/questions/35479508/cartopy-set-xlabel-set-ylabel-not-ticklabels

    """
    if ax is None:
        ax = plt.gca()
    
    va = kwargs.pop('va', 'bottom')
    ha = kwargs.pop('ha', 'center')
    rotation = kwargs.pop('rotation', 'horizontal')
    rotation_mode = kwargs.pop('rotation_mode', 'anchor')
    
    transform = kwargs.pop('transform', ax.transAxes)

    return ax.text(x, y, s, va=va, ha=ha, rotation=rotation, 
                   rotation_mode=rotation_mode,
                   transform=transform, **kwargs)

# ----------------------------------------------------------------------


def _get_ax_iterable(obj):
    """ get iterable axes from matplotlib axes or figure
    """

    if isinstance(obj, plt.Figure):
        axes = obj.get_axes()
    else:
        axes = obj

    if not isinstance(axes, collections.Iterable):
        axes = [axes]

    return np.array(axes).flatten()


# ----------------------------------------------------------------------


def add_subplot_label(obj, x=-0.08, y=1.05, start=0, **kwargs):
    """
    add labels to axes, currently adds (a), (b), etc.

    Parameters
    ----------
    obj : matplotlib axis of figure object
        sets font size of the axis or all axes of figure
    x : float
        x position
    y : float
        y position
    """



    axes = _get_ax_iterable(obj)

    va = kwargs.pop('va', None)

    if va is None:
        va = kwargs.pop('verticalalignment', 'bottom')

    for n, ax in enumerate(axes, start):
        txt = '(' + string.ascii_lowercase[n] + ')'
        ax.text(x, y, txt, transform=ax.transAxes,
                va=va, **kwargs)

# ----------------------------------------------------------------------


def add_subplot_label_edge(obj, border=5, fontsize=8, start=0,
                           **kwargs):
    """
    add labels to axes, currently adds (a), (b), etc.

    Parameters
    ----------
    obj : matplotlib axis of figure object
        sets font size of the axis or all axes of figure
    border : float
        border around rectangle
    fontsize : float
        fontsize of label
    start : int
        with which letter to start (a == 0)

    ..note::
    The fontsize needs to be set here - changing it afterwards leads to 
    an offset.
    """

    axes = _get_ax_iterable(obj)

    # get arguments
    facecolor = kwargs.pop('facecolor', '0.95')
    edgecolor = kwargs.pop('edgecolor', '0.95')

    lw = 1.

    # construct arguments
    bbox = dict(facecolor=facecolor,
                edgecolor=edgecolor,
                pad=border,
                lw=lw)

    
    
    for n, ax in enumerate(axes, start):
        txt = '(' + string.ascii_lowercase[n] + ')'

        # border line and width of spines shifts rectangle by half their width
        left_margin = (lw + ax.spines['left'].get_linewidth()) / 2.
        top_margin = (lw + ax.spines['top'].get_linewidth()) / 2.

        # we need to offset the rectangle, such that it just touches
        # the spines
        offset = (border + left_margin, - border - top_margin)
        
        ax.annotate(txt, xy=(0, 1), xycoords='axes fraction',
                    xytext=offset, textcoords='offset points',
                    fontsize=fontsize,
                    bbox=bbox, 
                    ha='left', va='top')





def map_gridlines(ax, left=True, bottom=True, right=False, top=False, 
                  lat=None, lon=None, format=True):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='0.5', linestyle='-', alpha=0.5)

    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}



    gl.xlabels_top = top
    gl.xlabels_bottom = bottom
    gl.ylabels_left = left
    gl.ylabels_right = right

    if format:
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    if lon is not None:
        # if np.isscalar(lon):
        #     lon = np.arange(0, 360, lon)
        gl.xlocator = mticker.FixedLocator(lon)
    if lat is not None:
        # if np.isscalar(lat):
        #     lat = np.arange(-90, 91, lat)
        gl.ylocator = mticker.FixedLocator(lat)


# -----------------------------------------------------------------------------

def hatch(x, y, z, **kwargs):

    ax = kwargs.pop('ax', plt.gca())

    # extend grid
    x, y = np.meshgrid(x, y)

    # select relevant points
    x = x.ravel()[z.ravel()]
    y = y.ravel()[z.ravel()]
  
    ms = kwargs.pop('ms', 1)
    color = kwargs.pop('color', '0.25')


    ax.plot(x, y, '.', ms=ms, color=color, **kwargs)

# -----------------------------------------------------------------------------

def hatch_xarray(ds, x='lon', y='lat', **kwargs):

    transform = kwargs.pop('transform', ccrs.PlateCarree())

    lon = ds[x]
    lat = ds[y]

    hatch(lon, lat, ds.values, transform=transform, **kwargs)


# =============================================================================


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mplcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# -----------------------------------------------------------------------------
# OROGRAPHY Colormap

cmap_orog = truncate_colormap(plt.get_cmap('gist_earth'), 0.33)


# -----------------------------------------------------------------------------


class xlabel_at(object):
    """
    draw an xlabel at a arbitrary x position
    
    Example
    -------
    f, axes = plt.subplots()
    ax = axes
    ax.plot([0, 2], [0, 5])
    cid = f.canvas.mpl_connect('draw_event', xlabel_at(ax, 0.5, 'label'))
    ax.set_xlabel('real label')
    """

    def __init__(self, ax, xpos, text):
        """
 
        Parameters
        ----------
        ax: matplotlib axes
            The axes to add the xlabel.
        xpos : float
            The position of the center of the xlabel in data coordinates.
        text : string
            The displayed text.
    
        """

        super(xlabel_at, self).__init__()
        
        self.ax = ax
        self.xpos = xpos
        self.text = text        
        
        self.f = ax.get_figure()

        # convert xpos to axes fraction
        self.xpos_transAxes = self._xpos_to_transAxes()
        
        
        xpos = self.xpos_transAxes
        ypos = self._get_label_position_transAxes()
        
        # add the annotation
        self.t = self.ax.annotate(self.text, xy=(xpos, ypos),
                                  xycoords='axes fraction',
                                  annotation_clip=False,
                                  va='top', ha='center')

    def __call__(self, event=None):
        # is called when a draw_event is triggered
        self._move_textbox()
        
    def _xpos_to_transAxes(self):
        # convert transData to transAxes
        xpos = self.ax.transData.transform([self.xpos, 0])
        return self.ax.transAxes.inverted().transform(xpos)[0]
    
    def _get_label_position_transAxes(self):
        # get position of 'real' xlabel in Axes coordinate system
    
        # get the current coordinate system of text (can change)
        transLabel = self.ax.xaxis.label.get_transform()
        # get position
        pos = self.ax.xaxis.label.get_position()
        
        # convert to Axes coordinate system
        pos = transLabel.transform(pos)
        return self.ax.transAxes.inverted().transform(pos)[1]
    
    
    def _move_textbox(self):
        plt.draw()
        xpos = self.xpos_transAxes
        ypos = self._get_label_position_transAxes()
    
        self.t.set_transform(self.ax.transAxes)
        self.t.set_position((xpos, ypos))
        
        return False






