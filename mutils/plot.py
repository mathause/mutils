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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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
                 projection=ccrs.PlateCarree(), coastlines=False, 
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

    Returns
    -------
    legend : handle
        Legend handle.
    """

    ax = kwargs.pop('ax', plt.gca())

    facecolor = kwargs.pop('facecolor', '0.95')
    linewidth = kwargs.pop('linewidth', 0.0)

    kwargs['frameon'] = kwargs.pop('frameon', True)

    legend = ax.legend(*args, **kwargs)
    rect = legend.get_frame()

    rect.set_facecolor(facecolor)
    rect.set_linewidth(linewidth)

    return legend

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

# =============================================================================


def map_ticks(lon, lat, ax=None, left=True, right=False, bottom=True, top=False,
              fontsize=None, direction='in'):
    """
    convinience for map ticks
    
    Parameters
    ----------
    lon : 1d-array
        Locations of the longitude ticks & labels.
    lat : 1d-array
        Locations of the latitude ticks & labels.    
    ax : GeoAxes object, optional
        Axes to add the ticks to. Uses current axes if none are given.
    left : bool, optional.
        If True adds ticklabels on the left side. See note. Default: True.
    right : bool, optional.
        If True adds ticklabels on the right side. See note. Default: False.
    bottom : bool, optional.
        If True adds ticklabels on the bottom. See note. Default: True.
    top : bool, optional.
        If True adds ticklabels on the left side. See note. Default: False.
    fontsize : int, optional
        If given set fontsize. Default: None.
    direction: 'in' | 'out' | 'inout', optional
        The direction of the ticks. Default: 'in'.
    
    ..note::

    * It's only possible to have one of left/ right, so if both
      are True, use the one that is not the default (i.e. right).
      Dito for bottom/ top. If you want to have a label for both use
      map_gridlines.
    
    * You cannot set the zorder of ticks (yet?)! So don't be surprised
      if they are not shown... 


    Mathias Hauser    
    """    
    
    
    if ax is None:
        ax = plt.gca()

    # we need to restrict lat and lon to the limits of the plot - 
    # else the limits are changed
    
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()

    # lon = lon[(lon > xlim[0]) & (lon < xlim[1])]
    # lat = lat[(lat > ylim[0]) & (lat < ylim[1])]

    extent = ax.get_extent(ccrs.PlateCarree())
    xmin, xmax, ymin, ymax = extent

    # lon = lon[(lon >= xmin) & (lon <= xmax)]
    # lat = lat[(lat >= ymin) & (lat <= ymax)]

    ax.set_xticks(lon, crs=ccrs.PlateCarree())
    ax.set_yticks(lat, crs=ccrs.PlateCarree())

    # the LONGITUDE_FORMATTER and LATITUDE_FORMATTER are actually meant for gridlines
    # however, they make better strings than LongitudeFormatter and LatitudeFormatter
    
    lon_formatter = LONGITUDE_FORMATTER
    lat_formatter = LATITUDE_FORMATTER
    
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # get rid of 'lat' and 'lon': relevant for xarray
    ax.set_xlabel('')
    ax.set_ylabel('')

    # it's only possible to have one of left/ right, top/ bottom
    # so if both are True, use the one that is not the default

    if right:
        ax.yaxis.tick_right()
    elif left:
        ax.yaxis.tick_left()
    else:
        ax.set_yticklabels('')
        
    if bottom:
        ax.xaxis.tick_bottom()
    elif top:
        ax.xaxis.tick_top()
    else:
        ax.set_xticklabels('')
        
        
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    # well this is just my preference
    ax.tick_params('both', direction=direction)

    if fontsize:
        plt.setp(ax.get_xticklabels(), 'fontsize', fontsize)
        plt.setp(ax.get_yticklabels(), 'fontsize', fontsize)

# =============================================================================


def map_gridlines(ax, left=True, bottom=True, right=False, top=False, 
                  lat=None, lon=None, format=True, fontsize=8,
                  lines=None, xlines=True, ylines=True,
                  gl_kws=dict(draw_labels=True,
                  linewidth=0.5, color='0.5', linestyle='-', alpha=0.5)):

    """
    add map gridlines

    Parameters
    ----------
    ax : axes object
        axes object to add gridlines to
    left : bool, optional
        If True, adds labels to the left of the plot. Default: True.
    bottom : bool, optional
        If True, adds labels to the bottom of the plot. Default: True.
    right : bool, optional
        If True, adds labels to the right of the plot. Default: False.
    top : bool, optional
        If True, adds labels to the top of the plot. Default: False.
    lat : ndarray or None, optional
        Position of latitude gridlines/ ticks. If None uses automatic
        positions. Default: None
    lon : ndarray or None, optional
        Position of longitude gridlines/ ticks. If None uses automatic
        positions. Default: None
    format : bool
        If True formats lat and lon ticks (10°N, ...), using 
        LONGITUDE_FORMATTER, LATITUDE_FORMATTER. Default: True.
    fontsize: int or None, optional
        If given sets the fontsize of the labels, if None, uses rcParam.
        Default: 8.
    lines : None, bool , optional
        If True, adds x and y gridlines, if False not. If None uses
        the values of xlines and ylines. Default: None.
    xlines : bool
        If True, adds x gridlines, if False not. Default: True.
    ylines : bool
        If True, adds y gridlines, if False not. Default: True.
    gl_kws : dict, optional
        Style of the gridlines. See function spec for defaults.


    ..note: 
    * It possible to add labels to the left *and* to the right.


    """



    if lines is not None:
        xlines = ylines = lines

    gl = ax.gridlines(crs=ccrs.PlateCarree(), **gl_kws)

    if fontsize is not None:
        gl.xlabel_style = {'size': fontsize}
        gl.ylabel_style = {'size': fontsize}

    gl.xlines = xlines
    gl.ylines = ylines

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

    return gl


# -----------------------------------------------------------------------------

def hatch(x, y, z, **kwargs):

    ax = kwargs.pop('ax', plt.gca())

    # extend grid
    x, y = np.meshgrid(x, y)

    thin = kwargs.pop('thin', False)

    if thin:
        x = x[::2, ::2]
        y = y[::2, ::2]
        z = z[::2, ::2]

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


def get_map_layout(nrows, ncols, aspect, cbar_space=2.5, cbar_vertical=True, 
                   width=17., projection=ccrs.PlateCarree(), coastlines=False,
                        **kwargs):
    """
    geo-subplots with good height of figure, given the layout and aspect 

    Parameters
    ----------
    nrows : int
        Number of rows of the subplot grid.
    ncols : int
        Number of columns of the subplot grid.
    aspect : float
        Aspect ratio of the map plot. As a first guess, use
         "len(lon) / lat(lat)". (Careful: integer division.)
    cbar_space : float, optional
        Space that the colorbar will take in cm. Default: 2.5 cm. 
        Needs a solution, when the colorbar is horizontal.
    cbar_vertical : bool, optional
        If True, assumes the colorbar is vertical, if False that it is 
        horizontal. Default: True.
    width : float, optional
        Width of the figure in cm. Default: 17 cm.
    projection : cartopy projection, optional
        Default: cartopy.crs.PlateCarree()
    coastlines : bool, optional
        If True adds coarse resolution coastlines. Default: False.

    Returns
    -------
    f : figure
        Figure handle.
    axes : axes
        Axes handle.

    """


    f, axes = get_subplots(nrows, ncols)
    
    width_inch, height_inch = _get_map_layout(nrows, ncols, aspect, cbar_space,
                                              cbar_vertical, width)
    
    f.set_figwidth(width_inch)
    f.set_figheight(height_inch)

    return f, axes



def _get_map_layout(nrows, ncols, aspect, cbar_space=2.5, cbar_vertical=True, 
                    width=17.):



    width_inch = width / 2.54
    cbar_space_inch = cbar_space / 2.54

    if cbar_vertical:
        width_of_subplot = (width_inch - cbar_space_inch) / ncols
        height_of_subplot = width_of_subplot / aspect
        height_inch = nrows * height_of_subplot
    else:
        width_of_subplot = width_inch / ncols
        height_of_subplot = width_of_subplot / aspect
        height_inch = nrows * height_of_subplot + cbar_space_inch


    return width_inch, height_inch

# -----------------------------------------------------------------------------


def set_map_layout_old(axes, cbar_space=2.5, cbar_vertical=True, width=17.0):
    """
    set figure height, given width and colorbar setting

    Needs to be called after all plotting is done.
       
    Parameters
    ----------
    axes : ndarray of (Geo)Axes
        Array with all axes of the figure.
    cbar_space: float, optional
        Width or height of the colorbar in cm. Default: 2.5
    cbar_vertical : bool, optional
        If True, assumes the colorbar is vertical, else assumes it's
        horizontal. Default: True
    width : float
        Width of the full figure in cm. Default 17

    ..note: currently only works if all the axes have the same aspect
    ratio.
    """

    # calculate the width that is available for plots
    if cbar_vertical:
        width_plots = width - cbar_space
    else:
        width_plots = width
    
    if isinstance(axes, plt.Axes):
        ax = axes
    else:
        # assumes the first of the axes is representative for all
        ax = axes.flat[0]
    
    f = ax.get_figure()
    
    # data ratio is the aspect
    aspect = ax.get_data_ratio()
    print(aspect)

    # get geometry tells how many subplots there are
    n_rows, n_cols, __ = ax.get_geometry()
    print(n_rows, n_cols)
    
    width_one_plot = width_plots / n_cols
    
    height_plots = width_one_plot * n_rows * aspect
    
    if cbar_vertical:
        height = height_plots
    else:
        height = height_plots + cbar_space
    
    print(width_one_plot, height, width)
    
    f.set_figwidth(width / 2.54)
    f.set_figheight(height / 2.54)

# -----------------------------------------------------------------------------


def set_map_layout(axes, width=17.0, cbar_space=None, cbar_vertical=None):
    """
    set figure height, given width and colorbar setting

    Needs to be called after all plotting is done.
       
    Parameters
    ----------
    axes : ndarray of (Geo)Axes
        Array with all axes of the figure.
    width : float
        Width of the full figure in cm. Default 17

    ..note: currently only works if all the axes have the same aspect
    ratio.
    """

    if (cbar_space is not None) or (cbar_vertical is not None):
        msg = ("Warning: 'cbar_space' and 'cbar_vertical' is "
               "currently ignored. Ussing 'set_map_layout_old'.")
        print(msg)

        set_map_layout_old(axes, cbar_space=cbar_space,
                           cbar_vertical=cbar_vertical, width=width)
        return 

    if isinstance(axes, plt.Axes):
        ax = axes
    else:
        # assumes the first of the axes is representative for all
        ax = axes.flat[0]
    
    # read figure data
    f = ax.get_figure()

    bottom = f.subplotpars.bottom
    top = f.subplotpars.top
    left = f.subplotpars.left
    right = f.subplotpars.right
    hspace = f.subplotpars.hspace
    wspace = f.subplotpars.wspace

    # data ratio is the aspect
    aspect = ax.get_data_ratio()
    # get geometry tells how many subplots there are
    nrow, ncol, __ = ax.get_geometry()


    # width of one plot, taking into account
    # left * wf, (1-right) * wf, ncol * wp, (1-ncol) * wp * wspace
    wp = (width - width * (left + (1-right))) / (ncol + (ncol-1) * wspace) 

    hp = wp * aspect

    # height of figure
    height = (hp * (nrow + ((nrow - 1) * hspace))) / (1. - (bottom + (1 - top)))


    f.set_figwidth(width / 2.54)
    f.set_figheight(height / 2.54)

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






