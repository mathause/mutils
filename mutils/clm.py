#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: Mathias Hauser
#Date: 


# CLM convinience functions


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

try:
    import seaborn as sns
    sns.set(style='whitegrid')
except:
    ImportError


from .plot import format_ax_months, legend, set_font_size

_legend = legend

# ======================================================================
# clm layers
# ======================================================================

nlevsoi = 10
nlevgrnd = 15

_i = np.arange(1, nlevgrnd + 1)

_fs = 0.025

# node depth
node = _fs * (np.exp(0.5*(_i - 0.5)) - 1)

# thickness
thick = np.ones_like(node)
thick[0] = 0.5 * (node[0] + node[1])
thick[1:-1] = 0.5 * (node[2:] - node[:-2])
thick[-1] = node[-1] - node[-2]

# depth
depth = np.ones_like(node)
depth[:-1] = 0.5 * (node[:-1] + node[1:])
depth[-1] = node[-1] + 0.5 * thick[-1]

depth_all = np.concatenate(([0], depth))

# ----------------------------------------------------------------------


def _plt_layers(ax, ylim, txt):

    x = np.ones_like(_i)

    # "water"
    ax.axhspan(0, -3.8, color='#a6cee3')
    
    # layer interface
    for n in depth:
        ax.axhline(-n, lw=1, color='0.05')

    # middle of layer (node)
    ax.plot(x, -node, '.', color='r')
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_ylabel('Depth [m]')


def _write_node_thick_depth(ax, idx):

    # clear axes
    sns.despine(ax=ax, bottom=True, left=True)
    ax.set_xticks([])
    ax.set_yticks([])

    # title line
    ax.text(0.05, 1, 'i', va='center', ha='right')
    ax.text(0.4, 1, 'Node', va='center', ha='right')
    ax.text(0.7, 1, 'Thick', va='center', ha='right')
    ax.text(1, 1, 'Depth*', va='center', ha='right')

    # numbers
    opt = dict(va='center', ha='right')
    msg = '{0:5.3f} m'
    for ii, n in enumerate(node[idx]):
        ax.text(0.05, -ii, '{}'.format(ii + 1), **opt)
        ax.text(0.40, -ii, msg.format(node[ii]), **opt)
        ax.text(0.70, -ii, msg.format(thick[ii]), **opt)
        ax.text(1.00, -ii, msg.format(depth[ii]), **opt)

    ax.set_ylim(-8, 2)


def plot_clm_layers():

    x = np.ones_like(_i)

    gs = gridspec.GridSpec(2, 2)

    # plot all layers
    ax = plt.subplot(gs[0])
    _plt_layers(ax, [-45, 0], 10)

    # plot layers down to 5 m
    ax = plt.subplot(gs[1])
    _plt_layers(ax, [-5, 0], 10)

    # write first half of Node / Thick / Depth
    ax = plt.subplot(gs[2])
    _write_node_thick_depth(ax, range(8))

    # write second half of Node / Thick / Depth
    ax = plt.subplot(gs[3])
    _write_node_thick_depth(ax, range(8, 15))

    ax.text(0.05, -7, '*Layer interface', va='center', ha='left')

# ======================================================================
# root fraction per pft
# ======================================================================


def get_root_fraction(ra, rb):

    rf = np.ones(nlevsoi)

    rf[:nlevsoi] = 0.5 * (np.exp(-ra * depth_all[:nlevsoi])
                          + np.exp(-rb * depth_all[:nlevsoi])
                          - np.exp(-ra * depth_all[1:nlevsoi + 1])
                          - np.exp(-rb * depth_all[1:nlevsoi + 1]))

    rf[nlevsoi - 1] = 0.5 * (np.exp(-ra * depth_all[nlevsoi - 1])
                             + np.exp(-rb * depth_all[nlevsoi - 1]))

    return rf


def get_root_fraction_pft():
    rf1 = get_root_fraction(7, 2)
    rf2 = get_root_fraction(7, 1)
    rf3 = get_root_fraction(6, 2)
    rf4 = get_root_fraction(7, 1.5)
    rf5 = get_root_fraction(11, 2)
    rf6 = get_root_fraction(6, 3)

    return rf1, rf2, rf3, rf4, rf5, rf6

# ======================================================================
# interpolation of SM data
# ======================================================================


def _get_doy():
    # define constants

    # use any non-leap year
    date_range = pd.date_range('2001-01-01', '2001-12-31')

    # month for every doy
    month = date_range.month
    # day of month for every doy
    dayofmonth = date_range.day

    # days per month
    ndaypm = np.array((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))

    # number of days per year
    ndaypy = 365

    doy = np.arange(1, 366) + 365

    # doy in the middle of the month
    _a = pd.date_range('2001-01-01', '2001-12-31', freq='M').dayofyear
    _b = pd.date_range('2001-01-01', '2001-12-31', freq='MS').dayofyear
    doy_month = (_a - 1 + _b) / 2. + 365

    return date_range, month, dayofmonth, ndaypm, doy, doy_month, ndaypy

# ----------------------------------------------------------------------


def sm_weights_monthly():
    """
    daily weights to interpolate monthly sm data as done in CLM

    Returns
    =======
    TS1, TS2 : array of int
        month to use for interpolation
    tw1, tw2 : array of float
        corresponding weights

    Usage
    =====
    see example

    ..note::
      TS1 and TS2 are 1..12, to select from a python array, subtract 1

    """

    __, month, dayofmonth, ndaypm, __, __, __ = _get_doy()

    # fractional day of month (e.g. 1. Jan = 1./31)
    t = (dayofmonth - 0.5) / ndaypm[month - 1]

    # are we in the first or second half of the month
    it1 = np.floor(t + 0.5)
    it2 = it1 + 1

    # which months do we need?
    TS1 = (month + it1 - 1).astype(np.int)
    TS2 = (month + it2 - 1).astype(np.int)

    # restrict months to 1..12
    TS1[TS1 == 0] = 12
    TS2[TS2 == 13] = 1

    # calculate the weight
    tw1 = (it1 + 0.5) - t
    tw2 = 1 - tw1

    return TS1, TS2, tw1, tw2


# ----------------------------------------------------------------------

def plot_sm_weights_monthly():
    # plot the weights and corresponding months

    TS1, TS2, tw1, tw2 = sm_weights_monthly()

    doy = np.arange(366, 366 + 365)

    f, axes = plt.subplots(2, 1)

    ax = axes[0]
    ax.plot(doy, tw1, '.', label="Weight 1")
    ax.plot(doy, tw2, '.', label="Weight 2")
    ax.set_ylabel('Weight')

    legend(ax=ax, ncol=2)

    ax = axes[1]
    ax.plot(doy, TS1, '.', label="Month of Weight 1")
    ax.plot(doy, TS2, '.', label="Month of Weight 2")
    ax.set_ylim(-1, 13)
    ax.set_ylabel('Month')
    
    legend(ax=ax, ncol=2, loc='best')
    format_ax_months(axes[0])
    format_ax_months(axes[1])
    return axes


# ----------------------------------------------------------------------

def _example_daily_sm():
    # construct an example SM dataset

    __, __, __, __, doy, _, ndaypy = _get_doy()

    # low amplitude sin curve
    SM_daily = 0.2 * np.sin(doy * ((np.pi * 2) / (ndaypy))) + 1.5

    # add a dry period in summer
    idx = slice(150, 200)
    SM_daily[idx] -= 0.4 * np.sin(np.arange(50) * ((np.pi * 2) / (50 * 2)))

    # noise
    np.random.seed(123)
    SM_daily += (np.random.randn(ndaypy) * 0.01)

    return SM_daily

# ----------------------------------------------------------------------


def example_sm_weights_monthly(SM_daily=_example_daily_sm(), ax=None, 
                               legend=True):
    # show interpolation for an example sm dataset

    date_range, month, dayofmonth, ndaypm, doy, doy_month, ndaypy = _get_doy()

    # get monthly means
    SM_monthly = pd.Series(SM_daily, date_range).resample('M').mean().values

    # get interpolation function
    TS1, TS2, tw1, tw2 = sm_weights_monthly()
    # calculate the interpolated daily values
    SM_monthly_interp1 = tw1 * SM_monthly[TS1 - 1]
    SM_monthly_interp2 = tw2 * SM_monthly[TS2 - 1]

    SM_monthly_interp = SM_monthly_interp1 + SM_monthly_interp2

    # calculate monthly means again (from interpolated data)
    SM_monthly_interp_monthly = pd.Series(SM_monthly_interp, 
                                          date_range).resample('M').mean().values

    # start plotting
    if ax is None:
        f, ax = plt.subplots(1, 1)

    ax.step(doy, SM_daily, label='Daily', zorder=2, color='#1f78b4')
    
    ax.plot(doy_month, SM_monthly, 'o', ms=6, label='Monthly', color='#1f78b4', 
            zorder=3, markeredgecolor='0.05', markeredgewidth=0.5)

    ax.step(doy, SM_monthly_interp, label='Monthly, Interpolated',
            zorder=1, color='#ff7f00')

    ax.plot(doy_month, SM_monthly_interp_monthly, 'o', ms=6,
            label='Monthly of Interpolated', color='#ff7f00',
            markeredgecolor='0.05', markeredgewidth=0.5)

    RMSE = np.sqrt(np.mean((SM_monthly_interp - SM_daily)**2))
    msg = "RMSE: {rmse:5.2f} mm".format(rmse=np.asscalar(RMSE))

    ax.set_title(msg, loc='right')
    ax.set_ylabel('Soil Moisture [mm]')
    
    if legend:
        _legend(ax=ax, ncol=2, loc='lower left')
    format_ax_months(ax=ax)
    set_font_size(ax, 8)
    
    return ax
