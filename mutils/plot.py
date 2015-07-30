#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: Mathias Hauser
#Date:

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as dates





def get_doy(dt=1):
    """
    
    """
    begday = dates.date2num(datetime(2, 1, 1))
    endday = dates.date2num(datetime(2, 12, 31))
    days = np.arange(begday, endday + 1, dt)
    return days




def format_ax_months(ax, xlim=(366, 365*2)):

    # format x axis
    hfmt = dates.DateFormatter('%b')
    ax.xaxis.set_major_locator(dates.DayLocator(1))
    ax.xaxis.set_minor_locator(dates.DayLocator(15))
    ax.xaxis.set_minor_formatter(hfmt)

    ax.tick_params(which='minor', length=0)
    
    ax.set_xticklabels([])

    ax.set_xlim(xlim)














