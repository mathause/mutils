#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: Mathias Hauser
#Date: 


import cartopy.crs as ccrs

def pcolormesh(ax, data, lat, lon, trans=ccrs.PlateCarree(), **kwargs):
    """
    plot grid on geoAxes

    Parameters
    ----------
    ax : geoAxes
        GeoAxes instance of cartopy
    da : Dataset
        2D xray Dataset to plot
    dlat, dlon : float
        half the lat/ lon grid spacing to center the gridcells
    trans : cartopy.crs instance
        transformation information

    Returns
    -------
    h : handle
        pcolormesh handle

    ..todo::
      infer dlat and dlon from the da data
    """

    lons, lats = np.meshgrid(lon, lat)

    data = np.ma.masked_invalid(data)
    h = ax.pcolormesh(lons, lats, data, transform=trans, **kwargs)
    
    return h













