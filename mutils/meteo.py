#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: Mathias Hauser
#Date: 07.2015

import numpy as np
from numpy import deg2rad as _d2r

def rgpot_daily_mean(lats, solar_constant=1366.):
    """
    potential radiation at earth surface in absence of atmosphere

    Parameters
    ----------
    lat : float
        latitude of location
    solar_constant : float
        solar constant

    Returns
    -------
    out : ndarray
        n_lats x 365 days

    Notes
    -----
    Algorithm as in Guillod et al.
    time is in UTC
    this is longitude-independent

    """

    lats = np.asarray(lats)

    out = np.zeros(shape=(len(lats), 365))

    lon = 0
    for i, lat in enumerate(lats):
        out[i, :] = rgpot(lat, lon, solar_constant).mean(axis=0)

    return out

def rgpot(lat, lon, solar_constant=1366.):
    """
    potential radiation at earth surface in absence of atmosphere

    Parameters
    ----------
    lat : float
        latitude of location
    lon : float
        longitude of location
    solar_constant : float
        solar constant

    Returns
    -------
    rgpot : ndarray




    Notes
    -----
    Algorithm as in Guillod et al.
    time is in UTC

    """


    # day of the year
    doy = np.arange(1, 366)
    
    # declination after wikipedia 
    # https://en.wikipedia.org/wiki/Position_of_the_Sun
    dec = np.arcsin(np.sin(_d2r(-23.44)) * np.cos(_d2r(360/365.24 * (doy + 10) + 
                 360/np.pi*0.0167*np.sin(_d2r(360/365.24*(doy-2))))))

    # time of the day
    time_of_day = np.arange(0, 24, 1./60)

    # in -pi...pi
    hour_angle = (time_of_day + lon*24/360-12)*np.pi/12

    # create a grid
    # days x timesteps_per_day
    dec, hour_angle = np.meshgrid(dec, hour_angle)


    p1 = np.sin(_d2r(lat)) * np.sin(dec)
    p2 = np.cos(_d2r(lat))*np.cos(dec)*np.cos(hour_angle)

    cos_z = p1 + p2

    # don't return negative insolation
    return solar_constant * np.fmax(cos_z, 0)






def NormCosWgt(lat):
    """Returns cosine-weighted latitude"""
    return np.cos(_d2r(lat))


def get_csi(LWdown, T, e, e_ad=0.22, k=0.47, return_all=False):
    """
    Clear Sky Index after Marty and Philipona, 2000

    Parameters
    ----------
    LWdown : ndarray
        longwave down as measured/ simulated [W/m**2]
    T : ndarray
        temperature at lowest level [K]
    e : ndarray
        water vapor pressure [Pa]
    e_ad : float
        altitude dependent clear-sky emittance of a completely dry atmosphere
    k : float
        location dependent coefficient
    return_all : bool
        if False returns CSI, else returns (CSI, app_em, clear_sky_app_em)
    
    Returns
    -------
    CSI : ndarray
        clear sky index = 


    Notes
    -----
    According to Marty and Philipona
    CSI <= 1 : clear sky, no clouds
    CSI > 1 : cloudy sky, overcast
    """

    # stephan bolzman constant
    sigma = 5.67 * 10**-8

    # apparent emittance of actual sky
    app_em = LWdown / (sigma * T**4)

    # clear sky apparent emmitance
    cs_app_em = e_ad + k * (e * T) ** (1./7)

    CSI = app_em / cs_app_em

    if return_all:
        return (CSI, app_em, cs_app_em)
    else:
        return CSI
