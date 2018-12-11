#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Hauser
# Date: 07.2015

from functools import wraps
import numpy as np
import warnings

# definitions
# Abbr.  | Name                            | Unit
# -------+---------------------------------+-----------
# e      | partial pressure                | [Pa]
# rho_v  | absolute humidity/ mass density | [kg/m**3]
# w_v    | water vapor mixing ratio*       | [kg/kg]
# q_v    | specific humidity**             | [kg/kg]
# m_v    | mass of water vapor             | [kg]
# m_d    | mass of dry air                 | [kg]
# m      | mass of moist air               | [kg]

#  * m_v / m_d
# ** m_v / m

# CONSTANTS
# specific gas constant of water vapor (R*/M_w)
R_v = 461.5  # J kg**-1 K**-1

# molecular weight of water
M_w = 18  # g mole**-1

# Mw/Md (molar masses of air)
epsilon = 0.622

# specific heat of air at constant pressure
c_p_air = 1.012 # J / (g * K) (room conditions)

# latent heat of vaporization
latent_heat = 2.26 # MJ / kg 

# =============================================================================

def _check_T_bound(T):
    # T = np.asarray(T)
    if np.any(T <= 150.):
        raise ValueError('Temperature must be in K')
    return T

# =============================================================================

def psychrometric_const(P=102.425):
    """

    Parameters
    ----------
    p : float or ndarray
        Atmospheric pressure in kPa.


    Returns
    -------
    psi : ndarray
        psychrometric const [kPa / °C]


    Units
    -----
    c_p_air:        kJ / (kg K)
    P:              kPa
    latent_heat:    MJ / kg
    epsilon:        1

    """

    print('Warning: psi was corrected is now * 10 ** 3 (was * 10**-3)')
    return c_p_air * 10**-3 * P / (latent_heat * epsilon)
    

def virtual_temp(T, q_v):
    """
    temperature dry air would need to have the same density as moist air

    Parameters
    ----------
    T : ndarray
        temperature [K]
    q_v : ndarray

    Returns
    -------
    T_v : ndarray
        virtual temperature
    """
    T = _check_T_bound(T)
    return T * (1 + 0.608 * q_v)

# =============================================================================


def e_2_rho_v(e, T):
    """
    absolute humidity to partial pressure

    Parameters
    ----------
    e : ndarray
        partial pressure of water vapor[]
    T : ndarray
        temperature [K]

    Returns
    -------
    rho_v : ndarray
        absolute humidity [kg m**-3]
    """
    T = _check_T_bound(T)
    return e / (R_v * T)

# =============================================================================


def rho_v_2_e(rho_v, T):
    """
    absolute humidity to partial pressure

    Parameters
    ----------
    rho_v : ndarray
        absolute humidity [kg m**-3]
    T : ndarray
        temperature

    Returns
    -------
    e : ndarray
        partial pressure of water vapor [Pa]
    """
    T = _check_T_bound(T)
    return R_v * rho_v * T

# =============================================================================


def e_2_q_v(e, p):
    """
    convert water vapor partial pressure to specific humidity

    Parameters
    ----------
    e : ndarray
         water vapor partial pressure [Pa]
    p : ndarray
        atmospheric pressure [Pa]

    Returns
    -------
    q_v : ndarray
        specific humidity [kg/kg]

    ..note::
      Approximation, exact formula is:
      :math: q_v = frac{epsilon*e}{p - e + epsilon * e}
    """
    return epsilon * e / p

# =============================================================================


def q_v_2_e(q_v, p):
    """
    convert specific humidity to water vapor partial pressure

    Parameters
    ----------
    q_v : ndarray
         specific humidity [kg/kg]
    p : ndarray
        atmospheric pressure [Pa]

    Returns
    -------
    e : ndarray
        partial water vapor pressure [Pa]

    ..note::
    Approximation, exact formula is (solved for e):
    :math: q_v = frac{epsilon*e}{p - e + epsilon * e}
    """
    return p * q_v / epsilon

# =============================================================================


def e_2_w_v(e, p):
    """
    convert water vapor partial pressure to water vapor mixing ratio

    Parameters
    ----------
    e : ndarray
         water vapor partial pressure [Pa]
    p : ndarray
        atmospheric pressure [Pa]

    Returns
    -------
    w_v : ndarray
        water vapor mixing ratio [kg/kg]

    ..note::
    Approximation, exact formula is:
    :math: w_v = frac{epsilon*e}{p - e}
    """
    return epsilon * e / p

# =============================================================================


def w_v_2_e(w_v, p):
    """
    convert water vapor mixing ratio to water vapor partial pressure

    Parameters
    ----------
    w_v : ndarray
         water vapor mixing ratio [kg/kg]
    p : ndarray
        pressure [Pa]

    Returns
    -------
    e : ndarray
        water vapor partial pressure [Pa]

    ..note::
    Approximation, exact formula is (solved for e):
   :math: w_v = frac{epsilon*e}{p - e}
    """
    return p * w_v / epsilon

# =============================================================================


def e_2_rel_hum(e, T, freezing_point=273.15):
    """
    converts partial pressure of water vapor to relative humidity

    relative humidity: ratio of the partial pressure of water vapor to the 
    equilibrium vapor pressure of water at the same temperature

    Parameters
    ----------
    e : ndarray
        partial pressure of water vapor [Pa]
    T : ndarray
        temperature [K]
    freezing_point : float
        treats temperatures smaller freezing_point as ice and larger as water, 
        needed for the saturation pressure

    Returns
    -------
    RH : ndarray
        relative humidity [-]
    """
    e_s = saturation_vapor_pressure(T, freezing_point)
    return e / e_s

# =============================================================================


def rel_hum_2_e(RH, T, freezing_point=273.15):
    """
    converts relative humidity to partial pressure of water vapor

    relative humidity: ratio of the partial pressure of water vapor to the 
    equilibrium vapor pressure of water at the same temperature

    Parameters
    ----------
    RH : ndarray
        relative humidity [-]
    T : ndarray
        temperature [K]
    freezing_point : float
        treats temperatures smaller freezing_point as ice and larger as water, 
        needed for the saturation pressure

    Returns
    -------
    e : ndarray
        partial pressure of water vapor [Pa]
    """
    e_s = saturation_vapor_pressure(T, freezing_point)
    return RH * e_s

# =============================================================================


def slope_saturation_vapour_pressure(T):
    """
    slope of saturation vapour pressure

    Parameters
    ----------
    T : ndarray
        temperture [K]

    Returns
    -------
    s : ndarray
        slope of the relationship between saturation vapour pressure and air
        temperature [kPa °C-1]

    Note
    ----
    http://agsys.cra-cin.it/tools/evapotranspiration/help/Slope_of_saturation_vapour_pressure_curve.html
    Tetens, 1930; Murray, 1967
    """

    T = _check_T_bound(T)

    # the formula is in °C
    T_C = T.copy() - 273.15

    dividend = 4098 * (0.6108 * np.exp((17.27 * T_C) / (T_C + 237.3) ))
    divisor = (T_C + 237.3) ** 2

    s = dividend / divisor

    return s

# =============================================================================


def saturation_vapor_pressure(T, freezing_point=273.15):
    """
    saturation vapor pressure after Murphy & Koop (2005)

    Parameters
    ----------
    T : ndarray
        temperature [K]
    freezing_point : float
        treats temperatures smaller freezing_point as ice and larger as water

    Returns
    -------
    e_s : ndarray
        saturation vapor pressure over water/ ice
    """
    T = _check_T_bound(T)

    e_s = np.zeros_like(T, dtype=np.float64)

    sel = T < freezing_point

    e_s[sel] = _mk_sat_vap_p_ice(T[sel])
    e_s[~sel] = _mk_sat_vap_p_water(T[~sel])

    return e_s

# =============================================================================


def _mk_sat_vap_p_ice(T):
    """
    saturation vapor pressure over ice after Murphy & Koop (2005)

    Parameters
    ----------
    T : ndarray
        temperature [K]

    Returns
    -------
    e_s_i : ndarray
        saturation vapor pressure over (solid) water
    """
    T = _check_T_bound(T)

    if np.any(T < 110) or np.any(T > 273.2):
        warnings.warn('Temperature outside valid range (110 K to 273.2 K)')

    e_s_i = np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) -
                   0.00728332 * T)

    return e_s_i

# =============================================================================

def _mk_sat_vap_p_water(T):
    """
    saturation vapor pressure over water after Murphy & Koop (2005)

    Parameters
    ----------
    T : ndarray
        temperature [K]

    Returns
    -------
    e_s_w : ndarray
        saturation vapor pressure over (liquid) water
    """
    T = _check_T_bound(T)

    if np.any(T < 123) or np.any(T > 332):
        warnings.warn('Temperature outside valid range (123 K to 332 K)')

    e_s_w = np.exp(54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T +
                   np.tanh(0.0415 * (T - 218.8)) *
                   (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025*T))

    return e_s_w


# def decorator_argument(name, min=0):
#     print name, min
#     def real_decorator(func):
#         def wrapper(*args, **kwargs):
#             print locals()
#             for arg in kwargs:
#                 print arg
#             for kwarg in kwargs:
#                 print kwarg
#             func(*args, **kwargs)
#         return wrapper
#     return real_decorator
