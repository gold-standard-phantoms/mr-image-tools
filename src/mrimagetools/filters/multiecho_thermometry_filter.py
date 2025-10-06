"""Multiecho Thermometry Filter

Functional implementation of fitting of multiecho magnitude data to a dual-resonance model
for MR thermometry applications.

Based on Sprinkhuizen, S.M., Bakker, C.J.G. and Bartels, L.W. (2010),
Absolute MR thermometry using time-domain analysis of multi-gradient-echo magnitude images.
Magn. Reson. Med., 64: 239-248. https://doi.org/10.1002/mrm.22429
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from mrimagetools.filters.adc_quantification_filter import calculate_r_squared
from mrimagetools.v2.containers.image import BaseImageContainer

GAMMA_H = 42.57747892e6  # Hz/T


def thermometry_signal_model(
    t: NDArray[np.floating],
    amplitude_1: float,
    amplitude_2: float,
    r2star_1: float,
    r2star_2: float,
    df: float,
    dphi_deg: float,
) -> NDArray[np.floating]:
    """Calculates the signal S(t) at time t according to the dual-resonance model.

    .. math::
        S(t) = \\sqrt{A_1^2e^{-2R_{2,1}^*t} + A_2^2e^{-2R_{2,2}^*t} + 2A_1A_2e^{-(R_{2,1}^* + R_{2,2}^*)t} \\cos(2\\pi\\Delta f_{12}t + \\Delta\\phi_{12})}

    :param t: The time vector.
    :type t: NDArray[np.floating]
    :param amplitude_1: The amplitude of the first signal component.
    :type amplitude_1: float
    :param amplitude_2: The amplitude of the second signal component.
    :type amplitude_2: float
    :param r2star_1: The R2* value of the first signal component.
    :type r2star_1: float
    :param r2star_2: The R2* value of the second signal component.
    :type r2star_2: float
    :param df: The frequency offset between the two signal components.
    :type df: float
    :param dphi_deg: The phase offset between the two signal components.
    :type dphi_deg: float
    :return: The signal S(t) at time t according to the dual-resonance model.
    :rtype: NDArray[np.floating]
    """
    dphi_rad = np.deg2rad(dphi_deg)
    radicand: NDArray[np.float64] = (
        amplitude_1**2 * np.exp(-2 * r2star_1 * t)
        + amplitude_2**2 * np.exp(-2 * r2star_2 * t)
        + 2
        * amplitude_1
        * amplitude_2
        * np.exp(-(r2star_1 + r2star_2) * t)
        * np.cos(2 * np.pi * df * t + dphi_rad)
    )
    radicand[radicand < 0] = 0  # Prevent negative values under the square root
    return np.sqrt(radicand)


def calculate_df_from_temperature(
    temperature_celsius: float, magnetic_field_tesla: float
) -> float:
    """Calculates frequency difference (df) from the temperature in Celsius and magnetic field strength in Tesla.

    .. math::
        T[°C] = 193.35 - 1.02 \\cdot 10^8 \\cdot \\frac{\\Delta f_{hm}[\\text{Hz}]}{\\gamma B_0}

    :param temperature_celsius: The temperature in Celsius.
    :type temperature_celsius: float
    :param magnetic_field_tesla: The magnetic field strength in Tesla.
    :type magnetic_field_tesla: float
    :return: The frequency difference (df) in Hz.
    :rtype: float
    """
    return ((193.35 - temperature_celsius) * GAMMA_H * magnetic_field_tesla) / 1.02e8


def calculate_temperature_from_df(df: float, magnetic_field_tesla: float) -> float:
    """Calculates temperature in Celsius from frequency difference (df) and magnetic field strength in Tesla.

    .. math::
        T[°C] = 193.35 - 1.02 \\cdot 10^8 \\cdot \\frac{\\Delta f_{hm}[\\text{Hz}]}{\\gamma B_0}

    :param df: The frequency difference (df) in Hz.
    :type df: float
    :param magnetic_field_tesla: The magnetic field strength in Tesla.
    :type magnetic_field_tesla: float
    :return: The temperature in Celsius.
    :rtype: float
    """
    return 193.35 - (1.02e8 * abs(df)) / (GAMMA_H * magnetic_field_tesla)


def calculate_temperature_uncertainty(
    df_uncertainty: float, magnetic_field_tesla: float
) -> float:
    """Calculates the uncertainty in temperature measurement based on the uncertainty in frequency difference (df)
    and magnetic field strength in Tesla.

    .. math::
        u(T) = 1.02 \\cdot 10^8 \\cdot \\frac{u(\\Delta f_{hm})}{\\gamma B_0}

    :param df_uncertainty: The uncertainty in frequency difference (df) in Hz.
    :type df_uncertainty: float
    :param magnetic_field_tesla: The magnetic field strength in Tesla.
    :type magnetic_field_tesla: float
    :return: The uncertainty in temperature measurement in Celsius.
    :rtype: float
    """
    return (1.02e8 * df_uncertainty) / (GAMMA_H * magnetic_field_tesla)


def lsq_fit_thermometry_signal_model(
    echo_times: NDArray[np.floating],
    signal_values: NDArray[np.floating],
    initial_guess: list[float],
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Performs a least squares fit of the thermometry signal model to the provided signal values.

    :param echo_times: The echo times.
    :type echo_times: NDArray[np.floating]
    :param signal_values: The signal values.
    :type signal_values: NDArray[np.floating]
    :param initial_guess: Initial guess for the parameters [amplitude_1, amplitude_2, r2star_1, r2star_2, df, dphi_deg].
    :type initial_guess: list[float]
    :return: A tuple containing the optimal parameters, the covariance of the parameters, and the coefficient of determination R^2.
    :rtype: tuple[NDArray[np.floating], NDArray[np.floating], float]
    """
    bounds = (0, [10, 10, 200, 200, 500, 360])
    try:
        popt, pcov, *_ = curve_fit(
            thermometry_signal_model,
            echo_times,
            signal_values,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000,
        )
    except RuntimeError:
        popt = np.array([np.nan] * len(initial_guess))
        pcov = np.full((len(initial_guess), len(initial_guess)), np.nan)

    r_squared = calculate_r_squared(
        signal_values, thermometry_signal_model(echo_times, *popt)
    )
    return popt, pcov, r_squared
