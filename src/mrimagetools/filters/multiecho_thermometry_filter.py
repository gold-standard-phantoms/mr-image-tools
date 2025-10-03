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
        $S(t) = \sqrt{A_1^2e^{-2R_{2,1}^*t} + A_2^2e^{-2R_{2,2}^*t} + 2A_1A_2e^{-(R_{2,1}^* + R_{2,2}^*)t} \cos(2\pi\Delta f_{12}t + \Delta\phi_{12})}$

    Parameters
    ----------
    t : NDArray[np.floating]
        The time vector.
    amplitude_1 : float
        The amplitude of the first signal component.
    amplitude_2 : float
        The amplitude of the second signal component.
    r2star_1 : float
        The R2* value of the first signal component.
    r2star_2 : float
        The R2* value of the second signal component.
    df : float
        The frequency offset between the two signal components.
    dphi_deg : float
        The phase offset between the two signal components.

    Returns
    ----------
    NDArray[np.floating]
        The signal S(t) at time t according to the dual-resonance model.

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
        $T[°C] = 193.35 - 1.02 \cdot 10^8 \cdot \frac{\Delta f_{hm}[\text{Hz}]}{\gamma B_0}$

    Parameters
    ----------
    temperature_celsius : float
        The temperature in Celsius.
    magnetic_field_tesla : float
        The magnetic field strength in Tesla.

    Returns
    ----------
    float
        The frequency difference (df) in Hz.
    """
    return ((193.35 - temperature_celsius) * GAMMA_H * magnetic_field_tesla) / 1.02e8


def calculate_temperature_from_df(df: float, magnetic_field_tesla: float) -> float:
    """Calculates temperature in Celsius from frequency difference (df) and magnetic field strength in Tesla.

    .. math::
        $T[°C] = 193.35 - 1.02 \cdot 10^8 \cdot \frac{\Delta f_{hm}[\text{Hz}]}{\gamma B_0}$

    Parameters
    ----------
    df : float
        The frequency difference (df) in Hz.
    magnetic_field_tesla : float
        The magnetic field strength in Tesla.

    Returns
    ----------
    float
        The temperature in Celsius.
    """
    return 193.35 - (1.02e8 * abs(df)) / (GAMMA_H * magnetic_field_tesla)


def calculate_temperature_uncertainty(
    df_uncertainty: float, magnetic_field_tesla: float
) -> float:
    """Calculates the uncertainty in temperature measurement based on the uncertainty in frequency difference (df)
    and magnetic field strength in Tesla.

    .. math::
        $u(T) = 1.02 \cdot 10^8 \cdot \frac{u(\Delta f_{hm})}{\gamma B_0}$

    Parameters
    ----------
    df_uncertainty : float
        The uncertainty in frequency difference (df) in Hz.
    magnetic_field_tesla : float
        The magnetic field strength in Tesla.

    Returns
    ----------
    float
        The uncertainty in temperature measurement in Celsius.
    """
    return (1.02e8 * df_uncertainty) / (GAMMA_H * magnetic_field_tesla)


def lsq_fit_thermometry_signal_model(
    echo_times: NDArray[np.floating],
    signal_values: NDArray[np.floating],
    initial_guess: list[float],
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Performs a least squares fit of the thermometry signal model to the provided signal values.
    Parameters
    ----------
    echo_times : NDArray[np.floating]
        The echo times.
    signal_values : NDArray[np.floating]
        The signal values.
    initial_guess : list[float]
        Initial guess for the parameters [amplitude_1, amplitude_2, r2star_1, r2star_2, df, dphi_deg].

    Returns
    ----------
    tuple[NDArray[np.floating], NDArray[np.floating], float]
        A tuple containing the optimal parameters, the covariance of the parameters, and the coefficient of determinarion R^2.
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

    residuals = signal_values - thermometry_signal_model(echo_times, *popt)
    ss_res: np.floating = np.sum(residuals**2)
    ss_tot: np.floating = np.sum((signal_values - np.mean(signal_values)) ** 2)
    r_squared: float = 1 - (float(ss_res) / float(ss_tot)) if ss_tot != 0 else 0.0
    return popt, pcov, r_squared
