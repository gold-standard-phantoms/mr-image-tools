"""Multiecho Thermometry Filter

Functional implementation of fitting of multiecho magnitude data to a dual-resonance model
for MR thermometry applications.

Temperature is derived from the frequency difference between the two resonance peaks for ethylene glycol
For other substances, the frequency fitting can still be performed, but the temperature conversion will not be valid.

Based on Sprinkhuizen, S.M., Bakker, C.J.G. and Bartels, L.W. (2010),
Absolute MR thermometry using time-domain analysis of multi-gradient-echo magnitude images.
Magn. Reson. Med., 64: 239-248. https://doi.org/10.1002/mrm.22429
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
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
    df: Union[NDArray[np.floating], float],
    dphi_deg: float,
) -> NDArray[np.floating]:
    """Calculates the signal S(t) at time t according to the dual-resonance model.

    .. math::
        S(t) = \\sqrt{A_1^2e^{-2R_{2,1}^*t} + A_2^2e^{-2R_{2,2}^*t} + 2A_1A_2e^{-(R_{2,1}^* + R_{2,2}^*)t} \\cos(2\\pi\\Delta f_{12}t + \\Delta\\phi_{12})}

    Args:
        t (NDArray[np.floating]): The time vector.
        amplitude_1 (float): The amplitude of the first signal component.
        amplitude_2 (float): The amplitude of the second signal component.
        r2star_1 (float): The R2* value of the first signal component.
        r2star_2 (float): The R2* value of the second signal component.
        df (float): The frequency offset between the two signal components.
        dphi_deg (float): The phase offset between the two signal components.

    Returns:
        NDArray[np.floating]: The signal S(t) at time t according to the dual-resonance model.
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
    temperature_celsius: Union[NDArray[np.floating], float], magnetic_field_tesla: float
) -> Union[NDArray[np.floating], float]:
    """Calculates frequency difference (df) from temperature and magnetic field.

    .. math::
        T[°C] = 193.35 - 1.02 \\cdot 10^8 \\cdot \\frac{\\Delta f_{hm}[\\text{Hz}]}{\\gamma B_0}

    Args:
        temperature_celsius (float): The temperature in Celsius.
        magnetic_field_tesla (float): The magnetic field strength in Tesla.

    Returns:
        float: The frequency difference (df) in Hz.
    """
    return ((193.35 - temperature_celsius) * GAMMA_H * magnetic_field_tesla) / 1.02e8


def calculate_temperature_from_df(
    df: Union[NDArray[np.floating], float], magnetic_field_tesla: float
) -> Union[NDArray[np.floating], float]:
    """Calculates temperature from frequency difference (df) and magnetic field.

    .. math::
        T[°C] = 193.35 - 1.02 \\cdot 10^8 \\cdot \\frac{\\Delta f_{hm}[\\text{Hz}]}{\\gamma B_0}

    Args:
        df (float): The frequency difference (df) in Hz.
        magnetic_field_tesla (float): The magnetic field strength in Tesla.

    Returns:
        float: The temperature in Celsius.
    """
    return 193.35 - (1.02e8 * np.abs(df)) / (GAMMA_H * magnetic_field_tesla)


def calculate_temperature_uncertainty(
    df_uncertainty: float, magnetic_field_tesla: float
) -> float:
    """Calculates temperature uncertainty from frequency difference uncertainty.

    .. math::
        u(T) = 1.02 \\cdot 10^8 \\cdot \\frac{u(\\Delta f_{hm})}{\\gamma B_0}

    Args:
        df_uncertainty (float): The uncertainty in frequency difference (df) in Hz.
        magnetic_field_tesla (float): The magnetic field strength in Tesla.

    Returns:
        float: The uncertainty in temperature measurement in Celsius.
    """
    return (1.02e8 * df_uncertainty) / (GAMMA_H * magnetic_field_tesla)


def lsq_fit_thermometry_signal_model(
    echo_times: NDArray[np.floating],
    signal_values: NDArray[np.floating],
    initial_guess: list[float],
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Performs a least squares fit of the thermometry signal model.

    Args:
        echo_times (NDArray[np.floating]): The echo times.
        signal_values (NDArray[np.floating]): The signal values.
        initial_guess (list[float]): Initial guess for the parameters
            [amplitude_1, amplitude_2, r2star_1, r2star_2, df, dphi_deg].

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating], float]: A tuple
            containing the optimal parameters, the covariance of the parameters,
            and the coefficient of determination R^2.
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


AnalysisMethod = Literal["voxelwise", "regionwise", "regionwise_bootstrap"]


class MultiEchoThermometryParameters(BaseModel):
    """Parameters for the MultiEchoThermometryFilter.

    The shapes and affines of the input images must match for the filter to run.
    i.e. they must be co-located in world-space.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_multiecho: BaseImageContainer
    """An image with multiecho magnitude data. The fourth dimension is assumed to be the echo time dimension."""

    image_segmentation: BaseImageContainer
    """An integer label map image defining the region(s) to fit. The label map must be co-located with ``image_multiecho``.
    Voxels with the same integer value are treated as one region and fitted jointly."""

    echo_times: list[float]
    "List of echo times, one for each multi-echo volume. Length should be the same as the fourth dimension of ``image_multiecho``."

    magnetic_field_tesla: float
    """Magnetic field strength in Tesla."""

    analysis_method: AnalysisMethod = "regionwise"
    """The analysis method to use. Options are:
    - "voxelwise": Fit each voxel independently. This is the most computationally intensive option,
      but can capture spatial variations in the parameters.
    - "regionwise": Fit each region defined by the segmentation mask jointly.
      This is less computationally intensive and can improve SNR, but assumes that the parameters
      are constant within each region.
    - "regionwise_bootstrap": Fit each region defined by the segmentation mask jointly,
      using bootstrapping to estimate parameter uncertainties. This is the most robust option,
      but also the most computationally intensive.
    """
    n_bootstrap: int = 100
    """Number of bootstrap samples to use for the "regionwise_bootstrap" analysis method. default is 100."""

    @model_validator(mode="after")
    def check_shapes_match(self) -> "MultiEchoThermometryParameters":
        """Checks that the shapes of the input images match"""
        if self.image_multiecho.shape[:-1] != self.image_segmentation.shape:
            raise ValueError("Input images must have the same shape")
        return self

    @model_validator(mode="after")
    def check_affines_match(self) -> "MultiEchoThermometryParameters":
        """Checks that the affines of the input images match"""
        if not (self.image_multiecho.affine == self.image_segmentation.affine).all():
            raise ValueError("Input images must have the same affine")
        return self

    @model_validator(mode="after")
    def check_echo_times_length(self) -> "MultiEchoThermometryParameters":
        """Checks that the length of echo_times matches the fourth dimension of image_multiecho"""
        if len(self.echo_times) != self.image_multiecho.shape[-1]:
            raise ValueError(
                "Length of echo_times must match the fourth dimension of image_multiecho"
            )
        return self


@dataclass
class ThermometryResults:
    """Results for a single region from the MultiEchoThermometryFilter."""

    region_id: int
    temperature: float
    temperature_uncertainty: Tuple[float, float]
    r_squared: float


def multiecho_thermometry_filter(
    parameters: MultiEchoThermometryParameters,
) -> Tuple[List[ThermometryResults], BaseImageContainer]:
    """Filter to perform multi-echo thermometry analysis.

    Args:
        parameters (MultiEchoThermometryParameters): Parameters for the filter.

    Returns:
        tuple containing
        - ThermometryResults with the results for each region, with the following properties:
            - "id": int. The region ID from the segmentation mask.
            - "temperature": float. Estimated region temperature in °C
            - "temperature_uncertainty": tuple[float,float]. Estimated uncertainty in temperature in °C and k-value
            - "r_squared": float. Coefficient of determination for the fit.
        - an image container with the temperature map (in Celsius)
    """
    image_multiecho = parameters.image_multiecho
    image_segmentation = parameters.image_segmentation
    echo_times = np.array(parameters.echo_times)
    magnetic_field_tesla = parameters.magnetic_field_tesla
    analysis_method = parameters.analysis_method
    n_bootstrap = parameters.n_bootstrap

    image_temperature = image_multiecho.clone()
    image_temperature.image = np.zeros(image_segmentation.shape, dtype=np.float64)
    nx, ny, nz, n_echoes = image_multiecho.shape
    # loop over regions in segmentation mask
    regions = np.unique(image_segmentation.image)
    regions = regions[regions != 0]  # exclude background
    results = []
    for region in regions:
        if analysis_method == "regionwise":
            region_mask = image_segmentation.image == region
            # get the mean region signal for each echo time
            region_signal = np.array(
                [
                    np.mean(image_multiecho.image[..., echo][region_mask])
                    for echo in range(n_echoes)
                ]
            )
            initial_guess = [1.0, 1.0, 50.0, 50.0, 100.0, 0.0]
            fitted_params, pcov, r_squared = lsq_fit_thermometry_signal_model(
                echo_times, region_signal, initial_guess
            )
            df = fitted_params[4]
            df_uncertainty = (
                np.sqrt(np.abs(pcov[4, 4])) if not np.isnan(pcov[4, 4]) else np.nan
            )
            temperature = calculate_temperature_from_df(df, magnetic_field_tesla)
            temperature_uncertainty = calculate_temperature_uncertainty(
                df_uncertainty, magnetic_field_tesla
            )
            results.append(
                ThermometryResults(
                    region_id=int(region),
                    temperature=float(temperature),
                    temperature_uncertainty=(2 * temperature_uncertainty, 2),
                    r_squared=r_squared,
                )
            )
            image_temperature.image[image_segmentation.image == region] = temperature

    return results, image_temperature
