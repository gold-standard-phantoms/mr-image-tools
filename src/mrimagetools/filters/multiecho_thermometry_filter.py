"""Multiecho Thermometry Filter

Functional implementation of fitting of multiecho magnitude data
to a dual-resonance model for MR thermometry applications.

Temperature is derived from the frequency difference between
the two resonance peaks for ethylene glycol.For other substances,
the frequency fitting can still be performed, but the temperature
conversion will not be valid.

Based on Sprinkhuizen, S.M., Bakker, C.J.G. and Bartels, L.W. (2010),
Absolute MR thermometry using time-domain analysis of multi-gradient-echo magnitude images.
Magn. Reson. Med., 64: 239-248. https://doi.org/10.1002/mrm.22429

Chemical shift to temperature conversion is based on
Calibration of methanol and ethylene glycol nuclear magnetic resonance thermometers
David S. Raiford, Cherie L. Fisk, and Edwin D. Becker
Analytical Chemistry 1979 51 (12), 2050-2051
DOI: https://doi.org/10.1021/ac50048a040

"""

import pdb
from dataclasses import dataclass
from typing import List, Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.optimize import curve_fit

from mrimagetools.filters.adc_quantification_filter import calculate_r_squared
from mrimagetools.v2.containers.image import BaseImageContainer

GAMMA_H = 42.57747892e6  # Hz/T
RANDOM_SEED = 840275920  # fixed seed for reproducibility


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

    Attributes:
        image_multiecho (BaseImageContainer): An image with multiecho magnitude data. The fourth dimension is assumed to be the echo time dimension.
        image_segmentation (BaseImageContainer): An integer label map image defining the region(s) to fit. The label map must be co-located with ``image_multiecho``.
            Voxels with the same integer value are treated as one region and fitted jointly.
        echo_times (list[float]): List of echo times, one for each multi-echo volume. Length should be the same as the fourth dimension of ``image_multiecho``.
        magnetic_field_tesla (float): Magnetic field strength in Tesla.
        analysis_method (AnalysisMethod): The analysis method to use. Options are:
            - "voxelwise": Fit each voxel independently. This is the most computationally intensive option,
              but can capture spatial variations in the parameters.
            - "regionwise": Fit each region defined by the segmentation mask jointly.
            - "regionwise_bootstrap": Fit each region defined by the segmentation mask jointly with bootstrapping to estimate parameter uncertainties.
        n_bootstrap (int): Number of bootstrap samples to use for the "regionwise_bootstrap" analysis method. default is 100

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
    """Results for a single region from the MultiEchoThermometryFilter.

    Attributes:
        region_id (int): The region ID from the segmentation mask.
        region_mean_temperature (float): The mean temperature of the region in °C.
        region_temperature_uncertainty (Tuple[float, float]): The uncertainty in temperature in °C and k-value.
        r_squared (NDArray[np.floating]): The coefficient of determination for the fit.
        region_temperature_values (NDArray[np.floating]): The temperature values for each voxel in the region in °C.
        region_temperature_uncertainty_values (NDArray[np.floating]): The standard uncertainty (k=1) in temperature for each voxel in °C
        region_size (int): The number of voxels in the region.

    """

    region_id: int
    region_mean_temperature: float
    region_temperature_uncertainty: Tuple[float, float]  # (uncertainty in °C, k-value)
    r_squared: NDArray[np.floating]
    region_temperature_values: NDArray[np.floating]
    region_temperature_uncertainty_values: NDArray[np.floating]
    region_size: int = 0  # number of voxels in region


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

    ``regionwise`` calculates the mean signal within each region for each echo time, and fits the model to this mean signal.
    The resulting temperature is assigned to all voxels in the region. The fit uncertainty is calculated from the
    covariance matrix of the fit, i.e. `sqrt(diag(pcov))`. The uncertainty in temperature is derived from the uncertainty
    in the fitted frequency difference parameter (df), which is converted to temperature uncertainty using the function ``calculate_temperature_uncertainty``.
    This uncertainty represents the precision of the fit to the mean signal, and does not account for the uncertainty
    in the echo times.

    ``voxelwise`` fits the model to each voxel independently. The uncertainty for each voxel is calculated from the covariance matrix of the fit.
    The mean temperature for the region is calculated as the weighted mean of the voxel temperatures, using 1/uncertainty^2 as weights.

    ``regionwise_bootstrap`` fits the model to the mean signal within each region, using bootstrapping to estimate the uncertainty.
    Within each region, a signal vector is created by sampling with replacement from the voxel signals in the region.
    The model is fitted to this signal vector, and the temperature is calculated from the fitted df parameter.
    This is repeated ``n_bootstrap`` times, and the uncertainty is calculated as the standard deviation of the bootstrap temperatures.
    """
    image_multiecho = parameters.image_multiecho
    image_segmentation = parameters.image_segmentation
    echo_times = np.array(parameters.echo_times)
    magnetic_field_tesla = parameters.magnetic_field_tesla
    analysis_method = parameters.analysis_method
    n_bootstrap = parameters.n_bootstrap
    initial_guess = [1.0, 1.0, 50.0, 50.0, 100.0, 0.0]
    r_squared_threshold = 0.9  # threshold for acceptable fit quality

    # clone the multiecho image container to create the output temperature image
    image_temperature = image_multiecho.clone()
    image_temperature.image = np.zeros(image_segmentation.shape, dtype=np.float64)
    nx, ny, nz, n_echoes = image_multiecho.shape

    # loop over regions in segmentation mask
    regions = np.unique(image_segmentation.image)
    regions = regions[regions != 0]  # exclude background
    results = []

    for region in regions:
        region_temperature_values_list = []
        region_temperature_uncertainty_values_list = []
        r_squared_list = []
        region_mask = image_segmentation.image == region
        region_size = np.sum(region_mask)  # number of voxels in region

        if analysis_method == "regionwise":
            # Regionwise Method
            # perform fitting on a regionwise basis, based on the mean signal
            # in the region for each echo time
            #

            # get the mean region signal for each echo time
            region_signal = np.array(
                [
                    np.mean(image_multiecho.image[..., echo][region_mask])
                    for echo in range(n_echoes)
                ]
            )
            fitted_params, pcov, r_squared_value = lsq_fit_thermometry_signal_model(
                echo_times, region_signal, initial_guess
            )
            df = fitted_params[4]
            param_uncertainties = np.sqrt(np.diag(pcov))
            df_uncertainty = (
                param_uncertainties[4]
                if not np.isnan(param_uncertainties[4])
                else np.nan
            )
            mean_region_temperature = calculate_temperature_from_df(
                df, magnetic_field_tesla
            )
            region_temperature_uncertainty = calculate_temperature_uncertainty(
                df_uncertainty, magnetic_field_tesla
            )

            image_temperature.image[region_mask] = mean_region_temperature
            # one one value per region
            region_temperature_values = np.array([mean_region_temperature])
            region_temperature_uncertainty_values = np.array(
                [region_temperature_uncertainty]
            )
            r_squared = np.array([r_squared_value])

        elif analysis_method == "voxelwise":
            # Voxelwise Method
            # perform fitting on a voxelwise basis
            #

            count = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if region_mask[i, j, k]:
                            voxel_signal = image_multiecho.image[i, j, k, :]
                            (
                                fitted_params,
                                pcov,
                                r_squared_value,
                            ) = lsq_fit_thermometry_signal_model(
                                echo_times, voxel_signal, initial_guess
                            )
                            df = fitted_params[4]
                            param_uncertainties = np.sqrt(np.diag(pcov))
                            df_uncertainty = (
                                param_uncertainties[4]
                                if not np.isnan(param_uncertainties[4])
                                else np.nan
                            )
                            r_squared_list.append(r_squared_value)
                            region_temperature_values_list.append(
                                calculate_temperature_from_df(df, magnetic_field_tesla)
                            )

                            region_temperature_uncertainty_values_list.append(
                                calculate_temperature_uncertainty(
                                    df_uncertainty, magnetic_field_tesla
                                )
                            )
                            # assign voxel temperature to output image
                            image_temperature.image[
                                i, j, k
                            ] = region_temperature_values_list[count]
                            count += 1

            # calculate the weighted mean of the region temperature, using 1/uncertainty^2 as weights
            region_temperature_values = np.array(region_temperature_values_list)
            region_temperature_uncertainty_values = np.array(
                region_temperature_uncertainty_values_list
            )
            r_squared = np.array(r_squared_list)
            # avoid division by zero
            weights = np.divide(
                1.0,
                region_temperature_uncertainty_values**2,
                where=region_temperature_uncertainty_values != 0,
                out=np.zeros_like(region_temperature_uncertainty_values),
            )
            mean_region_temperature = np.average(
                region_temperature_values, weights=weights, axis=0
            )
            # calculate the region-based uncertainty
            region_temperature_uncertainty = np.sqrt(1.0 / np.sum(weights))

        elif analysis_method == "regionwise_bootstrap":
            # Regionwise Bootstrap Method
            # perform fitting on a regionwise basis, using bootstrapping to estimate uncertainty
            #
            #
            rng = np.random.default_rng(seed=RANDOM_SEED)

            # perform bootrap fitting on a regionwise basis
            for b in range(n_bootstrap):
                # sample with replacement from the voxel signals in the region

                sampled_indices = rng.choice(
                    np.where(region_mask.flatten())[0],
                    size=region_size,
                    replace=True,
                )
                sampled_signals = image_multiecho.image.reshape(-1, n_echoes)[
                    sampled_indices
                ]
                # calculate the mean signal of the sampled voxels for each echo time
                region_signal = np.mean(sampled_signals, axis=0)
                fitted_params, pcov, r_squared_value = lsq_fit_thermometry_signal_model(
                    echo_times, region_signal, initial_guess
                )
                df = fitted_params[4]  # fitted frequency difference
                param_uncertainties = np.sqrt(np.diag(pcov))
                df_uncertainty = (
                    param_uncertainties[4]
                    if not np.isnan(param_uncertainties[4])
                    else np.nan
                )
                region_temperature_uncertainty_values_list.append(
                    calculate_temperature_uncertainty(
                        df_uncertainty, magnetic_field_tesla
                    )
                )
                # calculate temperature from df
                region_temperature_values_list.append(
                    calculate_temperature_from_df(df, magnetic_field_tesla)
                )
                r_squared_list.append(r_squared_value)

            # calculate the mean and standard deviation of the bootstrapped temperatures
            region_temperature_values = np.array(region_temperature_values_list)
            region_temperature_uncertainty_values = np.array(
                region_temperature_uncertainty_values_list
            )
            r_squared = np.array(r_squared_list)
            mean_region_temperature = (
                np.average(region_temperature_values[r_squared >= r_squared_threshold])
                if np.any(r_squared >= r_squared_threshold)
                else np.nan
            )
            region_temperature_uncertainty = (
                np.std(region_temperature_values[r_squared >= r_squared_threshold])
                if np.any(r_squared >= r_squared_threshold)
                else np.nan
            )
            image_temperature.image[region_mask] = mean_region_temperature

        # add results to list
        results.append(
            ThermometryResults(
                region_id=int(region),
                region_mean_temperature=float(mean_region_temperature),
                region_temperature_uncertainty=(
                    float(region_temperature_uncertainty),
                    1,
                ),  # k=1
                r_squared=r_squared,
                region_size=int(region_size),
                region_temperature_values=region_temperature_values,
                region_temperature_uncertainty_values=region_temperature_uncertainty_values,
            )
        )

    return results, image_temperature
