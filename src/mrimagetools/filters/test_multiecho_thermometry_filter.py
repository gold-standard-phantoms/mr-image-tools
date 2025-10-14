"""Tests for multiecho thermometry filter functions."""

import pdb
import re
from turtle import rt
from typing import Dict, List, Literal, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from mrimagetools.filters.multiecho_thermometry_filter import (
    AnalysisMethod,
    MultiEchoThermometryParameters,
    calculate_df_from_temperature,
    calculate_temperature_from_df,
    calculate_temperature_uncertainty,
    lsq_fit_thermometry_signal_model,
    multiecho_thermometry_filter,
    thermometry_signal_model,
)
from mrimagetools.v2.containers.image import NiftiImageContainer

GAMMA_H = 42.57747892e6  # Hz/T


@pytest.fixture
def thermometry_test_data() -> (
    Tuple[
        float,
        float,
        float,
        List[float],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[float],
    ]
):
    """Provides a set of realistic parameters and data for thermometry model testing."""
    # Define known parameters
    magnetic_field_tesla = 3.0
    temperature_celsius = 37.0
    amplitude_1 = 1.0
    amplitude_2 = 0.5
    r2star_1 = 50.0  # 1/s
    r2star_2 = 100.0  # 1/s
    dphi_deg = 45.0

    # Calculate expected df from temperature
    df = (193.35 - temperature_celsius) * GAMMA_H * magnetic_field_tesla / 1.02e8

    # Generate echo times (24 echoes from 1ms to 24ms)
    echo_times = np.linspace(0.001, 0.024, 24)

    # True parameters for signal generation
    p_true = [amplitude_1, amplitude_2, r2star_1, r2star_2, df, dphi_deg]

    def signal_model(te, a1, a2, r2s1, r2s2, df, dphi) -> np.ndarray:
        dphi_rad = np.deg2rad(dphi)
        term1 = (a1**2) * np.exp(-2 * r2s1 * te)
        term2 = (a2**2) * np.exp(-2 * r2s2 * te)
        term3 = (
            2
            * a1
            * a2
            * np.exp(-(r2s1 + r2s2) * te)
            * np.cos(2 * np.pi * df * te + dphi_rad)
        )
        radicand = term1 + term2 + term3
        radicand[radicand < 0] = 0  # Prevent negative values under the square root
        return np.sqrt(radicand)

    # Generate ideal signal data
    signal_ideal = signal_model(echo_times, *p_true)

    # Generate noisy signal data for fitting tests
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.02, signal_ideal.shape)
    signal_noisy = signal_ideal + noise

    # Generate poorly sampled data that should lead to fitting failure
    echo_times_poor = np.array([0.001, 0.002, 0.003])  # Only 3 echoes
    signal_poor = thermometry_signal_model(echo_times_poor, *p_true) + rng.normal(
        0, 0.02, (3,)
    )

    # Provide a reasonable initial guess for the fitting function
    initial_guess = [0.9, 0.4, 40.0, 80.0, df * 1.1, 30.0]

    return (
        magnetic_field_tesla,
        temperature_celsius,
        df,
        p_true,
        echo_times,
        signal_ideal,
        signal_noisy,
        echo_times_poor,
        signal_poor,
        initial_guess,
    )


def test_calculate_df_from_temperature(
    thermometry_test_data: Tuple[
        float,
        float,
        float,
        List[float],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[float],
    ]
) -> None:
    """Tests the calculation of df from temperature."""
    (
        magnetic_field_tesla,
        temperature_celsius,
        expected_df,
        *_,
    ) = thermometry_test_data

    calculated_df = calculate_df_from_temperature(
        temperature_celsius, magnetic_field_tesla
    )

    assert np.isclose(
        calculated_df, expected_df, atol=1e-3
    ), f"Calculated df {calculated_df} does not match expected {expected_df}"


def test_calculate_temperature_from_df(
    thermometry_test_data: Tuple[
        float,
        float,
        float,
        List[float],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[float],
    ]
) -> None:
    """Tests the calculation of temperature from df."""
    (
        magnetic_field_tesla,
        expected_temperature,
        df,
        *_,
    ) = thermometry_test_data

    calculated_temperature = calculate_temperature_from_df(df, magnetic_field_tesla)
    assert np.isclose(
        calculated_temperature, expected_temperature, atol=1e-3
    ), f"Calculated temperature {calculated_temperature} does not match expected {expected_temperature}"


def test_calculate_temperature_uncertainty() -> None:
    """Tests the calculation of temperature uncertainty from df uncertainty."""
    magnetic_field_tesla = 3.0
    df_uncertainty = 1.0  # Hz

    expected_uncertainty = (1.02e8 * df_uncertainty) / (GAMMA_H * magnetic_field_tesla)
    calculated_uncertainty = calculate_temperature_uncertainty(
        df_uncertainty, magnetic_field_tesla
    )

    assert np.isclose(
        calculated_uncertainty, expected_uncertainty, atol=1e-6
    ), f"Calculated uncertainty {calculated_uncertainty} does not match expected {expected_uncertainty}"


def test_thermometry_signal_model(
    thermometry_test_data: Tuple[
        float,
        float,
        float,
        List[float],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[float],
    ]
) -> None:
    """Tests the thermometry signal model function."""
    (
        *_,
        p_true,
        echo_times,
        signal_ideal,
        _,
        _,
        _,
        _,
    ) = thermometry_test_data

    modeled_signal = thermometry_signal_model(echo_times, *p_true)

    assert np.allclose(
        modeled_signal, signal_ideal, atol=1e-6
    ), "Modeled signal does not match ideal signal"


def test_lsq_fit_thermometry_signal_model(
    thermometry_test_data: Tuple[
        float,
        float,
        float,
        List[float],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[float],
    ]
) -> None:
    """Tests the least squares fitting of the thermometry signal model."""
    (
        *_,
        p_true,
        echo_times,
        signal_ideal,
        signal_noisy,
        _,
        _,
        initial_guess,
    ) = thermometry_test_data

    # first test against ideal data
    (
        fitted_params,
        _,
        r_squared,
    ) = lsq_fit_thermometry_signal_model(echo_times, signal_ideal, initial_guess)

    # should recover paramaters almost perfectly for clean data
    assert np.allclose(
        fitted_params, p_true, rtol=1e-6
    ), f"Fitted parameters {fitted_params} do not match true parameters {p_true} within 1e-6 tolerance"

    assert np.isclose(
        r_squared, 1.0, rtol=1e-6
    ), f"Residual norm {r_squared} does not match expected 1.0"

    # now test against noisy data
    (
        fitted_params,
        _,
        r_squared,
    ) = lsq_fit_thermometry_signal_model(echo_times, signal_noisy, initial_guess)

    # Should recover parameters within reasonable error for noisy data
    assert np.allclose(
        fitted_params, p_true, rtol=0.1
    ), f"Fitted parameters {fitted_params} do not match true parameters {p_true} within 0.1 tolerance"
    assert r_squared > 0.95  # still good fit despite noise


@pytest.fixture
def thermometry_test_volume() -> (
    Tuple[
        float,
        np.ndarray,
        NiftiImageContainer,
        NiftiImageContainer,
        NiftiImageContainer,
        np.ndarray,
        List[int],
        List[float],
    ]
):
    region_id = [1, 2, 3]
    magnetic_field_tesla = 3.0
    region_temperature_celsius = [20.0, 25.0, 30.0]
    echo_times = np.linspace(0.001, 0.024, 24)
    amplitude_1 = 1.0
    amplitude_2 = 0.5
    r2star_1 = 1.0  # 1/s
    r2star_2 = 2.0  # 1/s
    dphi_deg = 45.0
    nx, ny, nz = 3, 6, 4  # small volume for testing

    # create a small 3x6x4 image with 3 regions
    true_temperature_map: np.ndarray = np.zeros((nx, ny, nz), dtype=np.float64)
    for i, temp in enumerate(region_temperature_celsius):
        true_temperature_map[i, :, :] = temp

    # create a segmentation mask
    segmentation_mask: np.ndarray = np.zeros((nx, ny, nz), dtype=np.int16)
    for i, rid in enumerate(region_id):
        segmentation_mask[i, :, :] = rid

    # simulate multi-echo data
    multi_echo_data = thermometry_signal_model(
        echo_times,
        amplitude_1,
        amplitude_2,
        r2star_1,
        r2star_2,
        calculate_df_from_temperature(
            np.repeat(
                true_temperature_map[:, :, :, np.newaxis], len(echo_times), axis=3
            ),
            magnetic_field_tesla,
        ),
        dphi_deg,
    )
    multi_echo_image = NiftiImageContainer(
        nifti_img=nib.nifti1.Nifti1Image(multi_echo_data, affine=np.eye(4))
    )
    # add some noise to create a noisy version
    rng = np.random.default_rng(43)
    noisy_multi_echo_image = NiftiImageContainer(
        nifti_img=nib.nifti1.Nifti1Image(
            multi_echo_data + rng.normal(0, 0.01, multi_echo_data.shape),
            affine=np.eye(4),
        )
    )
    segmentation_image = NiftiImageContainer(
        nifti_img=nib.nifti1.Nifti1Image(segmentation_mask, affine=np.eye(4))
    )

    return (
        magnetic_field_tesla,
        true_temperature_map,
        segmentation_image,
        multi_echo_image,
        noisy_multi_echo_image,
        echo_times,
        region_id,
        region_temperature_celsius,
    )


def test_multiecho_thermometry_parameters(
    thermometry_test_volume: Tuple[
        float,
        np.ndarray,
        NiftiImageContainer,
        NiftiImageContainer,
        NiftiImageContainer,
        np.ndarray,
        List[int],
        List[float],
    ]
) -> None:
    """Test the MultiEchoThermometryParameters dataclass."""
    (
        magnetic_field_tesla,
        true_temperature_map,
        segmentation_image,
        multi_echo_image,
        noisy_multi_echo_image,
        echo_times,
        region_id,
        region_temperature_celsius,
    ) = thermometry_test_volume

    # test with valid parameters
    params = MultiEchoThermometryParameters(
        image_multiecho=multi_echo_image,
        image_segmentation=segmentation_image,
        echo_times=echo_times.tolist(),
        magnetic_field_tesla=magnetic_field_tesla,
        analysis_method="regionwise",
        n_bootstrap=100,
    )
    assert params.image_multiecho == multi_echo_image
    assert params.image_segmentation == segmentation_image
    assert params.echo_times == echo_times.tolist()
    assert params.magnetic_field_tesla == magnetic_field_tesla
    assert params.analysis_method == "regionwise"
    assert params.n_bootstrap == 100

    # test with invalid analysis_method
    with pytest.raises(ValueError):
        MultiEchoThermometryParameters(
            image_multiecho=multi_echo_image,
            image_segmentation=segmentation_image,
            echo_times=echo_times.tolist(),
            magnetic_field_tesla=magnetic_field_tesla,
            analysis_method="invalid_method",  # type: ignore
        )
    # test with wrong image shape
    image_multiecho_invalid_shape = NiftiImageContainer(
        nifti_img=nib.nifti1.Nifti1Image(np.zeros((1, 1, 1, 12)), affine=np.eye(4))
    )
    with pytest.raises(ValueError):
        MultiEchoThermometryParameters(
            image_multiecho=image_multiecho_invalid_shape,
            image_segmentation=segmentation_image,
            echo_times=echo_times.tolist(),
            magnetic_field_tesla=magnetic_field_tesla,
            analysis_method="regionwise",
        )

    # test with mismatched affine
    image_multiecho_invalid_affine = multi_echo_image.clone()
    image_multiecho_invalid_affine.nifti_image.set_sform(2 * np.eye(4))

    with pytest.raises(ValueError):
        MultiEchoThermometryParameters(
            image_multiecho=image_multiecho_invalid_affine,
            image_segmentation=segmentation_image,
            echo_times=echo_times.tolist(),
            magnetic_field_tesla=magnetic_field_tesla,
            analysis_method="regionwise",
        )

    # test with incorrect number of echoes
    with pytest.raises(ValueError):
        MultiEchoThermometryParameters(
            image_multiecho=multi_echo_image,
            image_segmentation=segmentation_image,
            echo_times=echo_times.tolist()[:-1],  # one less than actual
            magnetic_field_tesla=magnetic_field_tesla,
            analysis_method="regionwise",
        )


def test_multiecho_thermometry_filter_function_regionwise(
    thermometry_test_volume: Tuple[
        float,
        np.ndarray,
        NiftiImageContainer,
        NiftiImageContainer,
        NiftiImageContainer,
        np.ndarray,
        List[int],
        List[float],
    ]
) -> None:
    (
        magnetic_field_tesla,
        true_temperature_map,
        segmentation_image,
        multi_echo_image,
        _,
        echo_times,
        region_id,
        region_temperature_celsius,
    ) = thermometry_test_volume

    # test regionwise AnalysisMethod
    results, temperature_image = multiecho_thermometry_filter(
        parameters=MultiEchoThermometryParameters(
            image_multiecho=multi_echo_image,
            image_segmentation=segmentation_image,
            echo_times=echo_times.tolist(),
            magnetic_field_tesla=magnetic_field_tesla,
            analysis_method="regionwise",
        )
    )

    assert isinstance(temperature_image, NiftiImageContainer)
    # compare results to true temperature map

    np.testing.assert_array_almost_equal(
        temperature_image.image,
        true_temperature_map,
    )

    # compare results to expected region temperatures
    for res in results:
        # check the region id is in the expected list
        assert res.region_id in region_id
        # check the region temperature is close to the expected value
        assert np.isclose(
            res.region_mean_temperature,
            region_temperature_celsius[region_id.index(res.region_id)],
        )
        # check that the uncertainty is close to zero for clean data
        assert np.isclose(res.region_temperature_uncertainty[0], 0.00, atol=1e-6)
        # check that the uncertainty k-value is 1
        assert res.region_temperature_uncertainty[1] == 1
        # check that res.region_size is correct
        assert res.region_size == np.sum(segmentation_image.image == res.region_id)
        # check that the fit r_squared is close to 1
        assert np.allclose(res.r_squared, 1.0)
        # value arrays should equal the scalar values for regionwise analysis
        assert np.allclose(res.region_temperature_values, res.region_mean_temperature)
        # uncertainty arrays should equal the scalar values for regionwise analysis
        assert (
            res.region_temperature_uncertainty_values
            == res.region_temperature_uncertainty[0]
        ).all()


def test_multiecho_thermometry_filter_function_voxelwise(
    thermometry_test_volume: Tuple[
        float,
        np.ndarray,
        NiftiImageContainer,
        NiftiImageContainer,
        NiftiImageContainer,
        np.ndarray,
        List[int],
        List[float],
    ]
) -> None:
    (
        magnetic_field_tesla,
        true_temperature_map,
        segmentation_image,
        multi_echo_image,
        noisy_multi_echo_image,
        echo_times,
        region_id,
        region_temperature_celsius,
    ) = thermometry_test_volume

    # test voxelwise AnalysisMethod with clean data
    results, temperature_image = multiecho_thermometry_filter(
        parameters=MultiEchoThermometryParameters(
            image_multiecho=multi_echo_image,
            image_segmentation=segmentation_image,
            echo_times=echo_times.tolist(),
            magnetic_field_tesla=magnetic_field_tesla,
            analysis_method="voxelwise",
        )
    )
    assert isinstance(temperature_image, NiftiImageContainer)

    np.testing.assert_array_almost_equal(temperature_image.image, true_temperature_map)

    # compare results to expected region temperatures
    for res in results:
        # check the region id is in the expected list
        assert res.region_id in region_id
        # check the region temperature is close to the expected value
        assert np.isclose(
            res.region_mean_temperature,
            region_temperature_celsius[region_id.index(res.region_id)],
        )
        # check that the uncertainty is close to zero for clean data
        assert np.isclose(res.region_temperature_uncertainty[0], 0.00, atol=1e-6)
        # check that the uncertainty k-value is 1
        assert res.region_temperature_uncertainty[1] == 1
        # check that res.region_size is correct
        assert res.region_size == np.sum(segmentation_image.image == res.region_id)
        # check that the fit r_squared is close to 1
        assert np.allclose(res.r_squared, 1.0)
        # value arrays should equal the scalar values for regionwise analysis
        assert np.allclose(res.region_temperature_values, res.region_mean_temperature)

    # now test with noisy data
    results, noisy_temperature_image = multiecho_thermometry_filter(
        parameters=MultiEchoThermometryParameters(
            image_multiecho=noisy_multi_echo_image,
            image_segmentation=segmentation_image,
            echo_times=echo_times.tolist(),
            magnetic_field_tesla=magnetic_field_tesla,
            analysis_method="voxelwise",
        )
    )


def test_multiecho_thermometry_filter_function_regionwise_bootstrap(
    thermometry_test_volume: Tuple[
        float,
        np.ndarray,
        NiftiImageContainer,
        NiftiImageContainer,
        NiftiImageContainer,
        np.ndarray,
        List[int],
        List[float],
    ]
) -> None:
    (
        magnetic_field_tesla,
        true_temperature_map,
        segmentation_image,
        multi_echo_image,
        noisy_multi_echo_image,
        echo_times,
        region_id,
        region_temperature_celsius,
    ) = thermometry_test_volume
    n_bootstrap = 100
    # test regionwise AnalysisMethod with bootstrapping
    results, temperature_image = multiecho_thermometry_filter(
        parameters=MultiEchoThermometryParameters(
            image_multiecho=noisy_multi_echo_image,
            image_segmentation=segmentation_image,
            echo_times=echo_times.tolist(),
            magnetic_field_tesla=magnetic_field_tesla,
            analysis_method="regionwise_bootstrap",
            n_bootstrap=n_bootstrap,
        )
    )

    # basic checks - correct types and sizes
    assert isinstance(temperature_image, NiftiImageContainer)
    for res in results:
        # check the region id is in the expected list
        assert res.region_id in region_id

        # check that the length of region_temperature_values and r_squared arrays equal the n_bootstrap value
        assert res.region_temperature_values.size == n_bootstrap
        assert res.r_squared.size == n_bootstrap
        assert res.region_temperature_uncertainty_values.size == n_bootstrap
