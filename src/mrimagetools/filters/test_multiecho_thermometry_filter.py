"""Tests for multiecho thermometry filter functions."""

import pdb
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytest

from mrimagetools.filters.multiecho_thermometry_filter import (
    calculate_df_from_temperature,
    calculate_temperature_from_df,
    calculate_temperature_uncertainty,
    lsq_fit_thermometry_signal_model,
    thermometry_signal_model,
)

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
