"""Tests for ADC quantification filter functions."""

import numpy as np
import pytest
from numpy.typing import NDArray

from mrimagetools.filters.adc_quantification_filter import (
    ADCVolume,
    calculate_r_squared,
    fit_iwlls,
    fit_lls,
    fit_wlls2,
    process_dwi_volume,
)


class TestADCFitting:
    """Test suite for ADC fitting functions."""

    @pytest.fixture
    def synthetic_signals(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        """Generate synthetic DWI signals with known ADC and S0.

        Returns:
            signals, b_values, true_adc, true_s0
        """
        true_adc = 0.001  # mm²/s
        true_s0 = 1000.0
        b_values = np.array([0, 200, 400, 600, 800, 1000], dtype=np.float64)

        # Generate clean signals
        signals = true_s0 * np.exp(-b_values * true_adc)

        return signals, b_values, true_adc, true_s0

    @pytest.fixture
    def noisy_signals(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        """Generate noisy synthetic DWI signals.

        Returns:
            signals, b_values, true_adc, true_s0
        """
        true_adc = 0.001  # mm²/s
        true_s0 = 1000.0
        b_values = np.array([0, 200, 400, 600, 800, 1000], dtype=np.float64)

        # Generate clean signals
        clean_signals = true_s0 * np.exp(-b_values * true_adc)

        # Add Rician noise (SNR ~ 20)
        np.random.seed(42)
        noise_level = true_s0 / 20
        noise_real = np.random.normal(0, noise_level, size=clean_signals.shape)
        noise_imag = np.random.normal(0, noise_level, size=clean_signals.shape)
        signals = np.sqrt((clean_signals + noise_real) ** 2 + noise_imag**2)

        return signals, b_values, true_adc, true_s0

    def test_calculate_r_squared(self) -> None:
        """Test R-squared calculation."""
        # Perfect fit
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = observed.copy()
        r2 = calculate_r_squared(observed, predicted)
        assert np.isclose(r2, 1.0)

        # No correlation
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Mean value
        r2 = calculate_r_squared(observed, predicted)
        assert np.isclose(r2, 0.0)

        # Negative R² (worse than mean)
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed
        r2 = calculate_r_squared(observed, predicted)
        assert r2 < 0

    def test_fit_lls_clean(self, synthetic_signals) -> None:
        """Test LLS fitting on clean synthetic data."""
        signals, b_values, true_adc, true_s0 = synthetic_signals

        result = fit_lls(signals, b_values)

        # Should recover parameters almost perfectly for clean data
        assert np.isclose(result.adc, true_adc, rtol=1e-6)
        assert np.isclose(result.s0, true_s0, rtol=1e-6)
        assert np.isclose(result.r_squared, 1.0, rtol=1e-6)

    def test_fit_lls_noisy(self, noisy_signals) -> None:
        """Test LLS fitting on noisy synthetic data."""
        signals, b_values, true_adc, true_s0 = noisy_signals

        result = fit_lls(signals, b_values)

        # Should recover parameters within reasonable error for noisy data
        assert np.isclose(result.adc, true_adc, rtol=0.1)
        assert np.isclose(result.s0, true_s0, rtol=0.1)
        assert result.r_squared > 0.95  # Still good fit despite noise

    def test_fit_wlls2_clean(self, synthetic_signals) -> None:
        """Test WLLS2 fitting on clean synthetic data."""
        signals, b_values, true_adc, true_s0 = synthetic_signals

        result = fit_wlls2(signals, b_values)

        # Should recover parameters almost perfectly for clean data
        assert np.isclose(result.adc, true_adc, rtol=1e-6)
        assert np.isclose(result.s0, true_s0, rtol=1e-6)
        assert np.isclose(result.r_squared, 1.0, rtol=1e-6)

    def test_fit_wlls2_noisy(self, noisy_signals) -> None:
        """Test WLLS2 fitting on noisy synthetic data."""
        signals, b_values, true_adc, true_s0 = noisy_signals

        result = fit_wlls2(signals, b_values)

        # WLLS2 should perform better than LLS on noisy data
        assert np.isclose(result.adc, true_adc, rtol=0.1)
        assert np.isclose(result.s0, true_s0, rtol=0.1)
        assert result.r_squared > 0.95

    def test_fit_iwlls_clean(self, synthetic_signals) -> None:
        """Test iterative WLLS fitting on clean synthetic data."""
        signals, b_values, true_adc, true_s0 = synthetic_signals

        result = fit_iwlls(signals, b_values)

        # Should converge quickly for clean data
        assert result.iterations <= 3
        assert np.isclose(result.result.adc, true_adc, rtol=1e-6)
        assert np.isclose(result.result.s0, true_s0, rtol=1e-6)
        assert np.isclose(result.result.r_squared, 1.0, rtol=1e-6)

    def test_fit_iwlls_convergence(self, noisy_signals) -> None:
        """Test iterative WLLS convergence behavior."""
        signals, b_values, true_adc, true_s0 = noisy_signals

        # Test with different max iterations
        result_5 = fit_iwlls(signals, b_values, max_iterations=5)
        result_10 = fit_iwlls(signals, b_values, max_iterations=10)

        # Results should be very similar (converged)
        assert np.isclose(result_5.result.adc, result_10.result.adc, rtol=1e-4)
        assert result_5.iterations <= 5
        assert result_10.iterations <= 10

    def test_edge_case_zero_signals(self) -> None:
        """Test handling of zero or negative signals."""
        b_values = np.array([0, 500, 1000], dtype=np.float64)

        # All zeros
        signals = np.zeros(3, dtype=np.float64)
        result = fit_lls(signals, b_values)
        assert result.adc == 0.0
        assert result.s0 == 0.0
        assert result.r_squared == 0.0

        # Some negative values (should be filtered out)
        signals = np.array([100.0, -50.0, 25.0], dtype=np.float64)
        result = fit_lls(signals, b_values)
        assert result.adc > 0  # Should still compute something from valid signals

    def test_edge_case_insufficient_data(self) -> None:
        """Test handling of insufficient data points."""
        # Only one valid signal
        signals = np.array([100.0], dtype=np.float64)
        b_values = np.array([0], dtype=np.float64)

        result = fit_lls(signals, b_values)
        assert result.adc == 0.0
        assert result.s0 == 0.0
        assert result.r_squared == 0.0

    def test_method_comparison(self, noisy_signals) -> None:
        """Compare different fitting methods on the same data."""
        signals, b_values, true_adc, true_s0 = noisy_signals

        lls_result = fit_lls(signals, b_values)
        wlls2_result = fit_wlls2(signals, b_values)
        iwlls_result = fit_iwlls(signals, b_values)

        # All methods should give similar results
        assert np.isclose(lls_result.adc, wlls2_result.adc, rtol=0.2)
        assert np.isclose(wlls2_result.adc, iwlls_result.result.adc, rtol=0.1)

        # WLLS methods should generally have better R² than LLS for noisy data
        assert (
            wlls2_result.r_squared >= lls_result.r_squared - 0.01
        )  # Allow small tolerance
        assert iwlls_result.result.r_squared >= lls_result.r_squared - 0.01


class TestVolumeProcessing:
    """Test suite for volume processing functions."""

    @pytest.fixture
    def synthetic_volume(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Generate a small synthetic DWI volume.

        Returns:
            dwi_4d, b_values, true_adc_map
        """
        # Create a small 3x3x2 volume with 4 b-values
        nx, ny, nz = 3, 3, 2
        b_values = np.array([0, 300, 600, 1000], dtype=np.float64)

        # Create ADC map with spatial variation
        true_adc_map = np.zeros((nx, ny, nz), dtype=np.float64)
        true_adc_map[0, :, :] = 0.0008  # Low ADC
        true_adc_map[1, :, :] = 0.0012  # Medium ADC
        true_adc_map[2, :, :] = 0.0015  # High ADC

        # Generate DWI signals
        s0 = 1000.0
        dwi_4d = np.zeros((nx, ny, nz, len(b_values)), dtype=np.float64)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    adc = true_adc_map[i, j, k]
                    dwi_4d[i, j, k, :] = s0 * np.exp(-b_values * adc)

        return dwi_4d, b_values, true_adc_map

    def test_process_volume_lls(self, synthetic_volume) -> None:
        """Test volume processing with LLS method."""
        dwi_4d, b_values, true_adc_map = synthetic_volume

        result = process_dwi_volume(dwi_4d, b_values, method="lls")

        assert isinstance(result, ADCVolume)
        assert result.adc_map.shape == true_adc_map.shape
        assert result.s0_map.shape == true_adc_map.shape
        assert result.r2_map.shape == true_adc_map.shape

        # Check ADC values (converted to × 10⁻³ mm²/s in output)
        np.testing.assert_allclose(result.adc_map / 1000, true_adc_map, rtol=1e-5)

        # Check R² values (should be perfect for clean data)
        assert np.all(result.r2_map > 0.999)

    def test_process_volume_wlls2(self, synthetic_volume) -> None:
        """Test volume processing with WLLS2 method."""
        dwi_4d, b_values, true_adc_map = synthetic_volume

        result = process_dwi_volume(dwi_4d, b_values, method="wlls2")

        assert isinstance(result, ADCVolume)

        # Check ADC values
        np.testing.assert_allclose(result.adc_map / 1000, true_adc_map, rtol=1e-5)

    def test_process_volume_iwlls(self, synthetic_volume) -> None:
        """Test volume processing with iterative WLLS method."""
        dwi_4d, b_values, true_adc_map = synthetic_volume

        result = process_dwi_volume(dwi_4d, b_values, method="iwlls")

        assert isinstance(result, ADCVolume)

        # Check ADC values
        np.testing.assert_allclose(result.adc_map / 1000, true_adc_map, rtol=1e-5)

    def test_process_volume_with_mask(self, synthetic_volume) -> None:
        """Test volume processing with a binary mask."""
        dwi_4d, b_values, _ = synthetic_volume
        nx, ny, nz, _ = dwi_4d.shape

        # Create a mask that excludes half the volume
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[:, : ny // 2, :] = True

        result = process_dwi_volume(dwi_4d, b_values, mask=mask, method="lls")

        # Masked voxels should have zero values
        assert np.all(result.adc_map[~mask] == 0)
        assert np.all(result.s0_map[~mask] == 0)
        assert np.all(result.r2_map[~mask] == 0)

        # Unmasked voxels should have non-zero values
        assert np.all(result.adc_map[mask] > 0)
        assert np.all(result.s0_map[mask] > 0)

    def test_process_volume_invalid_method(self, synthetic_volume) -> None:
        """Test error handling for invalid fitting method."""
        dwi_4d, b_values, _ = synthetic_volume

        with pytest.raises(ValueError, match="Unknown method"):
            process_dwi_volume(dwi_4d, b_values, method="invalid")  # type: ignore

    def test_clinical_units_conversion(self, synthetic_volume) -> None:
        """Test that ADC values are correctly converted to clinical units."""
        dwi_4d, b_values, true_adc_map = synthetic_volume

        result = process_dwi_volume(dwi_4d, b_values, method="lls")

        # ADC should be in × 10⁻³ mm²/s (clinical units)
        # true_adc_map is in mm²/s, so multiply by 1000
        expected_clinical = true_adc_map * 1000

        np.testing.assert_allclose(result.adc_map, expected_clinical, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
