"""Tests for V1.6 additions: ALS/arPLS, multi-peak fitting, axis conversion,
and region-weighted similarity."""

from __future__ import annotations

import numpy as np
import pytest

from openspectra_workbench.core.spectrum import Spectrum
from openspectra_workbench.processing.baseline_advanced import (
    baseline_als,
    baseline_arpls,
)
from openspectra_workbench.processing.transforms import (
    HC_NM_EV,
    convert_axis,
    raman_shift,
)
from openspectra_workbench.analysis.fitting import (
    fit_peaks,
    gaussian,
    lorentzian,
    pseudo_voigt,
    voigt,
)
from openspectra_workbench.analysis.region_similarity import (
    region_weighted_similarity,
    FTIR_DEFAULT_REGIONS,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def synthetic_raman(seed: int = 0) -> Spectrum:
    """Three Lorentzian peaks on a curved fluorescence baseline + noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(200, 2000, 1801)
    peaks = (
        5.0 / (1 + ((x - 520) / 8) ** 2)
        + 3.0 / (1 + ((x - 1085) / 15) ** 2)
        + 4.0 / (1 + ((x - 1580) / 20) ** 2)
    )
    fluorescence = 2.0 * np.exp(-((x - 800) / 600) ** 2) + 0.5
    noise = rng.normal(0, 0.05, x.size)
    return Spectrum(
        x=x,
        y=peaks + fluorescence + noise,
        name="synthetic raman",
        technique="Raman",
        x_unit="cm-1",
        y_unit="counts",
    )


# ---------------------------------------------------------------------------
# ALS / arPLS baseline
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_als_removes_curved_fluorescence(self):
        s = synthetic_raman()
        corrected = baseline_als(s, lam=1e6, p=0.001)
        # Off-peak baseline should be near zero after correction
        off_peak = (s.x < 400) | ((s.x > 700) & (s.x < 1000)) | ((s.x > 1700))
        assert abs(np.mean(corrected.y[off_peak])) < 0.3
        # Original baseline shifted away from zero
        assert abs(np.mean(s.y[off_peak])) > 0.5
        assert corrected.metadata["baseline"]["method"] == "als"

    def test_als_preserves_peaks(self):
        s = synthetic_raman()
        corrected = baseline_als(s, lam=1e6, p=0.001)
        # Peak heights should remain large after correction
        for known in (520, 1085, 1580):
            idx = np.argmin(np.abs(s.x - known))
            assert corrected.y[idx] > 1.0, f"peak at {known} cm⁻¹ was destroyed"

    def test_arpls_removes_curved_fluorescence(self):
        s = synthetic_raman()
        corrected = baseline_arpls(s, lam=1e6)
        off_peak = (s.x < 400) | ((s.x > 700) & (s.x < 1000)) | ((s.x > 1700))
        assert abs(np.mean(corrected.y[off_peak])) < 0.3
        assert corrected.metadata["baseline"]["method"] == "arpls"

    def test_als_rejects_bad_p(self):
        s = synthetic_raman()
        with pytest.raises(ValueError):
            baseline_als(s, p=1.5)
        with pytest.raises(ValueError):
            baseline_als(s, p=0.0)

    def test_als_rejects_bad_lam(self):
        s = synthetic_raman()
        with pytest.raises(ValueError):
            baseline_als(s, lam=-1.0)

    def test_baseline_does_not_mutate_input(self):
        s = synthetic_raman()
        y_before = s.y.copy()
        _ = baseline_als(s)
        _ = baseline_arpls(s)
        np.testing.assert_array_equal(s.y, y_before)


# ---------------------------------------------------------------------------
# Axis conversions
# ---------------------------------------------------------------------------

class TestAxisConversion:
    def test_500nm_is_20000_wavenumbers(self):
        s = Spectrum(
            x=np.array([400.0, 500.0, 600.0]),
            y=np.array([1.0, 2.0, 1.0]),
            x_unit="nm",
        )
        converted = convert_axis(s, "cm-1")
        assert converted.x_unit == "cm-1"
        # 500 nm should map to 20000 cm⁻¹
        assert any(abs(converted.x - 20000.0) < 1.0)

    def test_500nm_is_2_48ev(self):
        s = Spectrum(
            x=np.array([400.0, 500.0, 600.0]),
            y=np.array([1.0, 2.0, 1.0]),
            x_unit="nm",
        )
        converted = convert_axis(s, "eV")
        # E (eV) = hc/λ ≈ 1239.84 / 500 ≈ 2.4797
        idx_500 = np.argmin(np.abs(s.x - 500))
        # After conversion the array is sorted ascending; find by value
        target = HC_NM_EV / 500.0
        assert any(abs(converted.x - target) < 1e-6)

    def test_round_trip_preserves_values(self):
        x = np.linspace(400, 700, 50)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm")
        # nm → cm⁻¹ → nm
        back = convert_axis(convert_axis(s, "cm-1"), "nm")
        np.testing.assert_allclose(back.x, x, rtol=1e-9)
        # nm → eV → nm
        back2 = convert_axis(convert_axis(s, "eV"), "nm")
        np.testing.assert_allclose(back2.x, x, rtol=1e-9)
        # nm → THz → nm
        back3 = convert_axis(convert_axis(s, "THz"), "nm")
        np.testing.assert_allclose(back3.x, x, rtol=1e-9)

    def test_conversion_reverses_x_when_needed(self):
        s = Spectrum(x=np.linspace(400, 700, 100), y=np.ones(100), x_unit="nm")
        converted = convert_axis(s, "cm-1")
        # Result must still be sorted ascending
        assert np.all(np.diff(converted.x) > 0)

    def test_unknown_unit_raises(self):
        s = Spectrum(x=np.arange(10.0), y=np.ones(10), x_unit="nm")
        with pytest.raises(ValueError):
            convert_axis(s, "furlongs")

    def test_missing_source_unit_raises(self):
        s = Spectrum(x=np.arange(10.0), y=np.ones(10))  # no x_unit
        with pytest.raises(ValueError):
            convert_axis(s, "cm-1")

    def test_unit_aliases(self):
        s = Spectrum(x=np.arange(400.0, 700.0), y=np.ones(300), x_unit="nanometres")
        converted = convert_axis(s, "wavenumber")
        assert converted.x_unit == "cm-1"

    def test_raman_shift_at_532nm(self):
        # 540 nm scattered with 532 nm laser: shift = 1e7*(1/532 - 1/540) ≈ 278.4 cm⁻¹
        s = Spectrum(
            x=np.array([535.0, 540.0, 545.0]),
            y=np.array([1.0, 2.0, 1.0]),
            x_unit="nm",
        )
        shifted = raman_shift(s, laser_nm=532.0)
        assert shifted.x_unit == "cm-1"
        expected = 1e7 * (1.0 / 532.0 - 1.0 / 540.0)
        assert min(abs(shifted.x - expected)) < 1.0
        assert shifted.metadata["raman_excitation_nm"] == 532.0
        assert shifted.technique == "Raman"

    def test_raman_shift_rejects_wrong_unit(self):
        s = Spectrum(x=np.array([100.0, 200.0]), y=np.ones(2), x_unit="cm-1")
        with pytest.raises(ValueError):
            raman_shift(s, laser_nm=532.0)

    def test_raman_shift_rejects_bad_laser(self):
        s = Spectrum(x=np.array([535.0, 540.0]), y=np.ones(2), x_unit="nm")
        with pytest.raises(ValueError):
            raman_shift(s, laser_nm=-1.0)


# ---------------------------------------------------------------------------
# Multi-peak fitting
# ---------------------------------------------------------------------------

class TestPeakFitting:
    def _three_pseudo_voigt(self):
        x = np.linspace(0, 100, 1001)
        truth = [
            (1.0, 25.0, 4.0, 0.3),
            (0.7, 50.0, 6.0, 0.5),
            (1.2, 75.0, 5.0, 0.7),
        ]
        y = np.zeros_like(x)
        for amp, c, fwhm, eta in truth:
            y += pseudo_voigt(x, amp, c, fwhm, eta)
        rng = np.random.default_rng(42)
        y += rng.normal(0, 0.005, x.size)
        return Spectrum(x=x, y=y, name="three_peaks"), truth

    def test_pseudo_voigt_recovers_three_peaks(self):
        s, truth = self._three_pseudo_voigt()
        result = fit_peaks(
            s,
            x0=[c for _, c, _, _ in truth],
            profile="pseudo_voigt",
            baseline="constant",
        )
        assert result.r_squared > 0.99
        recovered_centers = sorted(p.center for p in result.peaks)
        true_centers = sorted(c for _, c, _, _ in truth)
        for got, want in zip(recovered_centers, true_centers):
            assert abs(got - want) < 0.5
        # Each peak should have finite area and FWHM
        for p in result.peaks:
            assert p.area() > 0
            assert p.fwhm > 0

    def test_auto_seed_finds_correct_count(self):
        s, truth = self._three_pseudo_voigt()
        result = fit_peaks(s, n_peaks=3, profile="gaussian", baseline="constant")
        assert len(result.peaks) == 3
        assert result.r_squared > 0.95

    def test_lorentzian_profile_fits(self):
        # Generate pure Lorentzian; verify fit recovers it
        x = np.linspace(0, 100, 1001)
        y = lorentzian(x, 1.0, 50.0, 5.0)
        rng = np.random.default_rng(0)
        y = y + rng.normal(0, 0.001, x.size)
        s = Spectrum(x=x, y=y, name="single_lorentzian")
        result = fit_peaks(s, x0=[50.0], profile="lorentzian", baseline="none")
        assert abs(result.peaks[0].center - 50.0) < 0.1
        assert abs(result.peaks[0].fwhm - 5.0) < 0.5
        assert abs(result.peaks[0].amplitude - 1.0) < 0.05

    def test_voigt_recovers_known_widths(self):
        x = np.linspace(0, 100, 2001)
        y = voigt(x, 1.0, 50.0, 3.0, 4.0)
        s = Spectrum(x=x, y=y, name="single_voigt")
        result = fit_peaks(s, x0=[50.0], profile="voigt", baseline="none")
        peak = result.peaks[0]
        assert abs(peak.center - 50.0) < 0.1
        # Total FWHM should be larger than either individual width (Olivero-Longbothum)
        assert peak.fwhm > 4.0
        assert peak.fwhm < 8.0

    def test_region_restricts_fit(self):
        s, _ = self._three_pseudo_voigt()
        result = fit_peaks(
            s,
            x0=[50.0],
            profile="pseudo_voigt",
            baseline="constant",
            region=(35.0, 65.0),
        )
        assert len(result.peaks) == 1
        assert result.x_fit.min() >= 35.0
        assert result.x_fit.max() <= 65.0

    def test_invalid_profile_raises(self):
        s, _ = self._three_pseudo_voigt()
        with pytest.raises(ValueError):
            fit_peaks(s, x0=[50.0], profile="banana")

    def test_must_supply_x0_or_n_peaks(self):
        s, _ = self._three_pseudo_voigt()
        with pytest.raises(ValueError):
            fit_peaks(s, profile="gaussian", baseline="none")

    def test_gaussian_area_matches_analytic(self):
        # For a Gaussian with amp=1 and FWHM=2, area = sqrt(π/ln2) ≈ 2.1289
        x = np.linspace(-20, 20, 2001)
        y = gaussian(x, 1.0, 0.0, 2.0)
        s = Spectrum(x=x, y=y, name="g")
        result = fit_peaks(s, x0=[0.0], profile="gaussian", baseline="none")
        analytic = np.sqrt(np.pi / np.log(2.0))
        assert abs(result.peaks[0].area() - analytic) < 0.01


# ---------------------------------------------------------------------------
# Region-weighted similarity
# ---------------------------------------------------------------------------

class TestRegionSimilarity:
    def _ftir_sample(self, fingerprint_amp: float, ch_stretch_amp: float) -> Spectrum:
        x = np.linspace(400, 4000, 1801)
        y = (
            fingerprint_amp * np.exp(-((x - 1100) / 50) ** 2)
            + ch_stretch_amp * np.exp(-((x - 2900) / 30) ** 2)
            + 0.05
        )
        return Spectrum(x=x, y=y, technique="FTIR", x_unit="cm-1", name="sample")

    def test_self_similarity_is_one(self):
        s = self._ftir_sample(1.0, 0.5)
        result = region_weighted_similarity(s, s)
        assert result["final_score"] > 0.95
        assert result["pearson"] > 0.99

    def test_fingerprint_loss_hurts_more_than_ch_stretch_loss(self):
        # Build three samples:
        #   ref: has both a fingerprint peak and a C-H stretch peak
        #   missing_fp:  fingerprint peak gone (drastic shape change, region weight 2.0)
        #   missing_ch:  C-H stretch peak gone (drastic shape change, region weight 0.6)
        # Under FTIR weighting, losing the fingerprint should be penalised more.
        x = np.linspace(400, 4000, 1801)
        peak_fp = np.exp(-((x - 1100) / 50) ** 2)
        peak_ch = np.exp(-((x - 2900) / 30) ** 2)
        ref = Spectrum(x=x, y=peak_fp + peak_ch + 0.05, technique="FTIR",
                       x_unit="cm-1", name="ref")
        missing_fp = Spectrum(x=x, y=peak_ch + 0.05, technique="FTIR",
                              x_unit="cm-1", name="missing_fp")
        missing_ch = Spectrum(x=x, y=peak_fp + 0.05, technique="FTIR",
                              x_unit="cm-1", name="missing_ch")

        # With FTIR-default region weighting, losing the fingerprint should
        # produce a STRICTLY LOWER similarity than losing the C-H stretch.
        sim_no_fp = region_weighted_similarity(ref, missing_fp)["final_score"]
        sim_no_ch = region_weighted_similarity(ref, missing_ch)["final_score"]
        assert sim_no_fp < sim_no_ch, (
            f"weighted: missing-fp ({sim_no_fp:.3f}) should be < missing-ch "
            f"({sim_no_ch:.3f}) under FTIR weighting"
        )

    def test_auto_picks_ftir_defaults(self):
        s = self._ftir_sample(1.0, 0.5)
        result = region_weighted_similarity(s, s)  # auto=True by default
        assert result["regions_used"] == FTIR_DEFAULT_REGIONS

    def test_explicit_empty_regions_disables_weighting(self):
        s = self._ftir_sample(1.0, 0.5)
        result = region_weighted_similarity(s, s, regions=[])
        assert result["regions_used"] == []

    def test_output_schema_matches_v15(self):
        s = self._ftir_sample(1.0, 0.5)
        result = region_weighted_similarity(s, s)
        # Must contain the same keys as V1.5 composite_similarity, plus regions_used
        for key in ("pearson", "cosine", "rmse_score", "peak_score", "final_score"):
            assert key in result
        assert "regions_used" in result
