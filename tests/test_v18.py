"""Tests for V1.8 quantification: Beer-Lambert, PLS, Kubelka-Munk, CIE color."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openspectra_workbench.core.spectrum import Spectrum
from openspectra_workbench.analysis.quantification import (
    BeerLambertCalibration,
    build_calibration,
)
from openspectra_workbench.analysis.pls import PLSModel, train_pls
from openspectra_workbench.processing.kubelka_munk import (
    inverse_kubelka_munk,
    kubelka_munk,
)
from openspectra_workbench.analysis.color import (
    CIEColor,
    spectrum_to_color,
)


# =========================================================================
# Beer-Lambert
# =========================================================================

def _make_calibration_set(true_eps=0.5, true_b=0.05, noise_sd=0.001, seed=0):
    """Build a 6-point calibration with a single Gaussian band at 550 nm.

    response = ε · c + b, where ε and b are the values we want the fit to recover.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(400, 700, 601)
    band = np.exp(-((x - 550) / 15) ** 2)
    standards = []
    concentrations = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for c in concentrations:
        y = (true_eps * c + true_b) * band + rng.normal(0, noise_sd, x.size)
        standards.append((c, Spectrum(x=x, y=y, x_unit="nm",
                                      y_unit="absorbance", name=f"std c={c}")))
    return standards, true_eps, true_b


class TestBeerLambert:
    def test_recovers_known_slope_and_intercept(self):
        standards, eps, b0 = _make_calibration_set()
        cal = build_calibration(standards, wavelength=550, mode="height")
        # The peak of the band is exp(0) = 1, so response ≈ eps*c + b0
        assert abs(cal.slope - eps) < 0.01
        assert abs(cal.intercept - b0) < 0.01
        assert cal.r_squared > 0.999

    def test_lod_loq_ratios_per_ich(self):
        standards, _, _ = _make_calibration_set()
        cal = build_calibration(standards, wavelength=550)
        assert cal.loq / cal.lod == pytest.approx(10.0 / 3.3, rel=1e-9)

    def test_predict_returns_input_concentration(self):
        standards, _, _ = _make_calibration_set()
        cal = build_calibration(standards, wavelength=550)
        # Predict each training spectrum and check it matches the known concentration
        for true_c, spec in standards:
            pred = cal.predict(spec)
            assert abs(pred - true_c) < 0.05

    def test_predict_with_uncertainty(self):
        standards, _, _ = _make_calibration_set()
        cal = build_calibration(standards, wavelength=550)
        c, sigma = cal.predict_with_uncertainty(standards[2][1])
        assert sigma > 0
        assert abs(c - standards[2][0]) < 3 * sigma  # within 3σ

    def test_area_mode(self):
        standards, _, _ = _make_calibration_set()
        cal = build_calibration(
            standards, wavelength=550, mode="area", band_half_width=30
        )
        assert cal.r_squared > 0.99
        # Slope in area mode should differ from height mode, but R² still strong
        cal_h = build_calibration(standards, wavelength=550, mode="height")
        assert abs(cal.slope) > abs(cal_h.slope)  # area integrates ~ √π · σ ≈ 26.6 wider

    def test_area_requires_half_width(self):
        standards, _, _ = _make_calibration_set()
        with pytest.raises(ValueError):
            build_calibration(standards, wavelength=550, mode="area")

    def test_too_few_standards_raises(self):
        rng = np.random.default_rng(0)
        x = np.linspace(400, 700, 100)
        s = Spectrum(x=x, y=np.exp(-((x - 550) / 20) ** 2), x_unit="nm")
        with pytest.raises(ValueError):
            build_calibration([(1.0, s)], wavelength=550)


# =========================================================================
# PLS regression
# =========================================================================

def _make_pls_set(n_samples=30, seed=42):
    """Two analytes contributing distinct bands; PLS should disentangle them."""
    rng = np.random.default_rng(seed)
    x = np.linspace(1100, 2200, 551)  # NIR range, nm
    band_a = np.exp(-((x - 1450) / 30) ** 2)   # analyte A
    band_b = np.exp(-((x - 1940) / 40) ** 2)   # analyte B
    spectra = []
    targets = []
    for _ in range(n_samples):
        ca, cb = rng.uniform(0.1, 5.0, 2)
        y = ca * band_a + cb * band_b + rng.normal(0, 0.002, x.size)
        spectra.append(Spectrum(x=x, y=y, x_unit="nm", y_unit="absorbance"))
        targets.append([ca, cb])
    return spectra, np.array(targets)


class TestPLS:
    def test_two_analyte_recovery(self):
        spectra, targets = _make_pls_set()
        model = train_pls(spectra, targets, target_names=["A", "B"], max_components=8)
        assert model.r2_cv[0] > 0.95
        assert model.r2_cv[1] > 0.95
        assert 1 <= model.n_components <= 8

    def test_predict_individual_spectrum(self):
        spectra, targets = _make_pls_set()
        model = train_pls(spectra, targets, target_names=["A", "B"], max_components=8)
        pred = model.predict(spectra[0])
        assert "A" in pred and "B" in pred
        assert abs(pred["A"] - targets[0, 0]) < 0.5
        assert abs(pred["B"] - targets[0, 1]) < 0.5

    def test_save_and_load_roundtrip(self, tmp_path):
        spectra, targets = _make_pls_set()
        model = train_pls(spectra, targets, target_names=["A", "B"], max_components=5)
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = PLSModel.load(path)
        # Loaded model produces identical predictions
        a = model.predict(spectra[0])
        b = loaded.predict(spectra[0])
        assert a == pytest.approx(b)

    def test_single_target(self):
        spectra, targets = _make_pls_set()
        # Pick only the first column
        model = train_pls(spectra, targets[:, 0], max_components=5)
        assert model.r2_cv[0] > 0.95
        pred = model.predict(spectra[0])
        assert "target_1" in pred

    def test_target_count_mismatch_raises(self):
        spectra, targets = _make_pls_set()
        with pytest.raises(ValueError):
            train_pls(spectra, targets[:5], target_names=["A", "B"])

    def test_too_few_spectra_raises(self):
        spectra, targets = _make_pls_set(n_samples=3)
        with pytest.raises(ValueError):
            train_pls(spectra, targets[:3])


# =========================================================================
# Kubelka-Munk
# =========================================================================

class TestKubelkaMunk:
    def test_R_equals_1_gives_F_equals_0(self):
        x = np.linspace(400, 700, 100)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm", y_unit="reflectance")
        out = kubelka_munk(s)
        # F(1) = 0
        assert np.allclose(out.y, 0.0, atol=1e-6)
        assert out.y_unit == "K-M (k/s)"

    def test_R_equals_half_gives_F_equals_one_quarter(self):
        # F(0.5) = (0.5)² / (2*0.5) = 0.25 / 1 = 0.25
        x = np.linspace(400, 700, 100)
        s = Spectrum(x=x, y=0.5 * np.ones_like(x), x_unit="nm", y_unit="R")
        out = kubelka_munk(s)
        assert np.allclose(out.y, 0.25, atol=1e-6)

    def test_inverse_recovers_input(self):
        rng = np.random.default_rng(0)
        x = np.linspace(400, 700, 200)
        R = rng.uniform(0.05, 0.95, x.size)
        s = Spectrum(x=x, y=R, x_unit="nm", y_unit="reflectance")
        F = kubelka_munk(s)
        back = inverse_kubelka_munk(F)
        assert np.allclose(back.y, R, atol=1e-6)

    def test_percent_input_autoscaled(self):
        x = np.linspace(400, 700, 100)
        R_percent = 50.0 * np.ones_like(x)        # i.e. 50%
        R_fraction = 0.50 * np.ones_like(x)
        a = kubelka_munk(Spectrum(x=x, y=R_percent, x_unit="nm"))
        b = kubelka_munk(Spectrum(x=x, y=R_fraction, x_unit="nm"))
        assert np.allclose(a.y, b.y, atol=1e-6)

    def test_strict_mode_rejects_out_of_range(self):
        x = np.linspace(400, 700, 100)
        s = Spectrum(x=x, y=50.0 * np.ones_like(x), x_unit="nm")
        with pytest.raises(ValueError):
            kubelka_munk(s, autoscale_percent=False)

    def test_inverse_rejects_negative(self):
        x = np.linspace(400, 700, 100)
        s = Spectrum(x=x, y=-0.1 * np.ones_like(x), x_unit="nm")
        with pytest.raises(ValueError):
            inverse_kubelka_munk(s)


# =========================================================================
# CIE color
# =========================================================================

class TestCIEColor:
    def test_perfect_white_under_d65_gives_L100_a0_b0(self):
        # Perfect reflector (R = 1 everywhere) under D65 should produce
        # L* = 100, a* = b* = 0 (within rounding from 10 nm sampling).
        x = np.linspace(380, 780, 81)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm", y_unit="reflectance")
        c = spectrum_to_color(s, illuminant="D65")
        assert c.L == pytest.approx(100.0, abs=0.5)
        assert abs(c.a) < 0.5
        assert abs(c.b) < 0.5
        # Y of perfect white under D65 = 100 by definition
        assert c.Y == pytest.approx(100.0, abs=0.1)

    def test_perfect_black_gives_L0(self):
        x = np.linspace(380, 780, 81)
        s = Spectrum(x=x, y=np.zeros_like(x), x_unit="nm")
        c = spectrum_to_color(s, illuminant="D65")
        assert c.L == pytest.approx(0.0, abs=0.5)
        assert c.X == pytest.approx(0.0, abs=0.01)
        assert c.Y == pytest.approx(0.0, abs=0.01)
        assert c.Z == pytest.approx(0.0, abs=0.01)

    def test_d65_chromaticity_of_white_is_d65(self):
        # The chromaticity (x, y) of a perfect white under D65 must equal
        # the D65 white point: (0.31271, 0.32902) in CIE 015:2018.
        x = np.linspace(380, 780, 81)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm")
        c = spectrum_to_color(s, illuminant="D65")
        assert c.x == pytest.approx(0.31271, abs=0.005)
        assert c.y == pytest.approx(0.32902, abs=0.005)

    def test_red_filter_has_positive_a(self):
        # Spectrum that transmits only red wavelengths should be red:
        # positive a* (red-green axis) and small b*.
        x = np.linspace(380, 780, 401)
        y = np.where((x > 600) & (x < 700), 1.0, 0.0)
        s = Spectrum(x=x, y=y, x_unit="nm")
        c = spectrum_to_color(s)
        assert c.a > 30
        assert c.X > c.Z   # red has X >> Z

    def test_green_filter_has_negative_a(self):
        x = np.linspace(380, 780, 401)
        y = np.where((x > 500) & (x < 580), 1.0, 0.0)
        s = Spectrum(x=x, y=y, x_unit="nm")
        c = spectrum_to_color(s)
        assert c.a < -20

    def test_blue_filter_has_negative_b(self):
        x = np.linspace(380, 780, 401)
        y = np.where((x > 420) & (x < 480), 1.0, 0.0)
        s = Spectrum(x=x, y=y, x_unit="nm")
        c = spectrum_to_color(s)
        assert c.b < -20
        assert c.Z > c.X

    def test_unknown_illuminant_raises(self):
        x = np.linspace(380, 780, 81)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm")
        with pytest.raises(ValueError):
            spectrum_to_color(s, illuminant="F99")

    def test_wrong_x_unit_raises(self):
        x = np.linspace(380, 780, 81)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="cm-1")
        with pytest.raises(ValueError):
            spectrum_to_color(s)

    def test_illuminant_a_yields_warmer_color(self):
        # Perfect white under illuminant A (~2856 K, incandescent) should have
        # positive b* (yellower) compared to D65 (more neutral).
        x = np.linspace(380, 780, 81)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm")
        d65 = spectrum_to_color(s, illuminant="D65")
        a = spectrum_to_color(s, illuminant="A")
        # Both should be ~white under their own illuminant, so b* close to zero
        # (absolute Lab is computed against the illuminant's white point), so
        # we check XYZ instead: under A, X/Y > X/Y under D65 (warmer light).
        assert (a.X / a.Y) > (d65.X / d65.Y)
