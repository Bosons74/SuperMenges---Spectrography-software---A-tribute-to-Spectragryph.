"""Tests for V1.9–V1.12 features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openspectra_workbench.core.spectrum import Spectrum
from openspectra_workbench.core.hyperspectral import (
    HyperspectralCube, load_ascii_cube, load_envi,
)
from openspectra_workbench.analysis.chemical_imaging import (
    band_integral_map, peak_height_map,
)
from openspectra_workbench.analysis.clustering import kmeans_cube
from openspectra_workbench.analysis.difference_view import difference_view
from openspectra_workbench.processing.atr import atr_correct
from openspectra_workbench.processing.atmospheric import (
    mask_atmospheric_regions, subtract_atmospheric,
    CO2_REGIONS, H2O_REGIONS,
)
from openspectra_workbench.workflows.macros import Macro, list_available_ops
from openspectra_workbench.workflows.plugins import discover_plugins
from openspectra_workbench.app.themes import THEMES, LIGHT, DARK, apply_theme
from openspectra_workbench.app.preferences import Preferences
from openspectra_workbench.export.templates import (
    DEFAULT_HTML_TEMPLATE, render_template, render_template_file,
)


# ---------------------------------------------------------------------------
# Hyperspectral cube
# ---------------------------------------------------------------------------

def _make_cube(h=8, w=10, l=50, seed=0):
    rng = np.random.default_rng(seed)
    wv = np.linspace(800, 1800, l)
    # Two distinct chemical regions: top-half has a peak at 1000, bottom at 1500
    cube = np.zeros((h, w, l))
    for r in range(h):
        for c in range(w):
            if r < h // 2:
                cube[r, c, :] = np.exp(-((wv - 1000) / 25) ** 2)
            else:
                cube[r, c, :] = np.exp(-((wv - 1500) / 25) ** 2)
            cube[r, c, :] += rng.normal(0, 0.02, l)
    return HyperspectralCube(data=cube, wavelengths=wv, name="test_cube")


class TestHyperspectralCube:
    def test_construction_and_shape(self):
        c = _make_cube()
        assert c.shape == (8, 10, 50)
        assert c.height == 8 and c.width == 10 and c.n_wavelengths == 50

    def test_wavelength_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            HyperspectralCube(data=np.zeros((4, 4, 10)), wavelengths=np.arange(8))

    def test_data_must_be_3d(self):
        with pytest.raises(ValueError):
            HyperspectralCube(data=np.zeros((10,)), wavelengths=np.arange(10))

    def test_spectrum_at_returns_spectrum(self):
        c = _make_cube()
        s = c.spectrum_at(0, 0)
        assert isinstance(s, Spectrum)
        assert len(s.x) == c.n_wavelengths

    def test_spectrum_at_bounds(self):
        c = _make_cube()
        with pytest.raises(IndexError):
            c.spectrum_at(99, 99)

    def test_mean_spectrum_full(self):
        c = _make_cube()
        m = c.mean_spectrum()
        assert len(m.x) == c.n_wavelengths

    def test_mean_spectrum_with_mask(self):
        c = _make_cube()
        mask = np.zeros((c.height, c.width), dtype=bool)
        mask[0:2, :] = True
        m = c.mean_spectrum(mask)
        # Peak should be near 1000 cm⁻¹ for the top region
        peak_idx = int(np.argmax(m.y))
        assert abs(m.x[peak_idx] - 1000) < 50

    def test_mask_shape_validation(self):
        c = _make_cube()
        with pytest.raises(ValueError):
            c.mean_spectrum(mask=np.ones((3, 3), dtype=bool))


class TestAsciiCubeLoader:
    def test_round_trip(self, tmp_path):
        c = _make_cube(h=3, w=4, l=20)
        path = tmp_path / "cube.txt"
        with path.open("w") as f:
            f.write("#wavelengths " + " ".join(f"{w:g}" for w in c.wavelengths) + "\n")
            for r in range(c.height):
                for col in range(c.width):
                    row = [col, r] + list(c.data[r, col])
                    f.write(" ".join(f"{v:g}" for v in row) + "\n")
        loaded = load_ascii_cube(path)
        assert loaded.shape == c.shape
        assert np.allclose(loaded.wavelengths, c.wavelengths)


# ---------------------------------------------------------------------------
# Chemical imaging
# ---------------------------------------------------------------------------

class TestChemicalImaging:
    def test_band_integral_top_vs_bottom(self):
        c = _make_cube()
        # Integrate around 1000: top half should be brighter
        m = band_integral_map(c, 950, 1050)
        assert m.shape == (c.height, c.width)
        assert m[0:c.height // 2, :].mean() > m[c.height // 2:, :].mean()

    def test_band_integral_baseline(self):
        c = _make_cube()
        m1 = band_integral_map(c, 950, 1050, subtract_endpoint_baseline=False)
        m2 = band_integral_map(c, 950, 1050, subtract_endpoint_baseline=True)
        assert m1.shape == m2.shape

    def test_peak_height_map(self):
        c = _make_cube()
        m = peak_height_map(c, 950, 1050)
        assert m.shape == (c.height, c.width)
        assert m[0:c.height // 2, :].mean() > m[c.height // 2:, :].mean()

    def test_invalid_band_raises(self):
        c = _make_cube()
        with pytest.raises(ValueError):
            band_integral_map(c, 1500, 1000)

    def test_band_outside_cube_raises(self):
        c = _make_cube()
        with pytest.raises(ValueError):
            band_integral_map(c, 5000, 6000)


# ---------------------------------------------------------------------------
# k-means clustering
# ---------------------------------------------------------------------------

class TestKMeansClustering:
    def test_two_chemical_regions_separated(self):
        c = _make_cube()
        result = kmeans_cube(c, k=2, normalize=True)
        # The label map should have the top half in one cluster and the
        # bottom half in another (modulo cluster id ordering).
        top_labels = result.labels[0:c.height // 2, :].ravel()
        bottom_labels = result.labels[c.height // 2:, :].ravel()
        # Most pixels in each half should agree on their label
        top_majority = np.bincount(top_labels).max() / top_labels.size
        bot_majority = np.bincount(bottom_labels).max() / bottom_labels.size
        assert top_majority > 0.8 and bot_majority > 0.8
        # The dominant top label should differ from the dominant bottom label
        assert np.argmax(np.bincount(top_labels)) != np.argmax(np.bincount(bottom_labels))

    def test_centroids_shape(self):
        c = _make_cube()
        r = kmeans_cube(c, k=3)
        assert r.centroids.shape == (3, c.n_wavelengths)
        assert r.k == 3

    def test_cluster_mask(self):
        c = _make_cube()
        r = kmeans_cube(c, k=2)
        m = r.cluster_mask(0)
        assert m.dtype == bool
        assert m.shape == r.labels.shape
        assert m.sum() > 0

    def test_k_minimum(self):
        c = _make_cube()
        with pytest.raises(ValueError):
            kmeans_cube(c, k=1)


# ---------------------------------------------------------------------------
# ATR correction
# ---------------------------------------------------------------------------

class TestATR:
    def test_correction_amplifies_low_wavenumbers_less(self):
        # ATR over-amplifies low ν~ (long λ): correction multiplies by ν~/ν~_ref,
        # so values below the median should be scaled DOWN, above should be scaled UP.
        x = np.linspace(500, 4000, 100)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="cm-1", name="flat")
        out = atr_correct(s)
        nu_ref = float(np.median(x))
        # at x ≈ nu_ref, factor is approximately 1 (the nearest sample to the
        # median may not be exactly on it, so allow a small tolerance scaled
        # by the spacing).
        idx_mid = np.argmin(np.abs(x - nu_ref))
        spacing = float(np.median(np.diff(x)))
        tol = 2 * spacing / nu_ref
        assert abs(out.y[idx_mid] - 1.0) < tol
        # at x < nu_ref, factor < 1
        assert out.y[0] < 1.0
        # at x > nu_ref, factor > 1
        assert out.y[-1] > 1.0

    def test_wrong_unit_raises(self):
        s = Spectrum(x=np.linspace(400, 700, 50), y=np.ones(50), x_unit="nm")
        with pytest.raises(ValueError):
            atr_correct(s)

    def test_invalid_angle_raises(self):
        s = Spectrum(x=np.linspace(500, 4000, 50), y=np.ones(50), x_unit="cm-1")
        with pytest.raises(ValueError):
            atr_correct(s, angle_deg=0.0)
        with pytest.raises(ValueError):
            atr_correct(s, angle_deg=90.0)

    def test_n_must_exceed_sample(self):
        s = Spectrum(x=np.linspace(500, 4000, 50), y=np.ones(50), x_unit="cm-1")
        with pytest.raises(ValueError):
            atr_correct(s, crystal_n=1.0, sample_n=1.5)

    def test_n_reflections_divides(self):
        s = Spectrum(x=np.linspace(500, 4000, 50), y=np.ones(50), x_unit="cm-1")
        single = atr_correct(s, n_reflections=1)
        multi = atr_correct(s, n_reflections=5)
        assert np.allclose(multi.y, single.y / 5.0)


# ---------------------------------------------------------------------------
# Atmospheric correction
# ---------------------------------------------------------------------------

class TestAtmospheric:
    def test_subtract_with_perfect_background(self):
        # Sample = sample_band + alpha_true * atm_band ; subtraction should
        # recover sample_band with alpha ≈ alpha_true.
        x = np.linspace(400, 4000, 1801)
        sample_band = np.exp(-((x - 1700) / 30) ** 2)
        atm = np.exp(-((x - 2350) / 15) ** 2) + 0.5 * np.exp(-((x - 670) / 8) ** 2)
        alpha_true = 0.7
        s = Spectrum(x=x, y=sample_band + alpha_true * atm, x_unit="cm-1", name="s")
        bg = Spectrum(x=x, y=atm, x_unit="cm-1", name="bg")
        corrected = subtract_atmospheric(s, bg)
        recovered_alpha = corrected.metadata["atmospheric_correction"]["alpha"]
        assert abs(recovered_alpha - alpha_true) < 0.05
        # Residual should be approximately the sample band alone
        sample_only_idx = np.argmin(np.abs(x - 1700))
        assert abs(corrected.y[sample_only_idx] - 1.0) < 0.05

    def test_mask_replaces_with_interpolation(self):
        x = np.linspace(400, 4000, 1801)
        # Add a sharp CO2-region spike that should be removed by masking
        y = np.ones_like(x)
        co2_mask = (x > 2300) & (x < 2400)
        y[co2_mask] = 5.0
        s = Spectrum(x=x, y=y, x_unit="cm-1", name="s")
        out = mask_atmospheric_regions(s, regions=CO2_REGIONS)
        # In the CO2 window, y should now be near 1 (linearly interpolated)
        assert np.all(out.y[co2_mask] < 2.0)
        assert out.metadata["atmospheric_correction"]["method"] == "mask"

    def test_subtract_wrong_unit_raises(self):
        x = np.linspace(400, 4000, 100)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm")
        bg = Spectrum(x=x, y=np.ones_like(x), x_unit="cm-1")
        with pytest.raises(ValueError):
            subtract_atmospheric(s, bg)


# ---------------------------------------------------------------------------
# A−B difference view
# ---------------------------------------------------------------------------

class TestDifferenceView:
    def test_self_difference_is_zero(self):
        x = np.linspace(400, 700, 500)
        y = np.exp(-((x - 550) / 30) ** 2)
        s = Spectrum(x=x, y=y, x_unit="nm", name="s")
        result = difference_view(s, s)
        assert result.overall_rmse < 1e-9
        assert result.pearson_r > 0.999
        assert np.allclose(result.difference.y, 0.0, atol=1e-9)

    def test_local_rmse_higher_in_region_of_disagreement(self):
        x = np.linspace(400, 700, 500)
        a = Spectrum(x=x, y=np.exp(-((x - 500) / 20) ** 2), x_unit="nm", name="a")
        # b has the peak shifted to 600 — disagreement is concentrated there
        b = Spectrum(x=x, y=np.exp(-((x - 600) / 20) ** 2), x_unit="nm", name="b")
        result = difference_view(a, b)
        rmse = result.local_rmse
        # The local RMSE around 500-600 should be larger than at the edges
        edge = (rmse.x < 420) | (rmse.x > 680)
        middle = (rmse.x > 470) & (rmse.x < 630)
        assert rmse.y[middle].max() > rmse.y[edge].max()


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------

class TestMacros:
    def test_record_and_replay(self, tmp_path):
        x = np.linspace(400, 700, 500)
        s = Spectrum(x=x, y=np.exp(-((x - 550) / 20) ** 2) + 0.05, x_unit="nm",
                     name="raw")
        macro = Macro(name="test", steps=[
            {"op": "smooth_savgol", "args": {"window": 11, "polyorder": 3}},
            {"op": "normalize", "args": {"mode": "max"}},
        ])
        # Save and reload
        path = tmp_path / "m.json"
        macro.save(path)
        loaded = Macro.load(path)
        out = loaded.apply(s)
        assert out.y.max() == pytest.approx(1.0, abs=0.01)

    def test_unknown_op_raises(self):
        x = np.linspace(0, 10, 100)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm")
        m = Macro(name="bad", steps=[{"op": "demolish_universe", "args": {}}])
        with pytest.raises(ValueError):
            m.apply(s)

    def test_list_ops_includes_baselines(self):
        ops = list_available_ops()
        assert "baseline_arpls" in ops
        assert "baseline_als" in ops
        assert "kubelka_munk" in ops
        assert "atr_correct" in ops


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------

class TestPlugins:
    def test_discover_does_not_crash_when_no_plugins(self):
        result = discover_plugins()
        # All three groups should be present, even if empty
        assert "openspectra_workbench.readers" in result
        assert "openspectra_workbench.processors" in result
        assert "openspectra_workbench.exporters" in result


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

class TestThemes:
    def test_themes_registered(self):
        assert "light" in THEMES
        assert "dark" in THEMES

    def test_dark_has_dark_background(self):
        # Eyeball check: dark theme background is darker than light theme
        assert int(LIGHT.background.lstrip("#"), 16) > int(DARK.background.lstrip("#"), 16)

    def test_stylesheet_contains_colours(self):
        ss = DARK.stylesheet()
        assert DARK.background in ss
        assert DARK.foreground in ss


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

class TestPreferences:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "prefs.json"
        prefs = Preferences(theme="dark", default_smoothing_window=21)
        prefs.add_recent("/some/file.csv")
        prefs.add_recent("/another.csv")
        prefs.save(path)
        loaded = Preferences.load(path)
        assert loaded.theme == "dark"
        assert loaded.default_smoothing_window == 21
        assert loaded.recent_files == ["/another.csv", "/some/file.csv"]

    def test_load_missing_returns_defaults(self, tmp_path):
        prefs = Preferences.load(tmp_path / "nonexistent.json")
        assert prefs.theme == "light"

    def test_add_recent_dedups(self):
        prefs = Preferences()
        prefs.add_recent("a.csv")
        prefs.add_recent("b.csv")
        prefs.add_recent("a.csv")           # Should move to front, not duplicate
        assert prefs.recent_files == ["a.csv", "b.csv"]

    def test_recent_files_capped(self):
        prefs = Preferences()
        for i in range(20):
            prefs.add_recent(f"file_{i}.csv", max_items=5)
        assert len(prefs.recent_files) == 5


# ---------------------------------------------------------------------------
# Report templates
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_render_basic_placeholders(self):
        x = np.linspace(400, 700, 50)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm", name="alpha",
                     technique="UV-Vis")
        out = render_template("Hello {name} ({technique})", s)
        assert out == "Hello alpha (UV-Vis)"

    def test_unknown_placeholder_left_verbatim(self):
        x = np.linspace(0, 10, 30)
        s = Spectrum(x=x, y=np.ones_like(x), x_unit="nm", name="x")
        out = render_template("{name} {totally_unknown}", s)
        assert "{totally_unknown}" in out
        assert out.startswith("x ")

    def test_peak_table_html(self):
        x = np.linspace(400, 700, 1000)
        y = np.exp(-((x - 500) / 10) ** 2) + np.exp(-((x - 600) / 10) ** 2)
        s = Spectrum(x=x, y=y, x_unit="nm", name="two_peaks")
        out = render_template("{peak_table:html}", s)
        assert "<table>" in out
        assert "<tr>" in out

    def test_default_template_renders(self, tmp_path):
        x = np.linspace(400, 700, 1000)
        y = np.exp(-((x - 500) / 10) ** 2)
        s = Spectrum(x=x, y=y, x_unit="nm", name="single_peak", technique="UV-Vis")
        tpl = tmp_path / "t.html"
        tpl.write_text(DEFAULT_HTML_TEMPLATE)
        out_file = tmp_path / "out.html"
        render_template_file(tpl, s, out_file)
        out = out_file.read_text()
        assert "<html>" in out
        assert "single_peak" in out
        assert "UV-Vis" in out
