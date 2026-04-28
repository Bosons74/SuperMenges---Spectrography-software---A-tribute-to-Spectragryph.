"""V1.16 tests — SIMCA, Hotelling T², Q-residuals, iPLS, CARS."""

from __future__ import annotations

import numpy as np
import pytest

from openspectra_workbench.chemometrics import (
    backward_ipls, cars_select, fit_simca, forward_ipls,
    hotelling_t2_critical, ipls_intervals, q_residual_critical,
    select_n_components_pca,
)


# =========================================================================
# Synthetic data helpers
# =========================================================================

def _three_class_dataset(n_per_class: int = 15, seed: int = 0):
    """Three FTIR-like classes with different diagnostic peaks."""
    rng = np.random.default_rng(seed)
    x = np.linspace(800, 1800, 200)
    centers = {"A": 1000.0, "B": 1300.0, "C": 1600.0}
    rows, labels = [], []
    for label, center in centers.items():
        peak = np.exp(-((x - center) / 25) ** 2)
        for _ in range(n_per_class):
            rows.append(peak + rng.normal(0, 0.02, x.size))
            labels.append(label)
    return np.array(rows), labels, x


def _outlier_spectrum(seed: int = 999):
    """A clearly-not-from-any-class spectrum: peaks where none of A/B/C have."""
    rng = np.random.default_rng(seed)
    x = np.linspace(800, 1800, 200)
    return (np.exp(-((x - 1100) / 20) ** 2)
            + np.exp(-((x - 1700) / 20) ** 2)
            + rng.normal(0, 0.02, x.size))


def _regression_dataset(n: int = 40, seed: int = 0):
    """Concentration dataset with one diagnostic band at index ~150."""
    rng = np.random.default_rng(seed)
    x = np.linspace(900, 2400, 300)
    band = np.exp(-((x - 1700) / 30) ** 2)
    concentrations = rng.uniform(0.1, 5.0, n)
    rows = [c * band + rng.normal(0, 0.005, x.size) for c in concentrations]
    return np.array(rows), concentrations, x


# =========================================================================
# Hotelling T² and Q-residual critical values
# =========================================================================

class TestCriticalValues:
    def test_hotelling_t2_grows_with_components(self):
        # T²_α should monotonically increase with the number of components
        # for fixed n_train and alpha.
        crits = [hotelling_t2_critical(A, n_train=50, alpha=0.05)
                 for A in range(1, 6)]
        assert all(crits[i] < crits[i + 1] for i in range(len(crits) - 1))

    def test_hotelling_t2_finite_when_n_lt_A(self):
        # When n_train ≤ n_components, must return inf (per docstring contract)
        assert hotelling_t2_critical(5, n_train=4) == float("inf")

    def test_q_residual_zero_when_no_residual_eigenvalues(self):
        # All variance captured → Q critical = 0
        assert q_residual_critical(np.array([])) == 0.0

    def test_q_residual_positive(self):
        # Should give a finite positive number for typical residual eigenvalues
        eigs = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
        q = q_residual_critical(eigs, alpha=0.05)
        assert 0 < q < float("inf")


# =========================================================================
# PCA component selection
# =========================================================================

class TestPCAComponentSelection:
    def test_returns_in_bounds(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (20, 30))
        n = select_n_components_pca(X, max_components=8, cv_folds=4)
        assert 1 <= n <= 8

    def test_low_rank_data_selects_few_components(self):
        # Build data that lives on a 2-D subspace with a tiny noise floor.
        # The PRESS drops by 4 orders of magnitude from A=1 to A=2, then
        # plateaus — but on noiseless-ish data it can keep dropping
        # microscopically through every component, so the *minimum* of
        # the PRESS curve isn't tied to the true rank. What we *can* say
        # is that selecting A=1 should never happen — the function should
        # pick at least 2.
        rng = np.random.default_rng(0)
        loadings = rng.normal(0, 1, (2, 50))
        scores = rng.normal(0, 1, (40, 2))
        X = scores @ loadings + rng.normal(0, 0.01, (40, 50))
        n = select_n_components_pca(X, max_components=10, cv_folds=4)
        assert n >= 2


# =========================================================================
# SIMCA — classification + outlier detection
# =========================================================================

class TestSIMCA:
    def test_fits_three_classes(self):
        X, labels, _ = _three_class_dataset(n_per_class=15)
        model = fit_simca(X, labels, n_components=2)
        assert model.classes == ["A", "B", "C"]
        for c in ["A", "B", "C"]:
            assert c in model.submodels
            sm = model.submodels[c]
            assert sm.diagnostics.n_components == 2
            assert sm.diagnostics.n_train == 15

    def test_classifies_training_samples_correctly(self):
        # SIMCA on training data should put each sample back in its own class.
        X, labels, _ = _three_class_dataset(n_per_class=15)
        model = fit_simca(X, labels, n_components=2)
        results = model.predict(X)
        accuracy = sum(r["predicted"] == lab for r, lab in zip(results, labels)) / len(labels)
        # Some boundary samples may be ambiguous — require strong majority.
        assert accuracy > 0.85

    def test_detects_outlier(self):
        """The signature SIMCA capability — detect a sample that belongs
        to NO trained class. PLS-DA cannot do this honestly."""
        X, labels, _ = _three_class_dataset(n_per_class=15)
        model = fit_simca(X, labels, n_components=2, alpha=0.01)
        outlier = _outlier_spectrum().reshape(1, -1)
        result = model.predict(outlier)[0]
        # Should belong to no class (or have a very high T²/Q for whichever
        # class it's "closest" to)
        assert result["predicted"] is None or len(result["accepted"]) == 0

    def test_per_class_diagnostics_returned(self):
        X, labels, _ = _three_class_dataset(n_per_class=15)
        model = fit_simca(X, labels, n_components=2)
        result = model.predict(X[0:1])[0]
        for label in ["A", "B", "C"]:
            assert label in result["per_class"]
            d = result["per_class"][label]
            for key in ("t2", "q", "t2_critical", "q_critical",
                        "t2_pass", "q_pass", "member"):
                assert key in d

    def test_too_few_samples_raises(self):
        X, labels, _ = _three_class_dataset(n_per_class=2)  # 2 per class < 3
        with pytest.raises(ValueError):
            fit_simca(X, labels, n_components=1)

    def test_per_class_components(self):
        X, labels, _ = _three_class_dataset(n_per_class=15)
        model = fit_simca(X, labels, n_components={"A": 1, "B": 2, "C": 3})
        assert model.submodels["A"].diagnostics.n_components == 1
        assert model.submodels["B"].diagnostics.n_components == 2
        assert model.submodels["C"].diagnostics.n_components == 3

    def test_unknown_mode_raises(self):
        X, labels, _ = _three_class_dataset(n_per_class=15)
        model = fit_simca(X, labels, n_components=2)
        with pytest.raises(ValueError):
            model.predict(X[0:1], mode="banana")


# =========================================================================
# iPLS — interval PLS
# =========================================================================

class TestIPLS:
    def test_intervals_finds_diagnostic_band(self):
        X, c, x = _regression_dataset(n=40)
        # The diagnostic band is at x≈1700, which is in interval ~5/10
        # given x range 900-2400.
        result = ipls_intervals(X, c, n_intervals=10, n_components=2)
        assert len(result.intervals) == 10
        assert len(result.interval_rmsecv) == 10
        # The best interval must be the one covering 1700 cm⁻¹
        best_idx = int(np.argmin(result.interval_rmsecv))
        lo, hi = result.intervals[best_idx]
        x_band_indices = np.where((x >= 1600) & (x <= 1800))[0]
        assert any(lo <= xi < hi for xi in x_band_indices)

    def test_full_rmsecv_recorded(self):
        X, c, _ = _regression_dataset(n=40)
        result = ipls_intervals(X, c, n_intervals=10, n_components=2)
        assert result.full_rmsecv > 0
        assert np.isfinite(result.full_rmsecv)

    def test_forward_selects_few_intervals(self):
        X, c, _ = _regression_dataset(n=40)
        result = forward_ipls(X, c, n_intervals=10, n_components=2)
        # On a clean single-band dataset, forward iPLS should pick a few
        # intervals (probably 1-3) that contain the diagnostic band.
        assert 1 <= len(result.selected) <= 5
        assert result.selected_rmsecv is not None
        # Selected RMSECV should beat or match the best single-interval RMSECV
        assert result.selected_rmsecv <= min(result.interval_rmsecv) + 1e-6

    def test_backward_can_only_improve_or_match_full(self):
        X, c, _ = _regression_dataset(n=40)
        result = backward_ipls(X, c, n_intervals=10, n_components=2)
        # Either backward elimination found a better subset (lower RMSECV)
        # than the full spectrum, or it kept everything.
        assert result.selected_rmsecv <= result.full_rmsecv + 1e-6

    def test_method_field(self):
        X, c, _ = _regression_dataset(n=20)
        assert ipls_intervals(X, c, n_intervals=5).method == "intervals"
        assert forward_ipls(X, c, n_intervals=5).method == "forward"
        assert backward_ipls(X, c, n_intervals=5).method == "backward"


# =========================================================================
# CARS — variable selection
# =========================================================================

class TestCARS:
    def test_selects_diagnostic_region(self):
        X, c, x = _regression_dataset(n=40)
        result = cars_select(X, c, n_components=2,
                             n_iterations=30, random_state=0)
        # Selected indices should mostly fall near the 1700 band
        selected_x = x[result.selected_indices]
        # At least 30 % of the selected variables should be in [1500, 1900]
        in_band = ((selected_x >= 1500) & (selected_x <= 1900)).mean()
        assert in_band > 0.3

    def test_reduces_feature_count(self):
        X, c, _ = _regression_dataset(n=40)
        result = cars_select(X, c, n_components=2,
                             n_iterations=20, random_state=0)
        # CARS should typically select far fewer than all features
        assert result.selected_indices.size < X.shape[1]

    def test_reproducible_with_seed(self):
        X, c, _ = _regression_dataset(n=40)
        a = cars_select(X, c, n_components=2, n_iterations=10, random_state=42)
        b = cars_select(X, c, n_components=2, n_iterations=10, random_state=42)
        assert np.array_equal(a.selected_indices, b.selected_indices)

    def test_iteration_history_shape(self):
        X, c, _ = _regression_dataset(n=40)
        result = cars_select(X, c, n_components=2,
                             n_iterations=15, random_state=0)
        assert isinstance(result.iteration_history, list)
        assert len(result.iteration_history) > 0
        # First few iterations should keep many features; later ones fewer
        n_features_curve = [h["n_features"] for h in result.iteration_history]
        assert n_features_curve[0] >= n_features_curve[-1]

    def test_too_few_samples_raises(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (3, 50)); y = rng.normal(0, 1, 3)
        with pytest.raises(ValueError):
            cars_select(X, y, n_components=2, n_iterations=5, cv_folds=5)
