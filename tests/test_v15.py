"""V1.15 tests — chemometric preprocessing, PLS-DA, PCR, Kennard-Stone, permutation."""

from __future__ import annotations

import numpy as np
import pytest

from openspectra_workbench.chemometrics import (
    PCRModel, PLSDAModel, apply_osc, apply_preprocessing,
    autoscale, detrend, emsc, fit_pcr, fit_pls_da,
    kennard_stone_split, mean_centre, msc, osc_fit,
    pareto_scale, permutation_test, poisson_scale, range_scale,
    select_n_components_pls_da, selectivity_ratio, snv,
    vip_scores, y_randomisation,
)


# =========================================================================
# Synthetic data helpers
# =========================================================================

def _two_class_dataset(n_per_class: int = 30, seed: int = 0):
    """Two synthetic FTIR-like classes that differ only in one band."""
    rng = np.random.default_rng(seed)
    x = np.linspace(800, 1800, 200)
    base_a = np.exp(-((x - 1100) / 30) ** 2)            # band at 1100
    base_b = np.exp(-((x - 1500) / 30) ** 2)            # band at 1500
    rows, labels = [], []
    for _ in range(n_per_class):
        rows.append(base_a + rng.normal(0, 0.02, x.size)); labels.append("A")
    for _ in range(n_per_class):
        rows.append(base_b + rng.normal(0, 0.02, x.size)); labels.append("B")
    return np.array(rows), np.array(labels), x


def _regression_dataset(n: int = 40, seed: int = 0):
    """Synthetic concentration dataset where y is encoded by one band."""
    rng = np.random.default_rng(seed)
    x = np.linspace(900, 2400, 300)
    band = np.exp(-((x - 1700) / 30) ** 2)
    concentrations = rng.uniform(0.1, 5.0, n)
    rows = [c * band + rng.normal(0, 0.005, x.size) for c in concentrations]
    return np.array(rows), concentrations, x


# =========================================================================
# Preprocessing
# =========================================================================

class TestSNV:
    def test_zero_mean_unit_std_per_row(self):
        X = np.random.default_rng(0).normal(5, 2, (10, 100))
        out, _ = snv(X)
        assert np.allclose(out.mean(axis=1), 0, atol=1e-9)
        assert np.allclose(out.std(axis=1), 1, atol=1e-9)

    def test_constant_row_does_not_crash(self):
        X = np.ones((3, 50))
        out, _ = snv(X)
        # Constant rows → division by 1 (zero std fallback) → still zero mean
        assert np.all(np.isfinite(out))


class TestDetrend:
    def test_subtracts_polynomial_baseline(self):
        x = np.linspace(0, 10, 200)
        peak = np.exp(-((x - 5) / 0.5) ** 2)
        # Add a quadratic baseline only — detrend(order=2) should remove
        # most of it. The peak in the middle of the window does pull on the
        # polyfit, so the residual baseline is not exactly zero (this is an
        # inherent limitation of polynomial detrending — true baseline-only
        # methods like rubberband or arPLS handle peaks better).
        baseline = 0.1 + 0.05 * x + 0.02 * x ** 2
        rows = np.array([peak + baseline, peak + 2 * baseline])
        out, state = detrend(rows, x, order=2)
        # Residual baseline should be small (well below the original 0-3 range)
        away = (x < 3) | (x > 7)
        assert abs(out[0][away].mean()) < 0.1
        assert state["order"] == 2

    def test_x_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            detrend(np.zeros((2, 50)), np.arange(40))


class TestMSC:
    def test_mean_reference_recovered(self):
        rng = np.random.default_rng(42)
        ref = np.exp(-(np.arange(100) - 50) ** 2 / 50)
        # Each row is a + b*ref + tiny noise — MSC should recover ref
        rows = []
        for _ in range(10):
            a, b = rng.uniform(-0.1, 0.1), rng.uniform(0.5, 2.0)
            rows.append(a + b * ref + rng.normal(0, 0.001, ref.size))
        X = np.array(rows)
        corrected, state = msc(X)
        # Each corrected row should be ≈ ref
        for row in corrected:
            assert np.corrcoef(row, ref)[0, 1] > 0.999

    def test_state_holds_reference(self):
        X = np.random.default_rng(0).normal(1, 0.1, (5, 30))
        _, state = msc(X)
        assert state["method"] == "msc"
        assert state["reference"].shape == (30,)


class TestEMSC:
    def test_runs_with_polynomial_baseline(self):
        rng = np.random.default_rng(1)
        x = np.linspace(0, 10, 100)
        ref = np.exp(-((x - 5) / 0.6) ** 2)
        rows = []
        for _ in range(8):
            a, b = rng.uniform(-0.05, 0.05), rng.uniform(0.7, 1.3)
            slope = rng.uniform(-0.05, 0.05)
            rows.append(a + slope * x + b * ref + rng.normal(0, 0.001, x.size))
        X = np.array(rows)
        corrected, state = emsc(X, x, poly_order=1)
        for row in corrected:
            assert np.corrcoef(row, ref)[0, 1] > 0.99
        assert state["poly_order"] == 1


class TestScaling:
    def test_autoscale_zero_mean_unit_std(self):
        X = np.random.default_rng(0).normal(3, 4, (50, 20))
        out, state = autoscale(X)
        assert np.allclose(out.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(out.std(axis=0), 1, atol=1e-9)
        # Round-trip through apply_preprocessing
        out2 = apply_preprocessing(X, state)
        assert np.allclose(out, out2)

    def test_pareto_divides_by_sqrt_std(self):
        X = np.random.default_rng(0).normal(0, 4, (50, 5))
        out, state = pareto_scale(X)
        expected = (X - X.mean(axis=0)) / np.sqrt(X.std(axis=0))
        assert np.allclose(out, expected, atol=1e-9)

    def test_range_scale_in_unit_range(self):
        X = np.random.default_rng(0).uniform(0, 10, (30, 8))
        out, _ = range_scale(X)
        # Each column should span at most 1.0
        assert np.all(out.max(axis=0) - out.min(axis=0) <= 1.0 + 1e-9)

    def test_poisson_scale_for_counts(self):
        # Need enough samples to keep per-column variance estimates stable
        X = np.random.default_rng(0).poisson(50, (500, 8)).astype(float)
        out, _ = poisson_scale(X)
        # After dividing by sqrt(mean), per-column variance should be ~1
        assert np.all(out.var(axis=0) > 0.7)
        assert np.all(out.var(axis=0) < 1.3)

    def test_mean_centre_round_trip(self):
        X = np.random.default_rng(0).normal(5, 2, (10, 4))
        out, state = mean_centre(X)
        out2 = apply_preprocessing(X, state)
        assert np.allclose(out, out2)


class TestApplyPreprocessing:
    def test_replays_msc(self):
        X_train = np.random.default_rng(0).normal(1, 0.5, (10, 50))
        X_train_msc, state = msc(X_train)
        X_new = np.random.default_rng(99).normal(1, 0.5, (5, 50))
        out = apply_preprocessing(X_new, state)
        # Replayed transform should differ from a fresh fit on X_new
        independent_fit, _ = msc(X_new)
        assert out.shape == X_new.shape
        # The replayed reference is the *training* mean, so results differ
        assert not np.allclose(out, independent_fit)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            apply_preprocessing(np.zeros((2, 3)), {"method": "no_such_thing"})


class TestOSC:
    def test_orthogonal_components_removed(self):
        rng = np.random.default_rng(0)
        n, p = 30, 50
        # X has a y-correlated direction + a y-orthogonal nuisance
        y_dir = rng.normal(0, 1, p)
        nuisance_dir = rng.normal(0, 1, p)
        # Force nuisance to be orthogonal to y_dir at the start
        nuisance_dir -= (y_dir @ nuisance_dir / (y_dir @ y_dir)) * y_dir
        y = rng.normal(0, 1, n)
        nuisance_scores = rng.normal(0, 1, n)
        X = np.outer(y, y_dir) + np.outer(nuisance_scores, nuisance_dir)
        X += rng.normal(0, 0.01, X.shape)
        X_corr, fit = osc_fit(X, y, n_components=1)
        # Variance should drop
        assert X_corr.var() < X.var()

    def test_apply_osc_runs(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(0, 1, (20, 10))
        y = rng.normal(0, 1, 20)
        _, fit = osc_fit(X_train, y, n_components=1)
        X_new = rng.normal(0, 1, (5, 10))
        out = apply_osc(X_new, fit)
        assert out.shape == X_new.shape


# =========================================================================
# PLS-DA
# =========================================================================

class TestPLSDA:
    def test_fits_two_classes_perfectly(self):
        X, y, _ = _two_class_dataset(n_per_class=30)
        model = fit_pls_da(X, y, n_components=3)
        preds = model.predict(X)
        assert sum(p == t for p, t in zip(preds, y)) >= 55  # ≥ 92% accuracy

    def test_three_classes(self):
        rng = np.random.default_rng(0)
        x = np.linspace(800, 1800, 200)
        rows, labels = [], []
        for centre, lab in [(1100, "A"), (1300, "B"), (1500, "C")]:
            band = np.exp(-((x - centre) / 30) ** 2)
            for _ in range(20):
                rows.append(band + rng.normal(0, 0.02, x.size)); labels.append(lab)
        X = np.array(rows); y = np.array(labels)
        model = fit_pls_da(X, y, n_components=4)
        preds = model.predict(X)
        accuracy = sum(p == t for p, t in zip(preds, y)) / len(y)
        assert accuracy > 0.85

    def test_predict_proba_like(self):
        X, y, _ = _two_class_dataset(n_per_class=20)
        model = fit_pls_da(X, y, n_components=2)
        proba = model.predict_proba_like(X)
        assert proba.shape == (X.shape[0], 2)

    def test_select_n_components_finds_reasonable_value(self):
        X, y, _ = _two_class_dataset(n_per_class=20)
        result = select_n_components_pls_da(X, y, max_components=8, cv_folds=4)
        assert 1 <= result["best_n_components"] <= 8
        assert result["best_accuracy"] > 0.8

    def test_unknown_label_raises(self):
        X, y, _ = _two_class_dataset(n_per_class=10)
        model = fit_pls_da(X, y, n_components=2, classes=["A", "B"])
        # Predicting a new sample with a never-seen-before label is not a
        # thing for PLS-DA (we always predict an existing class), so just
        # confirm the API rejects fitting with classes that exclude a label
        with pytest.raises(ValueError):
            fit_pls_da(X, y, n_components=2, classes=["A"])  # missing B


class TestVIPAndSelectivity:
    def test_vip_sum_squares_equals_n_features(self):
        X, y, _ = _two_class_dataset(n_per_class=20)
        model = fit_pls_da(X, y, n_components=3)
        vip = vip_scores(model)
        # Classical identity: Σ VIP² = n_features
        assert abs(np.sum(vip ** 2) - X.shape[1]) < 1e-3

    def test_vip_peaks_at_diagnostic_band(self):
        X, y, x = _two_class_dataset(n_per_class=30)
        model = fit_pls_da(X, y, n_components=3)
        vip = vip_scores(model)
        # The two diagnostic bands are at 1100 and 1500; VIP should peak there
        peak_indices = np.argsort(vip)[-30:]
        peak_x = x[peak_indices]
        # At least one of the top-30 VIP values should be near 1100 or 1500
        near_a = ((peak_x >= 1050) & (peak_x <= 1150)).any()
        near_b = ((peak_x >= 1450) & (peak_x <= 1550)).any()
        assert near_a or near_b

    def test_selectivity_ratio_shape(self):
        X, y, _ = _two_class_dataset(n_per_class=20)
        model = fit_pls_da(X, y, n_components=3)
        sr = selectivity_ratio(model, X)
        assert sr.shape == (X.shape[1],)
        assert np.all(sr >= 0)


# =========================================================================
# PCR
# =========================================================================

class TestPCR:
    def test_recovers_concentrations(self):
        X, c, _ = _regression_dataset(n=40)
        model = fit_pcr(X, c, n_components=3)
        # PCR on this nicely-conditioned dataset should fit well
        assert model.r_squared_train > 0.95
        preds = model.predict(X)
        assert np.corrcoef(preds, c)[0, 1] > 0.97

    def test_cross_validation_q2(self):
        X, c, _ = _regression_dataset(n=40)
        model = fit_pcr(X, c, n_components=3, cv_folds=5)
        assert model.r_squared_cv is not None
        assert model.rmsecv is not None
        assert model.r_squared_cv > 0.9

    def test_summary_string(self):
        X, c, _ = _regression_dataset(n=20)
        model = fit_pcr(X, c, n_components=2, cv_folds=4)
        s = model.summary()
        assert "PCR" in s and "RMSEC" in s


# =========================================================================
# Kennard-Stone
# =========================================================================

class TestKennardStone:
    def test_split_sizes(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 5))
        train_idx, val_idx = kennard_stone_split(X, n_train=35)
        assert len(train_idx) == 35
        assert len(val_idx) == 15
        assert set(train_idx).isdisjoint(set(val_idx))
        assert set(train_idx) | set(val_idx) == set(range(50))

    def test_fraction_input(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (100, 4))
        train_idx, val_idx = kennard_stone_split(X, n_train=0.7)
        assert len(train_idx) == 70

    def test_first_two_are_most_distant(self):
        # Construct a dataset where two specific points are clearly furthest
        X = np.array([
            [0, 0], [0.1, 0.05], [0.2, 0.1],
            [10, 10],  # outlier 1
            [-10, -10],  # outlier 2 — these two should be the first picks
            [0.3, 0.2], [0.05, 0.0],
        ], dtype=float)
        train_idx, _ = kennard_stone_split(X, n_train=2)
        assert set(train_idx) == {3, 4}  # the two corners

    def test_invalid_n_train_raises(self):
        X = np.random.default_rng(0).normal(0, 1, (10, 3))
        with pytest.raises(ValueError):
            kennard_stone_split(X, n_train=0)
        with pytest.raises(ValueError):
            kennard_stone_split(X, n_train=11)


# =========================================================================
# Permutation testing & Y-randomisation
# =========================================================================

class TestPermutation:
    def test_significant_signal_gives_low_p(self):
        X, c, _ = _regression_dataset(n=40)
        # Score function: R² of a 3-component PLS via leave-one-out
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict

        def score(X, y):
            mod = PLSRegression(n_components=3)
            preds = cross_val_predict(mod, X, y, cv=5)
            ss_res = float(np.sum((preds - y) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result = permutation_test(score, X, c, n_permutations=50, random_state=0)
        assert result.observed_score > 0.9
        assert result.p_value < 0.05

    def test_random_noise_gives_p_near_uniform(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (40, 20))
        y = rng.normal(0, 1, 40)
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict

        def score(X, y):
            try:
                preds = cross_val_predict(PLSRegression(n_components=2), X, y, cv=5)
                ss_res = float(np.sum((preds - y) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            except Exception:
                return 0.0

        result = permutation_test(score, X, y, n_permutations=30, random_state=0)
        # Pure noise should not produce a vanishingly small p-value
        assert result.p_value > 0.05


class TestYRandomisation:
    def test_real_r2_above_null_cloud(self):
        X, c, _ = _regression_dataset(n=30)

        def fit_score(X, y):
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.model_selection import cross_val_predict
            mod = PLSRegression(n_components=3)
            mod.fit(X, y)
            preds_train = mod.predict(X).ravel()
            preds_cv = cross_val_predict(mod, X, y, cv=5).ravel()
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - float(np.sum((preds_train - y) ** 2)) / ss_tot if ss_tot > 0 else 0.0
            q2 = 1.0 - float(np.sum((preds_cv - y) ** 2)) / ss_tot if ss_tot > 0 else 0.0
            return r2, q2

        result = y_randomisation(fit_score, X, c, n_permutations=20, random_state=0)
        assert result.real_r2 > result.null_r2.mean() + 2 * result.null_r2.std()
        assert result.real_q2 > result.null_q2.mean() + 2 * result.null_q2.std()
