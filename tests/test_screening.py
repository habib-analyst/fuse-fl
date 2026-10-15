"""Unit tests for fuse-fl screening geometry."""

import numpy as np
import pytest
from scipy.optimize import nnls

from fuse_fl.screening import (
    FuseScores,
    TrustedReference,
    boba_screen,
    fuse_screen,
    fuse_step_rescale,
    hnobs_screen,
    recover_coefficients,
)


def _cone_ref(d=20, C=4, seed=0):
    rng = np.random.default_rng(seed)
    G = np.abs(rng.normal(size=(d, C))) + 0.1
    return TrustedReference(G=G, l_hat=np.arange(1, C + 1, dtype=float), coords=(0, d))


def test_nnls_honest_update_fits_cone():
    """An honest nonnegative mixture has ~zero cone residual."""
    ref = _cone_ref()
    rng = np.random.default_rng(1)
    a = rng.dirichlet([2.0] * 4)
    v = ref.G @ a + 1e-6 * rng.normal(size=20)
    scores = recover_coefficients(ref, v.reshape(1, -1))
    assert scores.Dcone[0] < 1e-3
    assert scores.L[0] == pytest.approx(a.sum(), rel=1e-2)


def test_radial_mass_tracks_fairness_scale():
    """Recovered mass L_i grows with the fairness scale (paper's r~0.94 claim)."""
    ref = _cone_ref()
    rng = np.random.default_rng(2)
    a0 = rng.dirichlet([2.0] * 4)
    masses, scales = [], []
    for s in (0.5, 1.0, 2.0, 5.0, 10.0):
        v = s * (ref.G @ a0)
        sc = recover_coefficients(ref, v.reshape(1, -1))
        masses.append(sc.L[0])
        scales.append(s)
    corr = np.corrcoef(scales, masses)[0, 1]
    assert corr > 0.99


def test_structural_test_detects_off_cone_update():
    """A random direction is not in the cone -> large Dcone, rejected stage II."""
    ref = _cone_ref()
    rng = np.random.default_rng(3)
    honest = np.stack([ref.G @ rng.dirichlet([2.0] * 4) for _ in range(8)])
    evil = rng.normal(size=(2, 20)) * 5
    V = np.vstack([honest, evil])
    res = fuse_screen(ref, V, f=2, kappa=3.0)
    assert res.keep.sum() == 8
    assert not res.keep[8] and not res.keep[9]
    assert res.rejected_stage2.sum() == 2


def test_radial_test_detects_magnitude_scaling():
    """A 10x-scaled honest-direction update is caught by the radial test."""
    ref = _cone_ref()
    rng = np.random.default_rng(4)
    a = rng.dirichlet([2.0] * 4)
    honest = np.stack([ref.G @ rng.dirichlet([2.0] * 4) for _ in range(8)])
    evil = (10.0 * (ref.G @ a)).reshape(1, -1)
    V = np.vstack([honest, evil])
    res = fuse_screen(ref, V, f=2, kappa=3.0)
    assert not res.keep[8]
    assert res.rejected_stage1.sum() == 1  # radial test, not structural


def test_radial_test_spares_legitimate_high_loss():
    """A 2x-amplified honest update (legit fairness growth) is NOT rejected.

    Note: with a declared budget f=2 and no genuine outlier, stage II spends
    the leftover budget on the largest (benign) cone residuals -- the declared
    budget is always spent. The guarantee is only that the legit update is kept.
    """
    ref = _cone_ref()
    rng = np.random.default_rng(5)
    base = [ref.G @ rng.dirichlet([2.0] * 4) for _ in range(8)]
    big_honest = 2.0 * (ref.G @ rng.dirichlet([2.0] * 4))
    V = np.vstack(base + [big_honest])
    res = fuse_screen(ref, V, f=2, kappa=3.0)
    assert res.keep[8]  # the legitimately amplified update survives


def test_two_stage_budget_accounting():
    """Stage I + Stage II together never exceed f rejections."""
    ref = _cone_ref()
    rng = np.random.default_rng(6)
    V = np.vstack([ref.G @ rng.dirichlet([2.0] * 4) for _ in range(6)])
    V = np.vstack([V, rng.normal(size=(4, 20)) * 8])
    res = fuse_screen(ref, V, f=3, kappa=3.0)
    assert (res.rejected_stage1.sum() + res.rejected_stage2.sum()) <= 3
    assert res.keep.sum() >= 7


def test_nonfinite_submission_rejected_without_crash():
    """Regression: a diverged (inf) submission must not crash NNLS; it is
    rejected outright in the pre-pass."""
    ref = _cone_ref()
    rng = np.random.default_rng(9)
    V = np.vstack([ref.G @ rng.dirichlet([2.0] * 4) for _ in range(6)])
    V[3, 5] = np.inf
    res = fuse_screen(ref, V, f=2, kappa=3.0)
    assert not res.keep[3]
    assert res.keep.sum() == 3  # 1 pre-pass + f=2 spent on the largest residuals
    keep = hnobs_screen(V, f=1)
    assert not keep[3]
    keep = boba_screen(ref, V, f=1)
    assert not keep[3]


def test_step_rescale_restores_fairness_denominator():
    """Eq. (5): denominator reconstructs the sum of F_tilde^q over retained."""
    ref = _cone_ref(C=3)
    rng = np.random.default_rng(7)
    V = np.vstack([ref.G @ rng.dirichlet([2.0] * 3) for _ in range(5)])
    res = fuse_screen(ref, V, f=1, kappa=3.0)
    scales = fuse_step_rescale(ref, res, q=1.0)
    assert len(scales) == res.keep.sum()
    assert np.all(scales >= 1.0)  # max(1, F_tilde^q)


def test_hnobs_rejects_largest_norms():
    V = np.vstack([np.ones(10) * 0.5] * 5 + [np.ones(10) * 100.0])
    keep = hnobs_screen(V, f=1)
    assert keep.sum() == 5 and not keep[5]


def test_boba_simplex_vs_cone():
    """Unit-sum simplex fit has large residual on a radially-scaled mixture
    (the paper's simplex-breaks-under-fairness point), the cone fit does not."""
    ref = _cone_ref()
    rng = np.random.default_rng(8)
    scaled = [(s * (ref.G @ rng.dirichlet([2.0] * 4))).astype(float)
              for s in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)]
    V = np.vstack(scaled)
    # BOBA rejects the 5x mixture: it cannot sit on the unit-sum simplex
    keep_boba = boba_screen(ref, V, f=1)
    assert not keep_boba[7]
    # FUSE keeps it: 5 < kappa * median(scale) and the cone fit is exact
    res = fuse_screen(ref, V, f=0, kappa=3.0)
    assert res.keep.all()
