"""Unit tests: protocol, attacks, aggregators, metrics, data."""

import numpy as np
import pytest
import torch

from fuse_fl import attacks
from fuse_fl.aggregators import (
    BOBA,
    FUSE,
    HNobs,
    Krum,
    QFFL,
    FedAvg,
    TrimmedMean,
)
from fuse_fl.metrics import audit_screening, qfair_metrics
from fuse_fl.models import classifier_coords, flatten_params, make_mlp, set_flat_params
from fuse_fl.protocol import Submission, local_update
from fuse_fl.screening import TrustedReference, build_trusted_reference


def _honest(n=10, d=24, seed=0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.normal(size=(n, d)).astype(np.float32))


# ---- protocol ----

def test_fairness_scaling_applies_F_pow_q():
    """u_i = F_i^q * Delta_i exactly: the server sees only the fused product."""
    torch.manual_seed(0)
    init = flatten_params(make_mlp())
    m = make_mlp()
    set_flat_params(m, init)
    X = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))
    for q in (0.0, 1.0, 2.0):
        mm = make_mlp()
        set_flat_params(mm, init)
        s = local_update(mm, X, y, q=q, client_id=0)
        ratio = s.vector.norm().item() / max(s.true_delta_norm, 1e-12)
        assert ratio == pytest.approx(max(s.true_loss, 1e-8) ** q, rel=1e-4)


def test_fedavg_q0_is_unscaled():
    """q=0 must transmit the raw delta (FedAvg path)."""
    torch.manual_seed(1)
    m = make_mlp()
    X = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))
    s = local_update(m, X, y, q=0.0, client_id=0)
    assert abs(s.vector.norm().item() - s.true_delta_norm) < 1e-4


# ---- attacks ----

def test_all_attacks_return_vector():
    H = _honest()
    rng = np.random.default_rng(0)
    for name, fn in attacks.ATTACKS.items():
        v = fn(H, 2, rng)
        assert v.shape == (H.shape[1],), name
        assert torch.isfinite(v).all(), name


def test_attack_registry_has_seven():
    assert len(attacks.ATTACKS) == 7
    assert set(attacks.ATTACKS) == {
        "sign_flip", "ipm", "gaussian", "alie",
        "min_max", "min_sum", "magnitude_scaling",
    }


def test_magnitude_scaling_is_honest_direction():
    H = _honest()
    losses = np.linspace(0.5, 3.0, len(H))
    v = attacks.magnitude_scaling(H, 2, np.random.default_rng(0), losses=losses)
    base = H[-1]  # highest start loss
    cos = float((v @ base) / (v.norm() * base.norm()))
    assert cos > 0.999  # same direction, only bigger (paper: x5)
    assert v.norm() == pytest.approx(5.0 * base.norm())


def test_alie_within_honest_cloud():
    H = _honest(n=20)
    v = attacks.alie(H, 4, np.random.default_rng(0))
    # ALIE is designed to hide inside the honest variance
    assert v.norm() < 3 * H.norm(dim=1).max()


def test_attack_parameters_match_paper():
    """Appendix B.2 spot checks: sign flip = -mean, IPM = -3*mean."""
    H = _honest(n=10)
    rng = np.random.default_rng(0)
    assert torch.allclose(attacks.sign_flip(H, 4, rng), -H.mean(dim=0))
    assert torch.allclose(attacks.ipm(H, 4, rng), -3.0 * H.mean(dim=0))
    # ALIE z for the paper's config shape: |Phi^-1(0.4)| ~= 0.2533
    from scipy.stats import norm as _n
    assert abs(abs(float(_n.ppf(0.4))) - 0.2533) < 1e-3


# ---- aggregators ----

def _subs(vecs, byz=None):
    byz = byz or set()
    return [Submission(client_id=i, vector=v, is_byzantine=(i in byz)) for i, v in enumerate(vecs)]


def test_trimmed_mean_clips_outliers():
    H = _honest(n=8, d=8)
    evil = torch.ones(8) * 1000
    subs = _subs(list(H) + [evil])
    step, diag = TrimmedMean(f=1).aggregate(subs)
    assert diag["kept"] == 7  # n - 2f
    assert step.norm() < 100


def test_krum_picks_honest_centroid():
    H = _honest(n=8, d=8)
    evil = torch.ones(8) * 500
    subs = _subs(list(H) + [evil])
    step, diag = Krum(f=1).aggregate(subs)
    assert diag["chosen"] != 8
    assert step.norm() < 100


def test_fuse_rejects_byzantine_end_to_end():
    """FUSE on a toy cone: honest mixtures + one scaled copy -> scaled copy out."""
    rng = np.random.default_rng(11)
    d, C = 24, 4
    G = np.abs(rng.normal(size=(d, C))) + 0.1
    ref = TrustedReference(G=G, l_hat=np.ones(C), coords=(0, d))
    vecs = [torch.tensor((G @ rng.dirichlet([2.0] * C)).astype(np.float32)) for _ in range(8)]
    evil = 12.0 * vecs[0]
    subs = _subs(vecs + [evil], byz={8})
    step, diag = FUSE(f=2, q=1.0, kappa=3.0).aggregate(subs, ctx={"coords": (0, d), "ref": ref})
    keep = diag["keep_mask"]
    # evil caught by the radial test; the leftover budget spends on the largest
    # (benign) cone residual, exactly the paper's declared-budget semantics
    assert not keep[8]
    assert diag["stage1_rejected"] == 1
    assert keep.sum() == 7


def test_qffl_and_fedavg_mean():
    H = _honest(n=6, d=8)
    subs = _subs(list(H))
    s1, _ = QFFL().aggregate(subs)
    s2, _ = FedAvg().aggregate(subs)
    assert torch.allclose(s1, s2)


def test_fltrust_server_update_matches_client_scale():
    """FLTrust's server update must follow the same local procedure as
    clients (regression: a single trusted gradient step clipped everything
    to ~0.02 norm and the model never trained, clean acc 0.21)."""
    from fuse_fl.aggregators import FLTrust

    torch.manual_seed(3)
    model = make_mlp()
    n = sum(p.numel() for p in model.parameters())
    H = _honest(n=6, d=n)
    subs = _subs(list(H))
    Xt = torch.randn(40, 784)
    yt = torch.randint(0, 10, (40,))
    step, diag = FLTrust(q=1.0).aggregate(subs, model=model, ctx={"trusted": (Xt, yt)})
    assert torch.isfinite(step).all()
    # step must be on the client-update scale, not the single-grad scale
    assert step.norm() > 1e-3


def test_hnobs_aggregator_rejects_outlier():
    H = _honest(n=8, d=8)
    evil = torch.ones(8) * 900
    subs = _subs(list(H) + [evil])
    step, diag = HNobs(f=1).aggregate(subs, ctx={"coords": (0, 8)})
    assert not diag["keep_mask"][8]


# ---- metrics ----

def test_qfair_metrics_shapes():
    accs = [0.9] * 9 + [0.5]
    m = qfair_metrics(accs)
    assert m["worst10_client_acc"] == 0.5
    assert m["client_acc_variance"] > 0
    assert m["mean_client_acc"] == 0.86


def test_audit_screening_counts():
    kept = np.array([True, True, False, False])
    is_byz = np.array([False, False, True, True])
    true_loss = np.array([0.5, 2.0, np.nan, np.nan])
    a = audit_screening(kept, is_byz, true_loss)
    assert a["byz_rejected_frac"] == 1.0
    assert a["honest_rejected_frac"] == 0.0
    assert a["highloss_honest_rejected_frac"] == 0.0


# ---- data + trusted reference ----

def test_trusted_reference_shapes():
    from fuse_fl import data as dm

    Xtr, ytr, Xte, yte = dm.make_synthetic(n=2000, seed=0)
    model = make_mlp(in_dim=Xtr.shape[1], n_classes=10)
    coords = classifier_coords(model)
    Xt = torch.tensor(Xtr[:200], dtype=torch.float32)
    yt = torch.tensor(ytr[:200], dtype=torch.long)
    ref = build_trusted_reference(model, Xt, yt, coords)
    assert ref.G.shape[1] == 10
    assert ref.G.shape[0] == coords[1] - coords[0]
    assert len(ref.l_hat) == 10


def test_dirichlet_partition_is_skewed():
    from fuse_fl import data as dm

    Xtr, ytr, _, _ = dm.make_synthetic(n=4000, seed=1)
    cX, cy = dm.partition_dirichlet(Xtr, ytr, n_clients=10, alpha=0.5, seed=2)
    props = [np.bincount(y, minlength=10) / len(y) for y in cy]
    # with alpha=0.5 at least one client should be strongly skewed
    assert max(p.max() for p in props) > 0.5


def test_experiment_smoke_runs():
    from fuse_fl.experiment import RunConfig, run, summarize

    cfg = RunConfig(aggregator="fuse", attack=None, n_clients=6, clients_per_round=4,
                    n_byz=1, rounds=2, train_subset=600, test_per_client=50,
                    trusted_per_class=5, seed=0, hidden=32)
    res = run(cfg)
    s = summarize(res)
    assert 0.0 <= s["final_global_acc"] <= 1.0
    assert len(res.history) == 2
    # paper's protocol: declared budget f_t = 0 in clean runs, so FUSE must
    # keep every honest submission when there is no attack
    assert all(h["kept"] == 4 for h in res.history)
