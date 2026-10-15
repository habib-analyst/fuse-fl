"""Experiment driver: seeded q-fair FL rounds with Byzantine Sybils.

Each round: sample `clients_per_round` honest clients, append `n_byz`
Byzantine Sybils (all submitting the chosen attack vector), run the
aggregator, update the model, log accuracy + screening audit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import attacks
from .aggregators import AGGREGATORS
from .metrics import accuracy, audit_screening, client_accuracies, qfair_metrics
from .models import classifier_coords, flatten_params, make_mlp, set_flat_params
from .protocol import Submission, local_update
from .screening import build_trusted_reference


@dataclass
class RunConfig:
    aggregator: str = "fuse"
    attack: str | None = None          # None = clean run
    n_clients: int = 20
    clients_per_round: int = 10
    n_byz: int = 3                     # Sybils appended each attacked round
    rounds: int = 15
    q: float = 1.0
    f: int = 3                         # declared rejection budget
    kappa: float = 3.0
    alpha: float = 1.0                 # Dirichlet skew
    lr: float = 0.05
    local_epochs: int = 2
    batch_size: int = 64
    hidden: int = 128
    train_subset: int = 12000
    test_per_client: int = 300
    trusted_per_class: int = 20
    seed: int = 42


@dataclass
class RunResult:
    config: RunConfig
    history: list = field(default_factory=list)   # per-round dicts
    final_global_acc: float = 0.0
    final_client: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    seconds: float = 0.0


def run(cfg: RunConfig, verbose: bool = False) -> RunResult:
    from . import data as data_mod

    t0 = time.time()
    torch.set_num_threads(4)
    rng = np.random.default_rng(cfg.seed)

    try:
        Xtr, ytr, Xte, yte = data_mod.load_mnist(subset=cfg.train_subset, seed=cfg.seed)
        source = "mnist"
    except Exception as e:
        if verbose:
            print(f"MNIST unavailable ({e}); using synthetic data")
        Xtr, ytr, Xte, yte = data_mod.make_synthetic(n=cfg.train_subset, seed=cfg.seed)
        source = "synthetic"
    in_dim = Xtr.shape[1] if Xtr.ndim == 2 else int(np.prod(Xtr.shape[1:]))
    n_classes = int(ytr.max()) + 1

    # trusted set held out BEFORE partitioning
    Xtr, ytr, Xtrust, ytrust = data_mod.holdout_trusted(
        Xtr, ytr, per_class=cfg.trusted_per_class, seed=cfg.seed
    )
    cX, cy = data_mod.partition_dirichlet(Xtr, ytr, cfg.n_clients, alpha=cfg.alpha, seed=cfg.seed + 1)
    label_props = []
    for yy in cy:
        hist = np.bincount(yy, minlength=n_classes).astype(float)
        label_props.append(hist / max(hist.sum(), 1))
    client_tests = [
        (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        for X, y in data_mod.client_test_splits(Xte, yte, label_props, per_client=cfg.test_per_client, seed=cfg.seed + 2)
    ]
    Xt_trust = torch.tensor(Xtrust, dtype=torch.float32)
    yt_trust = torch.tensor(ytrust, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.long)

    flat_dim = in_dim
    model = make_mlp(in_dim=flat_dim, hidden=cfg.hidden, n_classes=n_classes, seed=cfg.seed)
    coords = classifier_coords(model)

    agg_cls = AGGREGATORS[cfg.aggregator]
    # FedAvg sends unscaled deltas: q=0 on the client side.
    client_q = 0.0 if cfg.aggregator == "fedavg" else cfg.q
    # The paper declares f_t = 0 in clean (unattacked) conditions: with no
    # adversary, a rejection budget only eats honest submissions.
    eff_f = 0 if cfg.attack is None else cfg.f
    if cfg.aggregator in ("trimmed-mean", "krum", "h-nobs", "boba"):
        agg = agg_cls(cfg.f if cfg.aggregator in ("trimmed-mean", "krum") else eff_f)
    elif cfg.aggregator == "fuse":
        agg = agg_cls(eff_f, q=cfg.q, kappa=cfg.kappa)
    elif cfg.aggregator == "fltrust":
        agg = agg_cls(q=client_q, lr=cfg.lr, local_epochs=cfg.local_epochs, batch_size=cfg.batch_size)
    else:
        agg = agg_cls()

    history = []
    audit_accum = {"byz_rej": [], "honest_rej": [], "highloss_rej": []}
    atk_fn = attacks.ATTACKS[cfg.attack] if cfg.attack else None

    for rnd in range(cfg.rounds):
        chosen = rng.choice(cfg.n_clients, size=min(cfg.clients_per_round, cfg.n_clients), replace=False)
        subs: list[Submission] = []
        for ci in chosen:
            mcopy = make_mlp(in_dim=flat_dim, hidden=cfg.hidden, n_classes=n_classes, seed=0)
            set_flat_params(mcopy, flatten_params(model))
            subs.append(
                local_update(
                    mcopy,
                    torch.tensor(cX[ci], dtype=torch.float32),
                    torch.tensor(cy[ci], dtype=torch.long),
                    lr=cfg.lr,
                    local_epochs=cfg.local_epochs,
                    batch_size=cfg.batch_size,
                    q=client_q,
                    client_id=int(ci),
                    rng=rng,
                )
            )
        honest_vecs = torch.stack([s.vector for s in subs])
        honest_losses = np.array([s.true_loss for s in subs])
        if atk_fn is not None:
            evil = atk_fn(honest_vecs, cfg.n_byz, rng, losses=honest_losses)
            for b in range(cfg.n_byz):
                subs.append(
                    Submission(client_id=-1 - b, vector=evil.clone(), is_byzantine=True)
                )

        # server-side context: trusted reference rebuilt at current w_t
        ref = build_trusted_reference(model, Xt_trust, yt_trust, coords, lr=cfg.lr)
        ctx = {"coords": coords, "ref": ref, "trusted": (Xt_trust, yt_trust)}

        step, diag = agg.aggregate(subs, model=model, ctx=ctx)
        if not torch.isfinite(step).all():
            # registered collapse: record NaN and stop (paper's collapse rule)
            rec = {"round": rnd, "global_acc": float("nan"),
                   "mean_client_acc": float("nan"), "worst10_client_acc": float("nan"),
                   "client_acc_variance": float("nan"), "client_acc_min": float("nan"),
                   "client_acc_max": float("nan"), "kept": diag.get("kept"),
                   "collapsed": True}
            history.append(rec)
            if verbose:
                print(f"round {rnd:2d} COLLAPSED (non-finite step)")
            break
        set_flat_params(model, flatten_params(model) + step)

        gacc = accuracy(model, Xte_t, yte_t)
        cacc = client_accuracies(model, client_tests)
        fm = qfair_metrics(cacc)

        kept = diag.get("keep_mask")
        is_byz = np.array([s.is_byzantine for s in subs])
        true_loss = np.array([s.true_loss for s in subs])
        if kept is not None and len(kept) == len(subs):
            aud = audit_screening(np.asarray(kept, bool), is_byz, true_loss)
            for k, ak in (("byz_rej", "byz_rejected_frac"), ("honest_rej", "honest_rejected_frac"),
                          ("highloss_rej", "highloss_honest_rejected_frac")):
                v = aud[ak]
                if v == v:  # not nan
                    audit_accum[k].append(v)

        rec = {"round": rnd, "global_acc": gacc, **fm, "kept": diag.get("kept")}
        history.append(rec)
        if verbose:
            print(f"round {rnd:2d} acc={gacc:.3f} worst10={fm['worst10_client_acc']:.3f} kept={diag.get('kept')}")

    final_client = qfair_metrics(client_accuracies(model, client_tests))
    audit = {k: (float(np.mean(v)) if v else float("nan")) for k, v in audit_accum.items()}
    return RunResult(
        config=cfg,
        history=history,
        final_global_acc=history[-1]["global_acc"],
        final_client=final_client,
        audit=audit,
        seconds=time.time() - t0,
    )


def summarize(res: RunResult) -> dict:
    c = res.config

    def clean(v, nd=4):
        return None if v != v else round(v, nd)  # NaN (collapsed run) -> null

    return {
        "aggregator": c.aggregator,
        "attack": c.attack or "clean",
        "seed": c.seed,
        "rounds": c.rounds,
        "collapsed": any(h.get("collapsed") for h in res.history),
        "final_global_acc": clean(res.final_global_acc),
        "worst10_client_acc": clean(res.final_client["worst10_client_acc"]),
        "client_acc_variance": clean(res.final_client["client_acc_variance"], 6),
        "byz_rejected_frac": clean(res.audit.get("byz_rej", float("nan"))),
        "highloss_honest_rejected_frac": clean(res.audit.get("highloss_rej", float("nan"))),
        "seconds": round(res.seconds, 1),
    }
