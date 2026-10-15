"""Metrics: accuracy (global + worst-10% client), fairness, audit diagnostics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch: int = 512) -> float:
    correct, total = 0, 0
    for s in range(0, len(X), batch):
        pred = model(X[s : s + batch]).argmax(dim=1)
        correct += int((pred == y[s : s + batch]).sum())
        total += len(y[s : s + batch])
    return correct / max(total, 1)


def client_accuracies(model, client_tests) -> list[float]:
    return [accuracy(model, X, y) for X, y in client_tests]


def qfair_metrics(client_accs: list[float]) -> dict:
    """q-fairness view: variance + worst-10% of the client-accuracy distribution."""
    a = np.array(client_accs, dtype=float)
    k = max(1, int(round(0.1 * len(a))))
    worst10 = float(np.sort(a)[:k].mean())
    return {
        "mean_client_acc": float(a.mean()),
        "worst10_client_acc": worst10,
        "client_acc_variance": float(a.var()),
        "client_acc_min": float(a.min()),
        "client_acc_max": float(a.max()),
    }


def audit_screening(kept: np.ndarray, is_byz: np.ndarray, true_loss: np.ndarray) -> dict:
    """Oracle-audited screening report (only valid in simulation).

    Mirrors the paper's Table 2: high-loss-honest rejection (fairness wants
    low) vs Byzantine rejection (robustness wants high).
    """
    kept = np.asarray(kept, bool)
    rej = ~kept
    honest = ~np.asarray(is_byz, bool)
    out = {
        "byz_rejected_frac": float(rej[~honest].mean()) if (~honest).sum() else float("nan"),
        "honest_rejected_frac": float(rej[honest].mean()) if honest.sum() else float("nan"),
    }
    hl = honest & np.isfinite(true_loss)
    if hl.sum():
        thresh = np.quantile(true_loss[hl], 0.8)
        high_loss = hl & (true_loss >= thresh)
        out["highloss_honest_rejected_frac"] = float(rej[high_loss].mean()) if high_loss.sum() else float("nan")
    else:
        out["highloss_honest_rejected_frac"] = float("nan")
    return out
