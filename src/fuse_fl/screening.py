"""FUSE screening core (faithful to arXiv:2610.20264, Eq. 3-5) + comparison screens.

Pipeline per round:
  1. Trusted class references: from a small labeled set held by the server,
     compute class losses l_hat_c and class-conditioned reference directions
     g_c = [-eta * grad(mean class-c loss)] restricted to screening coords.
  2. Coefficient recovery: for each submission v_i, solve NNLS
     a_hat_i = argmin_{a>=0} ||v_i - G a||^2.
  3. Two scores: radial scale L_i = ||a_hat_i||_1 (coefficient mass) and
     structural residual D_i^cone = ||v_i - G a_hat_i||^2 / max(||v_i||^2, eps).
  4. Stage I (radial): reject up to f submissions with L_i > kappa * L_med.
     Stage II (structural): spend remaining budget on largest D_i^cone.
  5. Post-screening step-size control: p_hat_i = a_hat_i / max(L_i, eps),
     F_tilde_i = l_hat^T p_hat_i, and
     w_{t+1} = w_t + sum(u_i) / sum(max(1, F_tilde_i^q)) over retained i.

Also provided: norm screening (H-nobs) and unit-sum simplex screening
(BOBA-style) for the audit comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import nnls

from .models import flatten_params


@dataclass
class TrustedReference:
    G: np.ndarray           # (d_scr, C) class-conditioned reference directions
    l_hat: np.ndarray       # (C,) trusted class losses
    coords: tuple[int, int]  # screening coordinate slice


def build_trusted_reference(
    model: nn.Module,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    coords: tuple[int, int],
    lr: float = 0.05,
) -> TrustedReference:
    """Eq. (3): trusted class losses + class-conditioned directions."""
    C = int(yt.max().item()) + 1
    s, e = coords
    cols, losses = [], []
    for c in range(C):
        Xc = Xt[yt == c]
        if len(Xc) == 0:
            continue
        model.zero_grad()
        loss = nn.functional.cross_entropy(model(Xc), yt[yt == c])
        losses.append(float(loss.item()))
        grads = torch.autograd.grad(loss, model.parameters())
        flat = torch.cat([g.detach().reshape(-1) for g in grads])
        cols.append((-lr * flat)[s:e].numpy())
    G = np.stack(cols, axis=1) if cols else np.zeros((e - s, 1))
    return TrustedReference(G=G, l_hat=np.array(losses), coords=coords)


@dataclass
class FuseScores:
    L: np.ndarray        # radial scale (coefficient mass) per submission
    Dcone: np.ndarray    # structural cone residual per submission
    coef: list[np.ndarray]


def recover_coefficients(ref: TrustedReference, V: np.ndarray, eps: float = 1e-12) -> FuseScores:
    """Eq. (4): NNLS coefficient recovery for each submission row.

    Non-finite rows are never fit: they get mass 0 and cone residual 1
    (maximally suspicious) and are rejected by the caller's pre-pass.
    """
    n = V.shape[0]
    L = np.zeros(n)
    D = np.ones(n)
    coefs = [np.zeros(ref.G.shape[1]) for _ in range(n)]
    finite = np.isfinite(V).all(axis=1)
    for i in np.where(finite)[0]:
        a, rnorm = nnls(ref.G, V[i])
        coefs[i] = a
        L[i] = float(np.sum(a))
        denom = max(float(np.dot(V[i], V[i])), eps)
        D[i] = float(rnorm ** 2 / denom)
    return FuseScores(L=L, Dcone=D, coef=coefs)


@dataclass
class ScreenResult:
    keep: np.ndarray      # bool mask of retained submissions
    rejected_stage1: np.ndarray  # rejected by the radial test
    rejected_stage2: np.ndarray  # rejected by the structural test
    scores: FuseScores


def fuse_screen(
    ref: TrustedReference,
    V: np.ndarray,
    f: int,
    kappa: float = 3.0,
    eps: float = 1e-12,
) -> ScreenResult:
    """Two-stage screening. Stage 0: non-finite submissions are rejected
    outright (never aggregated; does not consume the f budget).
    Stage I: radial (L_i > kappa * L_med), up to f.
    Stage II: structural (largest D_i^cone), spending the remaining budget."""
    n = V.shape[0]
    keep = np.ones(n, dtype=bool)
    finite = np.isfinite(V).all(axis=1)
    stage0 = ~finite
    keep[stage0] = False
    scores = recover_coefficients(ref, V, eps=eps)
    stage1 = np.zeros(n, dtype=bool)
    finite_idx = np.where(finite)[0]
    L_med = float(np.median(scores.L[finite_idx])) if len(finite_idx) else 0.0
    # Stage I: radial test, most anomalous first, at most f
    order = finite_idx[np.argsort(-scores.L[finite_idx])]
    budget = f
    for i in order:
        if budget <= 0:
            break
        if scores.L[i] > kappa * max(L_med, eps):
            keep[i] = False
            stage1[i] = True
            budget -= 1
    # Stage II: structural test on what remains
    stage2 = np.zeros(n, dtype=bool)
    if budget > 0:
        remaining = np.where(keep)[0]
        order2 = remaining[np.argsort(-scores.Dcone[remaining])]
        for i in order2[:budget]:
            keep[i] = False
            stage2[i] = True
    return ScreenResult(keep=keep, rejected_stage1=stage1, rejected_stage2=stage2, scores=scores)


def fuse_step_rescale(
    ref: TrustedReference,
    res: ScreenResult,
    q: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Eq. (5): loss reconstruction per retained client -> step-size denominator."""
    scales = []
    for i, kept in enumerate(res.keep):
        if not kept:
            continue
        a = res.scores.coef[i]
        L = float(np.sum(a))
        p_hat = a / max(L, eps)
        # align coefficient vector with available trusted losses
        C = min(len(p_hat), len(ref.l_hat))
        F_tilde = float(np.dot(ref.l_hat[:C], p_hat[:C]))
        scales.append(max(1.0, F_tilde ** q))
    return np.array(scales)


# ---------------------------------------------------------------------------
# Comparison screens
# ---------------------------------------------------------------------------

def hnobs_screen(V: np.ndarray, f: int) -> np.ndarray:
    """H-nobs: reject the f largest-norm submissions (magnitude screening).

    Non-finite submissions are rejected outright first.
    """
    keep = np.ones(len(V), dtype=bool)
    finite = np.isfinite(V).all(axis=1)
    keep[~finite] = False
    norms = np.linalg.norm(np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    norms[~finite] = -np.inf  # already rejected; keep them out of the ranking
    cand = np.where(finite)[0]
    drop = cand[np.argsort(-norms[cand])[:f]]
    keep[drop] = False
    return keep


def boba_screen(ref: TrustedReference, V: np.ndarray, f: int, eps: float = 1e-12) -> np.ndarray:
    """BOBA-style: unit-sum simplex fit, reject f largest simplex residuals.

    Approximates min_{a>=0, sum a=1} ||v - G a|| via NNLS on the augmented
    system [G; lam*1^T] a ~= [v; lam] with a strong unit-sum penalty.
    """
    n = V.shape[0]
    resid = np.zeros(n)
    C = ref.G.shape[1]
    lam = float(np.linalg.norm(ref.G, "fro") / max(C, 1))
    Ga = np.vstack([ref.G, lam * np.ones((1, C))])
    finite = np.isfinite(V).all(axis=1)
    for i in np.where(finite)[0]:
        vi = np.concatenate([V[i], [lam]])
        _, rnorm = nnls(Ga, vi)
        denom = max(float(np.dot(V[i], V[i])), eps)
        resid[i] = float(rnorm ** 2 / denom)
    resid[~finite] = np.inf
    keep = np.ones(n, dtype=bool)
    keep[~finite] = False  # non-finite: rejected outright
    cand = np.where(finite)[0]
    drop = cand[np.argsort(-resid[cand])[:f]]
    keep[drop] = False
    return keep
