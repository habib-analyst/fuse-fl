"""Byzantine attacks on the fairness-scaled update interface.

Parameters follow the FUSE paper's Appendix B.2 exactly:
  - Sign Flip: -mean(honest)
  - IPM: -3 * mean(honest)
  - Gaussian: per-coordinate N(0, 200^2)
  - ALIE: mean - z*std, z = |Phi^{-1}((floor(n/2+1) - s)/(n - s))|
    (0.2533 for the paper's 10-honest/4-Byzantine config)
  - Min-Max / Min-Sum: gamma search (gamma_0=10, tol=1e-5) per
    Shejwalkar & Houmansadr 2021, shared Sybils
  - Magnitude Scaling: highest-start-loss honest submission x 5
    (the fairness-ambiguous attack: indistinguishable by magnitude alone
    from a legitimately amplified high-loss client)

Signature: attack(honest: Tensor[n, d], f: int, rng, losses=None) -> Tensor[d].
`losses` carries the honest clients' start losses for loss-targeted attacks.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm


def _mean(honest: torch.Tensor) -> torch.Tensor:
    return honest.mean(dim=0)


def sign_flip(honest: torch.Tensor, f: int, rng: np.random.Generator, losses=None) -> torch.Tensor:
    """-mean(honest)."""
    return -_mean(honest)


def ipm(honest: torch.Tensor, f: int, rng: np.random.Generator, losses=None) -> torch.Tensor:
    """Inner product manipulation: -3 * mean(honest)."""
    return -3.0 * _mean(honest)


def gaussian(
    honest: torch.Tensor, f: int, rng: np.random.Generator, losses=None, std: float = 200.0
) -> torch.Tensor:
    """Isotropic Gaussian, per-coordinate std 200 (paper's parameter)."""
    noise = torch.tensor(
        rng.normal(0.0, std, size=(honest.shape[1],)), dtype=torch.float32
    )
    return noise


def alie(honest: torch.Tensor, f: int, rng: np.random.Generator, losses=None) -> torch.Tensor:
    """A Little Is Enough (Baruch et al. 2019): mean - z_max * std."""
    m, s = len(honest), max(f, 1)
    ratio = (m // 2 + 1 - s) / (m - s)
    ratio = float(np.clip(ratio, 1e-6, 1 - 1e-6))
    z = abs(float(norm.ppf(ratio)))
    mu = _mean(honest)
    std = honest.std(dim=0, unbiased=False)
    return mu - z * std


def _malicious(honest: torch.Tensor, candidate: torch.Tensor, mode: str) -> bool:
    """Shejwalkar & Houmansadr maliciousness check: the poisoned update is
    'malicious' if it lies farther from the benign cloud than the cloud's
    own diameter (min-max) / summed spread (min-sum)."""
    d = torch.cdist(candidate.unsqueeze(0), honest).squeeze(0)
    if mode == "min_max":
        return bool((d.max() > torch.cdist(honest, honest).max()).item())
    # min_sum: compare against the benign sum-of-distances scale
    benign_sum = torch.cdist(honest, honest).sum()
    return bool((d.sum() > benign_sum).item())


def _gamma_search(
    honest: torch.Tensor,
    direction: torch.Tensor,
    mode: str,
    gamma0: float = 10.0,
    tol: float = 1e-5,
) -> float:
    """Two-phase gamma search: double until malicious, then bisect."""
    mu = _mean(honest)
    gamma, step = gamma0, gamma0 / 2.0
    # phase 1: shrink until not malicious
    while _malicious(honest, mu - gamma * direction, mode) and step > tol:
        gamma -= step
        step /= 2.0
    # phase 2: grow back to the boundary
    step = gamma0 / 2.0
    while step > tol:
        if _malicious(honest, mu - (gamma + step) * direction, mode):
            step /= 2.0
        else:
            gamma += step
    return gamma


def min_max(honest: torch.Tensor, f: int, rng: np.random.Generator, losses=None) -> torch.Tensor:
    """Min-Max: farthest-point evasion along -std, gamma searched."""
    mu = _mean(honest)
    std = honest.std(dim=0, unbiased=False)
    direction = std / (std.norm() + 1e-12)
    gamma = _gamma_search(honest, direction, "min_max")
    return mu - gamma * direction


def min_sum(honest: torch.Tensor, f: int, rng: np.random.Generator, losses=None) -> torch.Tensor:
    """Min-Sum: summed-distance evasion along -std, gamma searched."""
    mu = _mean(honest)
    std = honest.std(dim=0, unbiased=False)
    direction = std / (std.norm() + 1e-12)
    gamma = _gamma_search(honest, direction, "min_sum")
    return mu - gamma * direction


def magnitude_scaling(
    honest: torch.Tensor,
    f: int,
    rng: np.random.Generator,
    losses=None,
    factor: float = 5.0,
) -> torch.Tensor:
    """Copy the highest-start-loss honest submission, scale by 5."""
    if losses is not None:
        losses = np.asarray(losses, dtype=float)
        finite = np.isfinite(losses)
        idx = int(np.argmax(np.where(finite, losses, -np.inf)))
    else:
        idx = int(torch.argmax(honest.norm(dim=1)))
    return factor * honest[idx]


ATTACKS = {
    "sign_flip": sign_flip,
    "ipm": ipm,
    "gaussian": gaussian,
    "alie": alie,
    "min_max": min_max,
    "min_sum": min_sum,
    "magnitude_scaling": magnitude_scaling,
}
