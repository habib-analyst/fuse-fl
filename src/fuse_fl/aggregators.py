"""Aggregators: the server's update rule.

Each aggregator consumes a list of fairness-scaled submissions (plus any
server-side context it needs) and returns the retained-mean step to add to
the global model. The report card compares: FedAvg, q-FFL (unprotected),
Trimmed Mean, Krum, FLTrust, H-nobs, BOBA (simplex), FUSE.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .screening import (
    TrustedReference,
    boba_screen,
    build_trusted_reference,
    fuse_screen,
    fuse_step_rescale,
    hnobs_screen,
)


class Aggregator:
    name = "base"

    def aggregate(
        self,
        subs: list,
        model: nn.Module | None = None,
        ctx: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Return (step_to_add, diagnostics)."""
        raise NotImplementedError


def _stack(subs) -> torch.Tensor:
    return torch.stack([s.vector for s in subs])


def _kept_mean(subs, keep: np.ndarray) -> torch.Tensor:
    """Mean of kept submissions; zero step if nothing was kept (all corrupt)."""
    kept = [s for s, k in zip(subs, keep) if k]
    if not kept:
        return torch.zeros_like(subs[0].vector)
    return _stack(kept).mean(dim=0)


class FedAvg(Aggregator):
    """Vanilla FedAvg on *unscaled* deltas (clients send plain Delta_i)."""

    name = "fedavg"

    def aggregate(self, subs, model=None, ctx=None):
        step = _stack(subs).mean(dim=0)
        return step, {"kept": len(subs)}


class QFFL(Aggregator):
    """q-FFL with no defense: mean of the fairness-scaled submissions."""

    name = "q-ffl"

    def aggregate(self, subs, model=None, ctx=None):
        step = _stack(subs).mean(dim=0)
        return step, {"kept": len(subs)}


class TrimmedMean(Aggregator):
    name = "trimmed-mean"

    def __init__(self, f: int):
        self.f = f

    def aggregate(self, subs, model=None, ctx=None):
        V = _stack(subs)
        n = len(V)
        k = min(self.f, n - 1)
        Vs, _ = torch.sort(V, dim=0)
        trimmed = Vs[k : n - k] if n - 2 * k > 0 else Vs[n // 2 : n // 2 + 1]
        # soft clipping: no hard rejection mask to audit
        return trimmed.mean(dim=0), {"kept": len(trimmed), "keep_mask": None}


class Krum(Aggregator):
    name = "krum"

    def __init__(self, f: int):
        self.f = f

    def aggregate(self, subs, model=None, ctx=None):
        V = _stack(subs)
        n = len(V)
        keep = max(n - self.f - 2, 1)
        D = torch.cdist(V, V)
        scores = D.topk(keep + 1, largest=False).values.sum(dim=1)
        best = int(torch.argmin(scores))
        mask = np.zeros(n, dtype=bool)
        mask[best] = True
        return V[best], {"kept": 1, "chosen": best, "keep_mask": mask}


class FLTrust(Aggregator):
    """FLTrust: cosine-similarity reweighting + norm clipping against a
    server update computed on the trusted set.

    The server update follows the same local procedure as the clients
    (including fairness scaling), so norms are comparable.
    """

    name = "fltrust"

    def __init__(self, lr: float = 0.05, q: float = 1.0, local_epochs: int = 2, batch_size: int = 64):
        self.lr = lr
        self.q = q
        self.local_epochs = local_epochs
        self.batch_size = batch_size

    def aggregate(self, subs, model=None, ctx=None):
        import copy

        from .protocol import local_update

        assert model is not None and ctx is not None and "trusted" in ctx
        Xt, yt = ctx["trusted"]
        import numpy as np

        mcopy = copy.deepcopy(model)  # same architecture, same current weights
        sub0 = local_update(
            mcopy, Xt, yt, lr=self.lr, local_epochs=self.local_epochs,
            batch_size=self.batch_size, q=self.q, client_id=-999,
            rng=np.random.default_rng(0),
        )
        g0 = sub0.vector
        V = _stack(subs)
        n0 = g0.norm() + 1e-12
        cos = (V @ g0) / (V.norm(dim=1) * n0 + 1e-12)
        ts = torch.clamp(cos, min=0.0)
        clipped = V * (n0 / (V.norm(dim=1, keepdim=True) + 1e-12)).clamp(max=1.0)
        step = (ts.unsqueeze(1) * clipped).sum(dim=0) / (ts.sum() + 1e-12)
        return step, {"kept": int((ts > 0).sum()), "keep_mask": None}


def _screening_view(subs, coords):
    s, e = coords
    return np.stack([sub.vector[s:e].numpy() for sub in subs])


class HNobs(Aggregator):
    """H-nobs baseline: reject f largest-norm submissions, mean of rest."""

    name = "h-nobs"

    def __init__(self, f: int):
        self.f = f

    def aggregate(self, subs, model=None, ctx=None):
        coords = ctx["coords"]
        V = _screening_view(subs, coords)
        keep = hnobs_screen(V, min(self.f, len(subs) - 1))
        step = _kept_mean(subs, keep)
        return step, {"kept": int(keep.sum()), "keep_mask": keep}


class BOBA(Aggregator):
    """BOBA-style baseline: unit-sum simplex screening, mean of rest."""

    name = "boba"

    def __init__(self, f: int):
        self.f = f

    def aggregate(self, subs, model=None, ctx=None):
        coords = ctx["coords"]
        ref: TrustedReference = ctx["ref"]
        V = _screening_view(subs, coords)
        keep = boba_screen(ref, V, min(self.f, len(subs) - 1))
        step = _kept_mean(subs, keep)
        return step, {"kept": int(keep.sum()), "keep_mask": keep}


class FUSE(Aggregator):
    """FUSE (arXiv:2610.20264): radial + structural screening, then
    step-size control via reconstructed client losses."""

    name = "fuse"

    def __init__(self, f: int, q: float = 1.0, kappa: float = 3.0):
        self.f = f
        self.q = q
        self.kappa = kappa

    def aggregate(self, subs, model=None, ctx=None):
        coords = ctx["coords"]
        ref: TrustedReference = ctx["ref"]
        V = _screening_view(subs, coords)
        res = fuse_screen(ref, V, min(self.f, len(subs) - 1), kappa=self.kappa)
        kept_subs = [s for s, k in zip(subs, res.keep) if k]
        scales = fuse_step_rescale(ref, res, q=self.q)
        denom = float(scales.sum()) if len(scales) else 1.0
        step = _stack(kept_subs).sum(dim=0) / max(denom, 1e-12) if kept_subs else torch.zeros_like(subs[0].vector)
        return step, {
            "kept": int(res.keep.sum()),
            "keep_mask": res.keep,
            "stage1_rejected": int(res.rejected_stage1.sum()),
            "stage2_rejected": int(res.rejected_stage2.sum()),
            "L": res.scores.L.tolist(),
            "Dcone": res.scores.Dcone.tolist(),
        }


AGGREGATORS = {
    "fedavg": FedAvg,
    "q-ffl": QFFL,
    "trimmed-mean": TrimmedMean,
    "krum": Krum,
    "fltrust": FLTrust,
    "h-nobs": HNobs,
    "boba": BOBA,
    "fuse": FUSE,
}
