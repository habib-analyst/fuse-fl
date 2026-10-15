"""q-fair FL protocol: local client update + fairness scaling (update-only interface).

Honest client i computes a local delta and transmits only the fairness-scaled
message u_i = F_i(w_t)^q * Delta_i. The server never sees F_i and Delta_i
separately. Byzantine clients may transmit arbitrary vectors instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .models import flatten_params, set_flat_params


@dataclass
class Submission:
    client_id: int
    vector: torch.Tensor  # fairness-scaled update u_i (full model dim)
    is_byzantine: bool = False
    true_loss: float = float("nan")  # oracle info, for audit only
    true_delta_norm: float = float("nan")


def local_update(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.05,
    local_epochs: int = 2,
    batch_size: int = 64,
    q: float = 1.0,
    client_id: int = 0,
    rng: np.random.Generator | None = None,
) -> Submission:
    """Honest client: local SGD then fairness scaling.

    Matches the H-nobs / FUSE update-only formulation: the client folds the
    loss-dependent weight F_i(w_t)^q into the transmitted update locally.
    """
    model = model  # trained in place on a private copy
    w0 = flatten_params(model)
    with torch.no_grad():
        logits = model(X)
        f0 = float(nn.functional.cross_entropy(logits, y).item())
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = len(X)
    idx = np.arange(n)
    for _ in range(local_epochs):
        if rng is not None:
            rng.shuffle(idx)
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X[b]), y[b])
            loss.backward()
            opt.step()
    delta = flatten_params(model) - w0
    set_flat_params(model, w0)  # restore caller model
    scale = (max(f0, 1e-8)) ** q
    u = scale * delta
    return Submission(
        client_id=client_id,
        vector=u.detach(),
        is_byzantine=False,
        true_loss=f0,
        true_delta_norm=float(delta.norm().item()),
    )
