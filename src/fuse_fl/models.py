"""Models + screening-coordinate utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


def make_mlp(in_dim: int = 784, hidden: int = 128, n_classes: int = 10, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes),
    )


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data = flat[offset : offset + n].reshape(p.shape).clone()
        offset += n


def classifier_coords(model: nn.Module) -> tuple[int, int]:
    """Start/end offsets of the classifier (last Linear) block in the flat vector.

    FUSE screens in these coordinates on vision benchmarks.
    """
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    assert last is not None
    total = sum(p.numel() for p in model.parameters())
    cls_n = last.weight.numel() + last.bias.numel()
    return total - cls_n, total
