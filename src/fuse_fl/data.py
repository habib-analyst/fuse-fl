"""Data: MNIST loading (curl download to cache) + Dirichlet non-IID partitions.

The FUSE analysis assumes *shared class-conditional label skew*: every
honest client draws class-c samples from the same conditional Q_c, and
clients differ only in their label proportions p_i ~ Dirichlet(alpha).
``partition_dirichlet`` implements exactly that recipe.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path(os.environ.get("FUSE_FL_CACHE", Path.home() / ".cache" / "fuse_fl"))
MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _curl_download(url: str, dest: Path) -> None:
    """Download with curl (Python HTTP clients hang through this egress proxy)."""
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl", "-sSL", "--max-time", "300", "-o", str(tmp), url], check=True)
    tmp.rename(dest)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    assert data[2] == 8 and data[3] == 3, f"not an image idx file: {path}"
    n, rows, cols = int(data[4:8].view(">u4")[0]), int(data[8:12].view(">u4")[0]), int(
        data[12:16].view(">u4")[0]
    )
    return data[16:].reshape(n, rows, cols).astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    assert data[2] == 8 and data[3] == 1, f"not a label idx file: {path}"
    n = int(data[4:8].view(">u4")[0])
    return data[8:8 + n].astype(np.int64)


def load_mnist(subset: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST (download once to cache dir). Returns (Xtr, ytr, Xte, yte)."""
    rng = np.random.default_rng(seed)
    files = {}
    for key, url in MNIST_URLS.items():
        dest = CACHE_DIR / f"{key}.gz"
        if not dest.exists():
            _curl_download(url, dest)
        files[key] = dest
    Xtr = _read_idx_images(files["train_images"])
    ytr = _read_idx_labels(files["train_labels"])
    Xte = _read_idx_images(files["test_images"])
    yte = _read_idx_labels(files["test_labels"])
    if subset is not None and subset < len(Xtr):
        idx = rng.choice(len(Xtr), size=subset, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    return Xtr, ytr, Xte, yte


def make_synthetic(
    n: int = 6000, n_classes: int = 10, n_features: int = 64, sep: float = 2.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared class-conditional Gaussians (offline fallback, no downloads)."""
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=sep, size=(n_classes, n_features)).astype(np.float32)
    per = n // n_classes
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(rng.normal(loc=means[c], scale=1.0, size=(per, n_features)).astype(np.float32))
        ys.append(np.full(per, c, dtype=np.int64))
    X = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(X))
    cut = int(0.85 * len(X))
    return X[perm][:cut], y[perm][:cut], X[perm][cut:], y[perm][cut:]


def partition_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float = 1.0,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Dirichlet label-skew partition (shared class conditionals, skewed p_i).

    Mirrors the FUSE paper's experimental setup (Dirichlet alpha=1).
    Returns (client_X, client_y) index lists... actually arrays per client.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    class_idx = [np.where(y == c)[0] for c in range(n_classes)]
    for lst in class_idx:
        rng.shuffle(lst)
    # per-class Dirichlet split across clients
    ptrs = np.zeros(n_classes, dtype=int)
    out_x, out_y = [], []
    for i in range(n_clients):
        props = rng.dirichlet([alpha] * n_classes)
        take = []
        for c in range(n_classes):
            n_take = int(round(props[c] * len(class_idx[c])))
            s, e = ptrs[c], min(ptrs[c] + n_take, len(class_idx[c]))
            take.append(class_idx[c][s:e])
            ptrs[c] = e
        take = [t for t in take if len(t)]
        if not take:  # pathological Dirichlet draw: seed one sample
            c = int(rng.integers(n_classes))
            take = [class_idx[c][:1]]
        take = np.concatenate(take)
        rng.shuffle(take)
        out_x.append(X[take])
        out_y.append(y[take])
    return out_x, out_y


def holdout_trusted(X: np.ndarray, y: np.ndarray, per_class: int = 20, seed: int = 0):
    """Small trusted labeled set on the server: `per_class` examples per class.

    FUSE's trusted class references are built from this set. It must not
    overlap the client partitions (held out before partitioning).
    """
    rng = np.random.default_rng(seed)
    keep_idx, trust_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        trust_idx.append(chosen)
        keep_idx.append(np.setdiff1d(idx, chosen))
    trust_idx = np.concatenate(trust_idx)
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx], X[trust_idx], y[trust_idx]


def client_test_splits(
    Xte: np.ndarray, yte: np.ndarray, client_label_props: list[np.ndarray], per_client: int = 400, seed: int = 0
):
    """Per-client test splits following each client's label proportions.

    Used for worst-10% client accuracy: each client's model quality is
    measured on the distribution it actually sees.
    """
    rng = np.random.default_rng(seed)
    class_idx = [np.where(yte == c)[0] for c in range(int(yte.max()) + 1)]
    out = []
    for props in client_label_props:
        take = []
        for c, p in enumerate(props):
            n = int(round(p * per_client))
            pool = class_idx[c]
            take.append(rng.choice(pool, size=min(n, len(pool)), replace=False))
        take = np.concatenate([t for t in take if len(t)])
        out.append((Xte[take], yte[take]))
    return out
