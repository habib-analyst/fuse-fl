"""fuse-fl: open implementation + audit harness for FUSE.

FUSE (arXiv:2610.20264, Sun et al. 2026) is a fairness-aware update
screening aggregator for Byzantine-robust q-fair federated learning under
an *update-only* interface: the server observes only fairness-scaled
client updates, never the loss and descent direction separately.

This package implements:
  - a CPU-feasible q-fair FL simulator (Dirichlet non-IID partitions),
  - the FUSE radial + structural screening tests (faithful to the paper),
  - seven Byzantine attacks and eight baseline aggregators,
  - an audit report card (fairness + robustness side by side).
"""

__version__ = "0.1.0"
__all__ = [
    "data",
    "models",
    "protocol",
    "attacks",
    "screening",
    "aggregators",
    "metrics",
    "experiment",
    "report",
]
