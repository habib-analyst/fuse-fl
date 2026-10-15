# fuse-fl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Open implementation and audit harness for **FUSE: fairness-aware update
screening against Byzantine exploitation in q-fair federated learning**
(arXiv:2610.20264, Sun et al., Oct 2026).

In fair federated learning the server only sees fairness-scaled updates,
so a legitimately amplified high-loss client and a maliciously scaled
attack look identical by magnitude. FUSE fixes this with two complementary
geometric tests on the fairness-warped update cone: a **radial test**
(recovered coefficient mass vs. the round median) and a **structural test**
(NNLS cone residual vs. trusted class directions). The authors released no
code; this repo is the first open implementation.

## Quickstart

```bash
pip install fuse-fl   # or: pip install -e .
python examples/quickstart.py
```

```python
from fuse_fl.aggregators import FUSE
from fuse_fl.protocol import Submission

# subs: fairness-scaled updates u_i = F_i^q * Delta_i (update-only interface)
agg = FUSE(f=3, q=1.0, kappa=3.0)
step, diag = agg.aggregate(subs, model=model, ctx={
    "coords": classifier_coords(model),   # screening coordinates
    "ref": trusted_reference,             # built from the server's trusted set
    "trusted": (X_trust, y_trust),
})
model_params += step
```

Expected output of `examples/quickstart.py` (toy cone, 8 honest + 1 scaled Sybil):

```
q-ffl    kept=9 byzantine_kept=True
h-nobs   kept=8 byzantine_kept=False
fuse     kept=8 byzantine_kept=False
```

## Measured results

MNIST/MLP, Dirichlet(alpha=1.0) label skew, 20 clients (10 sampled/round),
3 Byzantine Sybils/round, q=1, rejection budget f=3, 12 rounds, seeds 40-41.
Seed-wise worst-attack global accuracy (the paper's primary statistic):

| aggregator | seed 40 (worst attack) | seed 41 (worst attack) |
|---|---|---|
| fedavg | 0.101 (min_sum) | 0.098 (min_sum) |
| q-ffl (unprotected) | 0.089 (min_sum) | 0.098 (min_sum) |
| trimmed-mean | 0.629 (ipm) | 0.551 (ipm) |
| krum | 0.096 (gaussian) | 0.089 (sign_flip) |
| fltrust | 0.792 (alie) | 0.812 (min_max) |
| h-nobs | 0.381 (sign_flip) | 0.096 (ipm) |
| boba | 0.096 (magnitude_scaling) | 0.101 (gaussian) |
| **fuse** | **0.114 (min_sum)** | **0.275 (min_sum)** |

FUSE vs the key fairness-ambiguous attack (magnitude_scaling, x5 on the
highest-loss client): 0.799/0.838 global accuracy, 0.81/0.56 Byzantine
rejection, 0.04/0.08 high-loss-honest rejection, i.e. at clean level.
Honest reading: at this CPU scale (12 rounds, short-trained trusted
references) FUSE beats the unprotected and norm/simplex baselines but does
not reproduce the paper's clean sweep; FLTrust is the strongest baseline
here. The method's guarantees assume concentrated honest losses and mature
references (see PAPER.md "What the two tests do and do NOT guarantee").

Full per-seed, per-attack audit (robustness + fairness side by side) is in
[`experiments/results/REPORT_CARD.md`](experiments/results/REPORT_CARD.md).

## How it works

Honest updates under shared class-conditional label skew are nonnegative
mixtures of class-conditioned reference directions. Fairness scaling keeps
the nonnegativity but replaces the unit-sum simplex with a nonnegative cone
whose coefficient mass is the fairness scale `F_i^q`. FUSE:

1. builds trusted class references from a small server-side labeled set,
2. recovers each submission's latent coefficients by NNLS,
3. rejects radial outliers (`L_i > kappa * L_med`) then the largest cone
   residuals, spending the declared budget in that order,
4. reconstructs each retained client's loss from normalized coefficients and
   rescales the retained mean to restore the fairness step size.

Also implemented for comparison: FedAvg, unprotected q-FFL, trimmed mean,
Krum, FLTrust, H-nobs (norm screening), BOBA (unit-sum simplex screening).

## Honest limits

- This repo is CPU-scale (MNIST/MLP); the paper evaluates CIFAR-10/ResNet-18
  and more. Numbers here are ours, not the paper's.
- Both tests are blind to an *adaptive* attacker that stays inside the
  trusted cone. The paper runs such an attack to isolate this boundary.
- Screening needs a trusted labeled set on the server; no trusted data, no
  reference directions. The declared budget `f` must cover the true
  Byzantine count.

## Citation

```bibtex
@misc{sun2026fuse,
  title={FUSE: Fairness-aware Update Screening against Byzantine Exploitation in $q$-Fair Federated Learning},
  author={Tiantong Sun and Jinlin Liu and Xingru Wei and Zequan Fang and Wei Chen and Qiang Yang and Xiaojin Zhang},
  year={2026},
  eprint={2610.20264},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

Independent implementation by Habib Ur Rehman; not affiliated with the paper authors.

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
