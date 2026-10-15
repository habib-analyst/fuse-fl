"""Quickstart: one FUSE round under a magnitude-scaling attack.

Shows the whole pipeline: q-fair submissions -> FUSE screening ->
step-size rescale. Expected output is printed at the end.
"""

import numpy as np
import torch

from fuse_fl.aggregators import FUSE, HNobs, QFFL
from fuse_fl.protocol import Submission
from fuse_fl.screening import TrustedReference, fuse_screen

torch.manual_seed(0)
rng = np.random.default_rng(0)

# toy honest world: updates live in a nonnegative cone (4 class directions)
d, C = 24, 4
G = np.abs(rng.normal(size=(d, C))) + 0.1
ref = TrustedReference(G=G, l_hat=np.ones(C), coords=(0, d))

honest = [torch.tensor((G @ rng.dirichlet([2] * C)).astype(np.float32)) for _ in range(8)]
subs = [Submission(client_id=i, vector=v) for i, v in enumerate(honest)]
# Byzantine Sybil: 12x-scaled honest direction (indistinguishable by norm alone
# from a legitimately amplified high-loss client)
subs.append(Submission(client_id=-1, vector=12.0 * honest[0], is_byzantine=True))

for agg_cls, kwargs in [(QFFL, {}), (HNobs, {"f": 1}), (FUSE, {"f": 1, "q": 1.0})]:
    agg = agg_cls(**kwargs)
    step, diag = agg.aggregate(subs, ctx={"coords": (0, d), "ref": ref})
    kept = diag.get("keep_mask")
    byz_kept = bool(kept[-1]) if kept is not None else "n/a"
    print(f"{agg.name:8s} kept={diag['kept']} byzantine_kept={byz_kept}")

# Expected output:
# q-ffl    kept=9 byzantine_kept=True
# h-nobs   kept=8 byzantine_kept=n/a
# fuse     kept=8 byzantine_kept=False
