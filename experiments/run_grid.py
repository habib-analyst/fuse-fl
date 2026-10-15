"""Run the full aggregator x attack robustness grid, append results as JSONL."""

import json
import sys
from pathlib import Path

from fuse_fl.experiment import RunConfig, run, summarize

OUT = Path(__file__).parent.parent / "results" / "grid.jsonl"

AGGREGATORS = ["fedavg", "q-ffl", "trimmed-mean", "krum", "fltrust", "h-nobs", "boba", "fuse"]
ATTACKS = [None, "sign_flip", "ipm", "gaussian", "alie", "min_max", "min_sum", "magnitude_scaling"]
SEEDS = [40, 41]
ROUNDS = 12


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["aggregator"], r["attack"], r["seed"]))
    total = len(AGGREGATORS) * len(ATTACKS) * len(SEEDS)
    n = 0
    with OUT.open("a") as f:
        for agg in AGGREGATORS:
            for atk in ATTACKS:
                for seed in SEEDS:
                    key = (agg, atk or "clean", seed)
                    if key in done:
                        n += 1
                        continue
                    cfg = RunConfig(
                        aggregator=agg,
                        attack=atk,
                        n_clients=20,
                        clients_per_round=10,
                        n_byz=3,
                        rounds=ROUNDS,
                        seed=seed,
                    )
                    res = run(cfg)
                    row = summarize(res)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    n += 1
                    acc = row['final_global_acc']
                    w10 = row['worst10_client_acc']
                    coll = " COLLAPSED" if row['collapsed'] else ""
                    print(f"[{n}/{total}] {agg} x {atk or 'clean'} seed={seed} "
                          f"acc={acc if acc is None else f'{acc:.3f}'} "
                          f"w10={w10 if w10 is None else f'{w10:.3f}'}{coll} "
                          f"({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
