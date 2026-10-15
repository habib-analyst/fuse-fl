"""Build the markdown audit report card from results/grid.jsonl."""

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
GRID = HERE.parent / "results" / "grid.jsonl"
REPORT = HERE.parent / "results" / "REPORT_CARD.md"


def main():
    rows = [json.loads(l) for l in GRID.read_text().splitlines() if l.strip()]
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[(r["aggregator"], r["seed"])].append(r)

    out = ["# fuse-fl audit report card", ""]
    out.append("MNIST/MLP, Dirichlet(alpha=1.0) label skew, 20 clients (10/round), "
               "3 Byzantine Sybils/round, q=1, f=3, 12 rounds, seeds 40-41.")
    out.append("")
    out.append("## Seed-wise worst-attack accuracy (the paper's primary statistic)")
    out.append("")
    out.append("| aggregator | worst attack (seed 40) | acc | worst attack (seed 41) | acc |")
    out.append("|---|---|---|---|---|")
    for agg in ["fedavg", "q-ffl", "trimmed-mean", "krum", "fltrust", "h-nobs", "boba", "fuse"]:
        cells = [agg]
        for seed in (40, 41):
            attacked = [r for r in by_seed[(agg, seed)]
                        if r["attack"] != "clean" and r["final_global_acc"] is not None]
            if not attacked:
                cells += ["n/a", "n/a"]
                continue
            w = min(attacked, key=lambda r: r["final_global_acc"])
            cells += [w["attack"], f"{w['final_global_acc']:.3f}"]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("## Clean accuracy (no attack)")
    out.append("")
    out.append("| aggregator | seed 40 global / worst-10% | seed 41 global / worst-10% |")
    out.append("|---|---|---|")
    for agg in ["fedavg", "q-ffl", "trimmed-mean", "krum", "fltrust", "h-nobs", "boba", "fuse"]:
        cells = [agg]
        for seed in (40, 41):
            r = next((x for x in by_seed[(agg, seed)] if x["attack"] == "clean"), None)
            cells.append(f"{r['final_global_acc']:.3f} / {r['worst10_client_acc']:.3f}" if r else "n/a")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("## Screening audit: Byzantine rejected vs high-loss honest rejected")
    out.append("")
    out.append("(oracle-audited; mean over attacked rounds; soft methods report n/a)")
    out.append("")
    out.append("| aggregator | attack | seed | byz rejected | high-loss honest rejected |")
    out.append("|---|---|---|---|---|")
    def _f(v, nd=2):
        return "n/a" if v is None or v != v else f"{v:.{nd}f}"

    for r in rows:
        if r["attack"] == "clean":
            continue
        br = _f(r["byz_rejected_frac"])
        hr = _f(r["highloss_honest_rejected_frac"])
        out.append(f"| {r['aggregator']} | {r['attack']} | {r['seed']} | {br} | {hr} |")
    out.append("")
    (REPORT).write_text("\n".join(out) + "\n")
    print("\n".join(out[:22]))


if __name__ == "__main__":
    main()
