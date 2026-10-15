"""Audit report card: fairness + robustness side by side, as markdown."""

from __future__ import annotations


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.{nd}f}"


def report_card(rows: list[dict], title: str = "FUSE-FL audit report card") -> str:
    """Render the robustness/fairness matrix as a markdown table."""
    lines = [f"# {title}", ""]
    lines.append(
        "| aggregator | attack | seed | global acc | worst-10% client acc | "
        "client acc var | byz rejected | high-loss honest rejected | seconds |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['aggregator']} | {r['attack']} | {r['seed']} | "
            f"{_fmt(r['final_global_acc'])} | {_fmt(r['worst10_client_acc'])} | "
            f"{_fmt(r['client_acc_variance'], 5)} | {_fmt(r['byz_rejected_frac'])} | "
            f"{_fmt(r['highloss_honest_rejected_frac'])} | {_fmt(r['seconds'], 1)} |"
        )
    lines.append("")
    lines.append(
        "Notes: `byz rejected` / `high-loss honest rejected` are oracle-audited "
        "(simulation ground truth). Untargeted attacks have no ASR; robustness is "
        "measured as retained accuracy under attack."
    )
    return "\n".join(lines) + "\n"


def worst_attack_summary(rows: list[dict]) -> str:
    """Seed-wise worst-attack table, in the style of the FUSE paper's Table 1."""
    from collections import defaultdict

    by = defaultdict(list)
    for r in rows:
        by[(r["aggregator"], r["seed"])].append(r)
    lines = ["# Seed-wise worst-attack summary", ""]
    lines.append("| aggregator | seed | worst attack | worst-attack global acc |")
    lines.append("|---|---|---|---|")
    for (agg, seed), rs in sorted(by.items()):
        attacked = [r for r in rs if r["attack"] != "clean"]
        if not attacked:
            continue
        worst = min(attacked, key=lambda r: r["final_global_acc"])
        lines.append(
            f"| {agg} | {seed} | {worst['attack']} | {_fmt(worst['final_global_acc'])} |"
        )
    return "\n".join(lines) + "\n"
