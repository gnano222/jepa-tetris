"""Aggregate causality + multistep results across multiple compute scales.

Takes pairs of (CF, single) result JSONs at several compute multipliers and
produces one Markdown report per metric with rows = compute scale and
columns = (single, cf, Δ). The output is the table layout used in the
counterfactual-training findings doc.

Usage:
    python scripts/scaling_summary.py \\
        --scale 0.5 --cf-causality results/..._05x.json --single-causality ... \\
                    --cf-multistep results/..._05x.json --single-multistep ... \\
        --scale 1   --cf-causality ... --single-causality ... \\
                    --cf-multistep ... --single-multistep ... \\
        --scale 3   --cf-causality ... --single-causality ... \\
                    --cf-multistep ... --single-multistep ... \\
        [--out results/scaling_summary.md]

The flags are matched positionally — the i-th `--scale` is paired with the
i-th instance of each of the four file flags.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HORIZONS_DEFAULT = (1, 2, 4, 8, 16)


def _load(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _fmt(x, places: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{places}f}"
    return str(x)


def _delta(cf, single, places: int = 4) -> str:
    if cf is None or single is None:
        return "—"
    d = cf - single
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{places}f}"


def _multistep_at(m: dict, metric: str, horizon: int) -> float | None:
    horizons = m.get("horizons", [])
    if horizon not in horizons:
        return None
    return m[metric][horizons.index(horizon)]


def render_scalar_table(
    title: str,
    rows: list[dict],
    extract,
    *,
    places: int = 4,
    direction: str = "higher_better",
) -> str:
    """One table per metric, rows = scales."""
    arrow = "↑" if direction == "higher_better" else "↓"
    headers = ["scale", "single", "cf", "Δ (cf − single)", f"better arm ({arrow})"]
    body: list[list[str]] = []
    for row in rows:
        s = extract(row["single"])
        c = extract(row["cf"])
        if s is None or c is None:
            winner = "—"
        elif direction == "higher_better":
            winner = "**cf**" if c > s else ("**single**" if s > c else "tie")
        else:
            winner = "**cf**" if c < s else ("**single**" if s < c else "tie")
        body.append([row["label"], _fmt(s, places), _fmt(c, places),
                     _delta(c, s, places), winner])
    return f"### {title}\n\n" + _md_table(headers, body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", action="append", required=True,
                        help="Label for the compute scale (e.g. '0.5x', '1x', '3x'). Repeatable.")
    parser.add_argument("--cf-causality", action="append", default=[])
    parser.add_argument("--single-causality", action="append", default=[])
    parser.add_argument("--cf-multistep", action="append", default=[])
    parser.add_argument("--single-multistep", action="append", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    n = len(args.scale)
    for name, lst in [
        ("--cf-causality", args.cf_causality),
        ("--single-causality", args.single_causality),
        ("--cf-multistep", args.cf_multistep),
        ("--single-multistep", args.single_multistep),
    ]:
        if len(lst) != n:
            raise ValueError(f"{name} has {len(lst)} entries, expected {n} to match --scale count")

    rows: list[dict] = []
    for i, label in enumerate(args.scale):
        rows.append({
            "label": label,
            "single": {
                "causality": _load(args.single_causality[i]),
                "multistep": _load(args.single_multistep[i]),
            },
            "cf": {
                "causality": _load(args.cf_causality[i]),
                "multistep": _load(args.cf_multistep[i]),
            },
        })

    sections: list[str] = ["# CF-vs-single scaling — compute parity sweep", ""]

    sections.append(render_scalar_table(
        "M1 top-1 (action retrieval; random=0.25, perfect=1.0)",
        rows,
        lambda b: b["causality"]["M1"]["top1"],
    ))
    sections.append("")
    sections.append(render_scalar_table(
        "M2 Spearman ρ (calibration of pairwise distances)",
        rows,
        lambda b: b["causality"]["M2_spearman_rho"],
    ))
    sections.append("")
    sections.append(render_scalar_table(
        "M4 ratio (no-op recognition; lower better)",
        rows,
        lambda b: b["causality"]["M4"]["ratio"],
        direction="lower_better",
    ))
    sections.append("")
    sections.append(render_scalar_table(
        "Multistep cos_sim @ k=1",
        rows,
        lambda b: _multistep_at(b["multistep"], "cos_sim", 1),
    ))
    sections.append("")
    sections.append(render_scalar_table(
        "Multistep cos_sim @ k=4",
        rows,
        lambda b: _multistep_at(b["multistep"], "cos_sim", 4),
    ))
    sections.append("")
    sections.append(render_scalar_table(
        "Multistep cos_sim @ k=16",
        rows,
        lambda b: _multistep_at(b["multistep"], "cos_sim", 16),
    ))
    sections.append("")
    sections.append(render_scalar_table(
        "Multistep mse @ k=16 (lower better)",
        rows,
        lambda b: _multistep_at(b["multistep"], "mse", 16),
        direction="lower_better",
    ))

    report = "\n".join(sections) + "\n"
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"\nwrote scaling summary to {args.out}")


if __name__ == "__main__":
    main()
