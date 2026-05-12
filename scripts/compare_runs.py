"""Side-by-side comparison report for two (or more) JEPA training runs.

Reads the JSON output of `causality_diagnostic.py --out` and
`multistep_accuracy.py --out` for each run and prints a single Markdown table
contrasting the runs across the metrics that matter for the
counterfactual-vs-single-action training comparison:

* Causality: M1 top-1, M2 Spearman ρ, M4 ratio, per-action prediction MSE.
* Multistep: cos_sim, mse, z_pred_std, off-diagonal cov at the requested horizons.

Usage:
    python scripts/compare_runs.py \\
        --label cf      --causality results/cf_caus.json    --multistep results/cf_multi.json \\
        --label single  --causality results/single_caus.json --multistep results/single_multi.json \\
        [--out results/compare_cf_vs_single.md]

The flags are matched positionally: the i-th `--label` is paired with the
i-th `--causality` and the i-th `--multistep`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ACTION_NAMES = ("LEFT", "RIGHT", "ROTATE", "DROP")


def _load(path: str | None) -> dict | None:
    if path is None:
        return None
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
    if isinstance(x, (int, bool)):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{places}f}"
    return str(x)


def render_causality_section(runs: list[dict]) -> str:
    """One-row-per-run table covering M1/M2/M4 + per-action MSE."""
    headers = ["run", "M1 top-1", "M2 ρ", "M4 ratio"] + [f"MSE({a})" for a in ACTION_NAMES]
    rows: list[list[str]] = []
    for run in runs:
        c = run["causality"]
        if c is None:
            rows.append([run["label"]] + ["—"] * (len(headers) - 1))
            continue
        m1 = c.get("M1", {}).get("top1")
        m2 = c.get("M2_spearman_rho")
        m4 = c.get("M4", {}).get("ratio")
        per_action = c.get("per_action_mse") or [None] * 4
        per_action_strs = [_fmt(x) for x in per_action]
        rows.append([run["label"], _fmt(m1), _fmt(m2), _fmt(m4)] + per_action_strs)
    return "### Causality (action distinguishability)\n\n" + _md_table(headers, rows)


def render_per_action_m1(runs: list[dict]) -> str:
    """Per-action breakdown of M1 retrieval (where most differences hide)."""
    headers = ["run"] + [f"M1({a})" for a in ACTION_NAMES]
    rows: list[list[str]] = []
    for run in runs:
        c = run["causality"]
        if c is None:
            rows.append([run["label"]] + ["—"] * 4)
            continue
        per_a = c.get("M1", {}).get("per_action") or {}
        cells = [_fmt(per_a.get(str(a)) if isinstance(per_a, dict) else per_a[a]) for a in range(4)]
        rows.append([run["label"]] + cells)
    return "### M1 retrieval — per action\n\n" + _md_table(headers, rows)


def render_multistep_section(runs: list[dict]) -> str:
    """Multistep cos_sim / mse / z_std / offdiag at each horizon, runs interleaved."""
    horizons: list[int] = []
    for run in runs:
        m = run["multistep"]
        if m is not None:
            horizons = m["horizons"]
            break
    if not horizons:
        return "### Multistep accuracy\n\n_(no multistep JSON provided)_"

    headers = ["metric", "horizon"] + [r["label"] for r in runs]
    rows: list[list[str]] = []
    for metric in ("cos_sim", "mse", "z_pred_std", "z_pred_offdiag_cov"):
        for i, k in enumerate(horizons):
            row = [metric, str(k)]
            for run in runs:
                m = run["multistep"]
                if m is None or metric not in m:
                    row.append("—")
                else:
                    places = 5 if metric == "z_pred_offdiag_cov" else 4
                    row.append(_fmt(m[metric][i], places=places))
            rows.append(row)
    return "### Multistep latent accuracy\n\n" + _md_table(headers, rows)


def render_metadata_section(runs: list[dict]) -> str:
    """Surface the JEPA paths and sample counts so it's clear which checkpoints
    were compared and on what eval data."""
    headers = ["run", "checkpoint", "causality n", "multistep n", "multistep buffer"]
    rows: list[list[str]] = []
    for run in runs:
        c = run["causality"] or {}
        m = run["multistep"] or {}
        rows.append([
            run["label"],
            _fmt(c.get("checkpoint") or m.get("jepa")),
            _fmt(c.get("n_states")),
            _fmt(m.get("n")),
            _fmt(m.get("buffer")),
        ])
    return "### Inputs\n\n" + _md_table(headers, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", action="append", required=True,
                        help="Display label for a run (repeatable, paired by order with --causality and --multistep).")
    parser.add_argument("--causality", action="append", default=[],
                        help="Path to causality_diagnostic JSON for the i-th run.")
    parser.add_argument("--multistep", action="append", default=[],
                        help="Path to multistep_accuracy JSON for the i-th run.")
    parser.add_argument("--out", default=None, help="Optional path to write the Markdown report.")
    args = parser.parse_args()

    n_runs = len(args.label)
    if not 1 <= n_runs <= 8:
        raise ValueError(f"need 1..8 --label entries, got {n_runs}")
    causality_paths = args.causality + [None] * (n_runs - len(args.causality))
    multistep_paths = args.multistep + [None] * (n_runs - len(args.multistep))

    runs = []
    for i, label in enumerate(args.label):
        runs.append({
            "label": label,
            "causality": _load(causality_paths[i]),
            "multistep": _load(multistep_paths[i]),
        })

    sections = [
        "# JEPA training-run comparison",
        "",
        render_metadata_section(runs),
        "",
        render_causality_section(runs),
        "",
        render_per_action_m1(runs),
        "",
        render_multistep_section(runs),
    ]
    report = "\n".join(sections) + "\n"
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"\nwrote report to {args.out}")


if __name__ == "__main__":
    main()
