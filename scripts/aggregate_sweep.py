"""Aggregate per-variant sweep results into a comparison table + JSON summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_get(obj, *keys, default=None):
    cur = obj
    for k in keys:
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _fmt(x, prec=3):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{prec}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/sweep")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    rows = []
    for variant_dir in sorted(root.iterdir()):
        if not variant_dir.is_dir():
            continue
        eval_path = variant_dir / "eval.json"
        if not eval_path.exists():
            continue
        eval_obj = json.loads(eval_path.read_text())
        bs_path = variant_dir / "buffer_stats.json"
        bs_obj = json.loads(bs_path.read_text()) if bs_path.exists() else {}

        horizons = eval_obj.get("horizons", [])
        cos_sim = eval_obj.get("cos_sim", [])
        z_std = eval_obj.get("z_pred_std", [])
        probe_r2 = eval_obj.get("probe_r2", {}) or {}

        # find indices for k=1, k=4, k=max
        idx_k1 = horizons.index(1) if 1 in horizons else None
        idx_k4 = horizons.index(4) if 4 in horizons else None
        idx_kmax = len(horizons) - 1 if horizons else None

        cos_k4 = cos_sim[idx_k4] if idx_k4 is not None else None

        row = {
            "variant": variant_dir.name,
            "frac_lc>0": _safe_get(bs_obj, "frac>0",
                                   default=_safe_get(bs_obj, "target_distributions",
                                                     "lines_cleared", "frac>0")),
            "mean_ep_len": _safe_get(bs_obj, "mean_episode_len",
                                     default=_safe_get(bs_obj, "composition", "mean_episode_len")),
            "horizons": horizons,
            "cos_sim": cos_sim,
            "z_pred_std": z_std,
            "probe_r2": probe_r2,
            "idx_k1": idx_k1,
            "idx_k4": idx_k4,
            "idx_kmax": idx_kmax,
            "cos_k4": cos_k4,
        }
        rows.append(row)

    # sort by cos_sim @ k=4 desc (None at end)
    rows.sort(key=lambda r: (r["cos_k4"] is None, -(r["cos_k4"] or 0.0)))

    if not rows:
        print(f"no eval.json files under {root}")
        return

    # build table
    horizons_ref = rows[0]["horizons"]
    kmax = horizons_ref[-1] if horizons_ref else None

    cols = [
        ("variant", 18),
        ("frac>0", 7),
        ("ep_len", 7),
        (f"zstd_k1", 8),
        (f"zstd_k{kmax}", 8),
    ]
    for k in horizons_ref:
        cols.append((f"cos_k{k}", 8))
    for k in horizons_ref:
        cols.append((f"R²holes_k{k}", 11))
    cols.append(("R²lines_k1", 11))
    cols.append((f"R²lines_k{kmax}", 11))

    header = " | ".join(f"{name:>{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))

    summary_records = []
    for r in rows:
        cos_sim = r["cos_sim"]
        z_std = r["z_pred_std"]
        pr2 = r["probe_r2"]
        holes_r2 = pr2.get("holes", [])
        lines_r2 = pr2.get("lines_cleared", [])

        zstd_k1 = z_std[r["idx_k1"]] if r["idx_k1"] is not None and z_std else None
        zstd_kmax = z_std[r["idx_kmax"]] if r["idx_kmax"] is not None and z_std else None

        vals = [
            r["variant"],
            _fmt(r["frac_lc>0"]),
            _fmt(r["mean_ep_len"]),
            _fmt(zstd_k1),
            _fmt(zstd_kmax),
        ]
        for i in range(len(horizons_ref)):
            vals.append(_fmt(cos_sim[i] if i < len(cos_sim) else None))
        for i in range(len(horizons_ref)):
            vals.append(_fmt(holes_r2[i] if i < len(holes_r2) else None))
        vals.append(_fmt(lines_r2[r["idx_k1"]]
                         if r["idx_k1"] is not None and lines_r2 else None))
        vals.append(_fmt(lines_r2[r["idx_kmax"]]
                         if r["idx_kmax"] is not None and lines_r2 else None))

        line = " | ".join(f"{v:>{w}}" for v, (_, w) in zip(vals, cols))
        print(line)

        summary_records.append({
            "variant": r["variant"],
            "frac_lc_gt0": r["frac_lc>0"],
            "mean_episode_len": r["mean_ep_len"],
            "horizons": horizons_ref,
            "cos_sim": cos_sim,
            "z_pred_std": z_std,
            "probe_r2_holes": holes_r2,
            "probe_r2_lines_cleared": lines_r2,
            "probe_r2_aggregate_height": pr2.get("aggregate_height", []),
        })

    out_path = root / "_summary.json"
    out_path.write_text(json.dumps(summary_records, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
