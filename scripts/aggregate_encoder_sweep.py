"""Aggregate encoder-sweep variants into a single ranked Markdown report.

Reads `results/encoder_sweep/<variant>/{multistep.json, causality.json,
train_log.jsonl, variant.json}` and emits `results/encoder_sweep/REPORT.md`
with one row per variant. Sorts by `cos_sim @ K=4` descending.

Also computes per-variant encoder parameter counts so width-vs-architecture
trade-offs are visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_tetris.models.encoder import StateEncoder


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "results" / "encoder_sweep"


def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _last_log_line(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        last = None
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def _at_horizon(ms: dict, key: str, h: int) -> float | None:
    horizons = ms.get("horizons", [])
    if h not in horizons:
        return None
    arr = ms.get(key, [])
    idx = horizons.index(h)
    if idx >= len(arr):
        return None
    return float(arr[idx])


def _probe_r2_at(ms: dict, target: str, h: int) -> float | None:
    pr2 = ms.get("probe_r2", {})
    arr = pr2.get(target, [])
    horizons = ms.get("horizons", [])
    if h not in horizons or not arr:
        return None
    idx = horizons.index(h)
    if idx >= len(arr):
        return None
    return float(arr[idx])


def _encoder_param_count(variant_cfg: dict) -> int | None:
    try:
        enc = StateEncoder(
            patch_dim=variant_cfg["patch_dim"],
            residual_blocks=variant_cfg.get("residual_blocks", 0),
            aux_channels=variant_cfg.get("aux_channels", False),
        )
        return sum(p.numel() for p in enc.parameters())
    except Exception:
        return None


def _fmt(x, prec: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int,)):
        return str(x)
    try:
        return f"{float(x):.{prec}f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_int(x) -> str:
    if x is None:
        return "—"
    return f"{int(x):,}"


def collect_rows(root: Path) -> list[dict]:
    rows = []
    for variant_dir in sorted(root.iterdir()):
        if not variant_dir.is_dir():
            continue
        cfg = _safe_load_json(variant_dir / "variant.json")
        if cfg is None:
            continue
        ms = _safe_load_json(variant_dir / "multistep.json") or {}
        cz = _safe_load_json(variant_dir / "causality.json") or {}
        last_log = _last_log_line(variant_dir / "train_log.jsonl") or {}
        timings = _safe_load_json(variant_dir / "timings.json") or {}

        row = {
            "name": cfg["name"],
            "phase": cfg["phase"],
            "patch_dim": cfg["patch_dim"],
            "residual_blocks": cfg.get("residual_blocks", 0),
            "aux_channels": cfg.get("aux_channels", False),
            "notes": cfg.get("notes", ""),
            "encoder_params": _encoder_param_count(cfg),
            # multistep
            "cos_k1": _at_horizon(ms, "cos_sim", 1),
            "cos_k4": _at_horizon(ms, "cos_sim", 4),
            "cos_k8": _at_horizon(ms, "cos_sim", 8),
            "mse_k8": _at_horizon(ms, "mse", 8),
            "z_std_k1": _at_horizon(ms, "z_pred_std", 1),
            "z_std_k8": _at_horizon(ms, "z_pred_std", 8),
            "r2_holes_k4": _probe_r2_at(ms, "holes", 4),
            "r2_height_k4": _probe_r2_at(ms, "aggregate_height", 4),
            "r2_lines_k1": _probe_r2_at(ms, "lines_cleared", 1),
            # causality
            "m1_top1": (cz.get("M1") or {}).get("top1"),
            "m2_rho": cz.get("M2_spearman_rho"),
            "m4_ratio": (cz.get("M4") or {}).get("ratio"),
            # training log final
            "final_z_std_mean": last_log.get("z_std_mean"),
            "final_loss": last_log.get("loss"),
            "final_step": last_log.get("step"),
            # timings
            "train_sec": timings.get("train_sec"),
        }
        rows.append(row)
    return rows


def render_markdown(rows: list[dict]) -> str:
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["cos_k4"] is None, -(r["cos_k4"] or 0.0)),
    )

    out = []
    out.append("# Encoder sweep — variant ranking\n")
    out.append("Sorted by `cos_sim @ K=4` (descending). All metrics from "
               "`multistep_accuracy.py` and `causality_diagnostic.py`. Higher "
               "is better unless noted.\n")

    headers = [
        "variant", "phase", "patch_dim", "params",
        "cos@1", "cos@4", "cos@8", "mse@8↓",
        "R²holes@4", "R²height@4",
        "M1↑", "M2ρ↑", "M4↓",
        "zstd@1",
    ]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows_sorted:
        cells = [
            r["name"],
            r["phase"],
            str(r["patch_dim"]),
            _fmt_int(r["encoder_params"]),
            _fmt(r["cos_k1"]),
            _fmt(r["cos_k4"]),
            _fmt(r["cos_k8"]),
            _fmt(r["mse_k8"], prec=4),
            _fmt(r["r2_holes_k4"], prec=3),
            _fmt(r["r2_height_k4"], prec=3),
            _fmt(r["m1_top1"], prec=3),
            _fmt(r["m2_rho"], prec=3),
            _fmt(r["m4_ratio"], prec=3),
            _fmt(r["z_std_k1"], prec=3),
        ]
        out.append("| " + " | ".join(cells) + " |")

    out.append("\n## Architecture flags\n")
    flag_headers = ["variant", "res_blocks", "aux_chans", "notes"]
    out.append("| " + " | ".join(flag_headers) + " |")
    out.append("|" + "|".join(["---"] * len(flag_headers)) + "|")
    for r in rows_sorted:
        cells = [
            r["name"],
            str(r["residual_blocks"]) if r["residual_blocks"] else "",
            "✓" if r["aux_channels"] else "",
            r["notes"],
        ]
        out.append("| " + " | ".join(cells) + " |")

    out.append("\n## Top picks\n")
    if rows_sorted and rows_sorted[0]["cos_k4"] is not None:
        winner = rows_sorted[0]
        out.append(f"- **By cos_sim @ K=4**: `{winner['name']}` "
                   f"(cos@4 = {_fmt(winner['cos_k4'])}, patch_dim={winner['patch_dim']})")
    rows_with_holes = [r for r in rows_sorted if r["r2_holes_k4"] is not None]
    if rows_with_holes:
        best_holes = max(rows_with_holes, key=lambda r: r["r2_holes_k4"])
        out.append(f"- **By R² holes @ K=4**: `{best_holes['name']}` "
                   f"(R²={_fmt(best_holes['r2_holes_k4'], 3)})")
    rows_with_m2 = [r for r in rows_sorted if r["m2_rho"] is not None]
    if rows_with_m2:
        best_m2 = max(rows_with_m2, key=lambda r: r["m2_rho"])
        out.append(f"- **By M2 calibration**: `{best_m2['name']}` "
                   f"(ρ={_fmt(best_m2['m2_rho'], 3)})")

    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--out", default=None,
                   help="Output Markdown path. Defaults to <root>/REPORT.md.")
    p.add_argument("--json-out", default=None,
                   help="Optional JSON dump of the row data.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    rows = collect_rows(root)
    if not rows:
        print(f"no variants found under {root}")
        return

    md = render_markdown(rows)
    print(md)

    out_path = Path(args.out) if args.out else (root / "REPORT.md")
    out_path.write_text(md)
    print(f"wrote {out_path}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
