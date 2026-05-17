"""Build a convergence curve across a series of training checkpoints.

`train.py` saves `jepa_step{N}.pt` every `--ckpt-every` steps. This script
runs the existing multistep-accuracy and causality diagnostics on each of
those checkpoints and collects the results into a single JSON keyed by
training step — the data behind a "metric vs. training steps" plot used to
judge whether one model converges faster than another.

Intermediate checkpoints are named `<out-stem>_step{N}.pt` (e.g.
`jepa-exp-tokengate-k6_step5000.pt`), so a glob targets one run cleanly.

Usage:
    python scripts/convergence_curve.py \\
        --checkpoints "checkpoints/jepa-exp-tokengate-k6_step*.pt" \\
        --buffer data/buffer.npz \\
        --out results/tokengate_k6_curve.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def ckpt_step(path: str) -> int:
    """Training step for a checkpoint — from the stored `step`, else the filename."""
    try:
        step = torch.load(path, map_location="cpu", weights_only=False).get("step")
        if step is not None:
            return int(step)
    except Exception:
        pass
    m = re.search(r"step(\d+)", Path(path).stem)
    return int(m.group(1)) if m else -1


def _run(script: str, extra: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *extra],
        check=True,
        cwd=REPO_ROOT,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoint paths or globs, e.g. 'checkpoints/jepa_step*.pt'.")
    p.add_argument("--buffer", required=True, help="Buffer for the multistep eval.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument("--causality-n", type=int, default=500,
                   help="States sampled for the causality diagnostic per checkpoint.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-causality", action="store_true",
                   help="Run only the multistep eval (faster).")
    args = p.parse_args()

    paths: list[Path] = []
    for c in args.checkpoints:
        if any(ch in c for ch in "*?["):
            paths.extend(sorted(Path().glob(c)))
        else:
            paths.append(Path(c))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"no checkpoints matched: {args.checkpoints}")

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        ms_path = f"{td}/ms.json"
        ca_path = f"{td}/ca.json"
        for path in paths:
            step = ckpt_step(str(path))
            print(f"--- {path.name} (step {step}) ---")

            _run("multistep_accuracy.py",
                 ["--jepa", str(path), "--buffer", args.buffer, "--out", ms_path])
            ms = json.loads(Path(ms_path).read_text())
            row: dict = {
                "step": step,
                "checkpoint": str(path),
                "horizons": ms["horizons"],
                "cos_sim": ms["cos_sim"],
                "mse": ms["mse"],
                "per_action_mse_k1": {k: v[0] for k, v in ms["per_action_mse"].items()},
            }

            if not args.skip_causality:
                _run("causality_diagnostic.py",
                     ["--jepa", str(path), "--n", str(args.causality_n),
                      "--seed", str(args.seed), "--out", ca_path])
                ca = json.loads(Path(ca_path).read_text())
                row["M1"] = ca["M1"]["top1"]
                row["M2"] = ca["M2_spearman_rho"]
                row["M4"] = ca["M4"]["ratio"]

            rows.append(row)

    rows.sort(key=lambda r: r["step"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"curve": rows}, indent=2))

    # Compact summary table.
    def at(row: dict, k: int) -> float:
        hs = row["horizons"]
        return row["cos_sim"][hs.index(k)] if k in hs else float("nan")

    print(f"\n{'step':>8} | {'cos@1':>7} | {'cos@4':>7} | {'cos@16':>7} | "
          f"{'M1':>6} | {'M2':>6}")
    print("-" * 56)
    for row in rows:
        m1 = row.get("M1", float("nan"))
        m2 = row.get("M2", float("nan"))
        print(f"{row['step']:>8} | {at(row, 1):>7.4f} | {at(row, 4):>7.4f} | "
              f"{at(row, 16):>7.4f} | {m1:>6.3f} | {m2:>6.3f}")
    print(f"\nwrote curve to {out_path}")


if __name__ == "__main__":
    main()
