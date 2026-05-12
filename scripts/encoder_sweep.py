"""Encoder architecture sweep orchestrator.

For each variant, run:
    1. JEPA training (encoder+predictor with counterfactual single-step)
    2. Probe training (frozen encoder -> lines/holes/height regression)
    3. Multistep accuracy eval (cos_sim/MSE/R² across horizons)
    4. Causality diagnostic (M1/M2/M4)

Outputs land in `results/encoder_sweep/<variant>/`. Idempotent: skips a stage
whose output file already exists. Use --rerun to force recomputation.

Phases:
    A: patch_dim width sweep
    B: architecture ablations at the winning width (residual_blocks, aux_channels)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "encoder_sweep"
TRAIN_BUFFER = "data/cf_train_100k.npz"
EVAL_BUFFER = "data/single_train_100k.npz"


@dataclass
class Variant:
    """One encoder configuration in the sweep."""

    name: str
    phase: str                                       # "A" | "B"
    patch_dim: int = 128
    residual_blocks: int = 0
    aux_channels: bool = False
    notes: str = ""

    def train_args(self) -> list[str]:
        out = [
            "--patch-dim", str(self.patch_dim),
            "--encoder-residual-blocks", str(self.residual_blocks),
        ]
        if self.aux_channels:
            out.append("--encoder-aux-channels")
        return out


def phase_a_variants() -> list[Variant]:
    return [
        Variant("A0_w128", "A", patch_dim=128, notes="default"),
        Variant("A1_w192", "A", patch_dim=192, notes="wider patches"),
        Variant("A2_w256", "A", patch_dim=256, notes="wider patches"),
        Variant("A3_w384", "A", patch_dim=384, notes="diminishing returns probe"),
    ]


def phase_b_variants(winning_width: int) -> list[Variant]:
    """Phase B fixes the Phase A winning patch_dim and ablates one architectural change."""
    return [
        Variant("B1_deep_residual", "B", patch_dim=winning_width, residual_blocks=2,
                notes="+ 2 residual blocks per stage"),
        Variant("B2_aux_channels", "B", patch_dim=winning_width, aux_channels=True,
                notes="+ hand-engineered aux channels"),
    ]


def run_cmd(cmd: list[str], log_path: Path | None = None) -> int:
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    else:
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def variant_done(variant_dir: Path) -> bool:
    return (
        (variant_dir / "jepa.pt").exists()
        and (variant_dir / "probe.pt").exists()
        and (variant_dir / "multistep.json").exists()
        and (variant_dir / "causality.json").exists()
    )


def run_variant(v: Variant, *, steps: int, eval_n: int, causality_n: int,
                rerun: bool = False, dry_run: bool = False) -> dict:
    var_dir = RESULTS_ROOT / v.name
    var_dir.mkdir(parents=True, exist_ok=True)

    config_path = var_dir / "variant.json"
    config_path.write_text(json.dumps(dataclasses.asdict(v), indent=2))

    jepa_ckpt = var_dir / "jepa.pt"
    probe_ckpt = var_dir / "probe.pt"
    multistep_json = var_dir / "multistep.json"
    causality_json = var_dir / "causality.json"
    train_log = var_dir / "train_log.jsonl"

    timings = {"variant": v.name, "phase": v.phase}

    if rerun or not jepa_ckpt.exists():
        t0 = time.time()
        cmd = [
            sys.executable, "-m", "jepa_tetris.train",
            "--buffer", TRAIN_BUFFER,
            "--counterfactual",
            "--steps", str(steps),
            "--out", str(jepa_ckpt),
            "--log-file", str(train_log),
        ] + v.train_args()
        if dry_run:
            print(f"DRY-RUN train: {' '.join(cmd)}")
        else:
            rc = run_cmd(cmd, log_path=var_dir / "train.stdout.log")
            if rc != 0:
                raise RuntimeError(f"[{v.name}] train returned {rc}")
        timings["train_sec"] = time.time() - t0
    else:
        timings["train_sec"] = "skipped"

    if rerun or not probe_ckpt.exists():
        t0 = time.time()
        cmd = [
            sys.executable, "-m", "jepa_tetris.train_probe",
            "--jepa", str(jepa_ckpt),
            "--buffer", EVAL_BUFFER,
            "--out", str(probe_ckpt),
        ]
        if dry_run:
            print(f"DRY-RUN probe: {' '.join(cmd)}")
        else:
            rc = run_cmd(cmd, log_path=var_dir / "probe.stdout.log")
            if rc != 0:
                raise RuntimeError(f"[{v.name}] probe returned {rc}")
        timings["probe_sec"] = time.time() - t0
    else:
        timings["probe_sec"] = "skipped"

    if rerun or not multistep_json.exists():
        t0 = time.time()
        cmd = [
            sys.executable, "scripts/multistep_accuracy.py",
            "--jepa", str(jepa_ckpt),
            "--probe", str(probe_ckpt),
            "--buffer", EVAL_BUFFER,
            "--n", str(eval_n),
            "--horizons", "1,2,4,8",
            "--out", str(multistep_json),
        ]
        if dry_run:
            print(f"DRY-RUN multistep: {' '.join(cmd)}")
        else:
            rc = run_cmd(cmd, log_path=var_dir / "multistep.stdout.log")
            if rc != 0:
                raise RuntimeError(f"[{v.name}] multistep returned {rc}")
        timings["multistep_sec"] = time.time() - t0
    else:
        timings["multistep_sec"] = "skipped"

    if rerun or not causality_json.exists():
        t0 = time.time()
        cmd = [
            sys.executable, "scripts/causality_diagnostic.py",
            "--jepa", str(jepa_ckpt),
            "--n", str(causality_n),
            "--out", str(causality_json),
        ]
        if dry_run:
            print(f"DRY-RUN causality: {' '.join(cmd)}")
        else:
            rc = run_cmd(cmd, log_path=var_dir / "causality.stdout.log")
            if rc != 0:
                raise RuntimeError(f"[{v.name}] causality returned {rc}")
        timings["causality_sec"] = time.time() - t0
    else:
        timings["causality_sec"] = "skipped"

    timing_path = var_dir / "timings.json"
    timing_path.write_text(json.dumps(timings, indent=2))
    return timings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["A", "B", "all", "smoke"], required=True)
    p.add_argument("--winning-width", type=int, default=128,
                   help="patch_dim picked from Phase A — used for B.")
    p.add_argument("--steps", type=int, default=50_000,
                   help="Training steps per variant.")
    p.add_argument("--eval-n", type=int, default=2000,
                   help="Sample count for multistep eval.")
    p.add_argument("--causality-n", type=int, default=500,
                   help="State count for causality diagnostic.")
    p.add_argument("--rerun", action="store_true",
                   help="Force re-run even if outputs exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Only run variants whose name matches one of these.")
    args = p.parse_args()

    if args.phase == "A":
        variants = phase_a_variants()
    elif args.phase == "B":
        variants = phase_b_variants(args.winning_width)
    elif args.phase == "smoke":
        variants = [Variant("smoke", "A", patch_dim=128, notes="200-step pipeline check")]
    else:  # "all"
        variants = phase_a_variants() + phase_b_variants(args.winning_width)

    if args.only:
        variants = [v for v in variants if v.name in set(args.only)]
        if not variants:
            raise SystemExit(f"no variants matched --only {args.only}")

    print(f"Sweep: phase={args.phase}, {len(variants)} variants, steps={args.steps}")
    print(f"Outputs: {RESULTS_ROOT}")

    all_timings = []
    for i, v in enumerate(variants, 1):
        print(f"\n=== [{i}/{len(variants)}] {v.name} ({v.phase}): {v.notes} ===")
        try:
            t = run_variant(
                v,
                steps=args.steps,
                eval_n=args.eval_n,
                causality_n=args.causality_n,
                rerun=args.rerun,
                dry_run=args.dry_run,
            )
            all_timings.append(t)
        except Exception as e:
            print(f"[{v.name}] FAILED: {e}", file=sys.stderr)
            all_timings.append({"variant": v.name, "phase": v.phase, "error": str(e)})

    summary_path = RESULTS_ROOT / f"_timings_phase_{args.phase}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_timings, indent=2))
    print(f"\nWrote sweep timings to {summary_path}")


if __name__ == "__main__":
    main()
