---
name: runpod-training-workflow
description: Use when launching GPU training runs on RunPod for jepa-tetris, uploading data to the network volume, running parallel branch-based experiments, retrieving checkpoints, or diagnosing why a pod didn't train.
---

# RunPod Training Workflow

## Overview

Training runs on RunPod via a Docker image with pre-installed deps. Code is git-cloned at pod startup — you push a branch, pod pulls it, trains, saves checkpoint to the network volume, then self-stops. The network volume (`/workspace`) persists data and checkpoints across pod cycles.

**Results download automatically.** After `make train`, a background watcher polls RunPod every 30 s. When the pod exits it spins up a CPU pod, rsyncs results and checkpoints locally, terminates the CPU pod, and fires a macOS notification. No manual action needed.

## Architecture

```
Local machine                    RunPod pod
─────────────────               ──────────────────────────────
git push branch      ──────►    docker_args bootstrap (API-set, never cached):
make train BRANCH=X             1. rm stale git locks
  └─ background watcher         2. git fetch + checkout -f + reset --hard
     polls every 30 s              (or clone if repo missing)
     → auto-download on exit    3. exec pod_startup.sh (from freshly-pulled branch)
     → macOS notification          a. pip install -e .
                                   b. symlink /workspace/{data,checkpoints,results}
                                   c. python -m jepa_tetris.train ...
                                   d. runpodctl pod stop || true  (auto-stop, no crash)
```

Key files:
- `scripts/runpod_pod.py` — local SDK wrapper: create/stop/delete/status/download/watch
- `scripts/pod_startup.sh` — runs inside pod: pip install + training + auto-stop
- `Makefile` — one-command interface
- `.env.runpod` — secrets (gitignored): API key, volume ID, image, repo URL
- `~/.jepa_watcher.log` — background watcher log

## Single Experiment

```bash
git add -p && git commit -m "..." && git push
make train                          # RTX A4500, 50k steps, branch=main
make train STEPS=200000             # longer run
make train-a100 STEPS=200000        # A100 SXM
make status                         # list running pods
# results appear automatically in ./checkpoints/ and ./results/ when done
```

## Parallel Experiments (branch per experiment)

Each branch → separate pod → separate checkpoint file. The watcher tracks the **last-launched** pod only.

```bash
git checkout -b exp-my-idea
# ... edit code ...
git commit -am "try X" && git push origin exp-my-idea

make train BRANCH=exp-my-idea STEPS=100000
make train BRANCH=exp-other-idea STEPS=100000   # second pod simultaneously
```

Pod is named `jepa-<branch>` — visible in `make status`.
Checkpoint lands at `checkpoints/jepa-<branch>.pt` — no collisions.

> **Note:** For parallel runs, `make stop` and the auto-watcher track only the last-launched pod via `.pod_id`. Stop others via the RunPod dashboard.

## Downloading Results

**Automatic (default):** results download when training completes. Check `~/.jepa_watcher.log` to see watcher status.

**Manual fallback:**
```bash
make download   # useful if watcher died or you need a mid-run snapshot
```

Both methods spin up a `cpu3c-2-4` CPU pod (2 vCPU, 4 GB RAM), rsync `/workspace/results/` and `/workspace/checkpoints/` locally, then terminate the CPU pod. Uses `~/.ssh/id_ed25519` by default; override with `SSH_KEY=/path/to/key`.

CPU pod image: `runpod/base:1.0.3-ubuntu2204`. Falls back to `cpu5c-2-4` then `cpu3c-4-8` if unavailable.

## Extra Training Flags

Set `JEPA_EXTRA_ARGS` to pass additional flags to `jepa_tetris.train`:

```bash
JEPA_EXTRA_ARGS="--encoder-stride-stages 2 --encoder-two-scale --batch-size 256" \
  make train BRANCH=exp-my-idea
```

These are forwarded as pod env vars and appended to the training command in `pod_startup.sh`.

**Always include `--batch-size 256` in `JEPA_EXTRA_ARGS`** — the training default is 2048, not 256. The established baseline (two-scale-50k) used batch 256. Forgetting this means 8× more data per step and an unfair comparison. Every experiment JEPA_EXTRA_ARGS should contain `--batch-size 256` unless deliberately testing a different batch size.

## Named Checkpoint Files

By default both pods and `make train` write to `checkpoints/jepa.pt` on the shared network volume — pods running in parallel will overwrite each other. To prevent this, set `JEPA_OUT` before launching:

```bash
JEPA_OUT="checkpoints/jepa-exp-film.pt" \
  JEPA_EXTRA_ARGS="--encoder-stride-stages 2 --encoder-two-scale --predictor-film --batch-size 256" \
  make train BRANCH=exp-film
```

The `JEPA_OUT` env var is read by `runpod_pod.py` and passed to the pod, which passes it to `pod_startup.sh` as `--out`. Each parallel run should get a unique name. Naming convention: `jepa-<branch>.pt` (matches the auto-set default when using `make train BRANCH=...`).

## Sample Budget and Batch Size

**Always compare at the same batch size AND step count.** `N samples = steps × batch_size`, but a run with batch 2048 at 6250 steps is NOT equivalent to batch 256 at 50000 steps — fewer gradient updates means worse convergence per sample even with identical total data seen. The established baseline uses **batch 256, 50k steps**. Match both when doing architecture comparisons.

## Uploading Data (one-time per volume)

```bash
make upload-data POD_SSH="root@174.x.x.x -p 22022 -i ~/.ssh/id_ed25519"
```

Requires a running pod (CPU or GPU) with the volume attached. Get the direct TCP SSH string from RunPod dashboard → pod → Connect.

## Common Issues

| Symptom | Fix |
|---|---|
| Watcher didn't download | Check `~/.jepa_watcher.log`. If watcher died, run `make download` manually |
| `manifest unknown` when CPU pod starts | Image tag wrong. Valid: `runpod/base:1.0.3-ubuntu2204` |
| Pod crashes instantly (uptimeInSeconds: 2) | Stale `.git/index.lock` on network volume. Fixed in bootstrap: `rm -f .git/index.lock .git/config.lock` |
| Pod restarts after training completes | `runpodctl pod stop` failing under `set -euo pipefail`. Fixed: `|| true` + `sleep 30` in `pod_startup.sh` |
| `data/buffer.npz` not found | Run `make upload-data` with a pod attached to the volume |
| Can't SSH into training pod | Training image has no openssh-server. Use `make download` instead |
| Download pod SSH times out | sshd slow to start; polls up to 5 min. If it fails, try again |
| `make download` leaves orphaned CPU pod | `finally` block terminates it. If interrupted, check `make status` and terminate via dashboard |
| `git pull` fails (private repo) | Use `https://TOKEN@github.com/USER/REPO.git` as `GITHUB_REPO` in `.env.runpod` |
| Run took way longer than expected | Check `JEPA_EXTRA_ARGS` — `--ar-weight 0.25` adds sequential AR rollout (~6× slower at large batch). Reduce steps proportionally or use batch 256 |
| Results not comparable to baseline | Missing `--batch-size 256` in `JEPA_EXTRA_ARGS`. Default is 2048 — 8× more data per step than the established baseline. Always add `--batch-size 256` explicitly. |
| Parallel pods overwrote each other's checkpoint | Both pods default to `checkpoints/jepa.pt`. Set `JEPA_OUT=checkpoints/jepa-<branch>.pt` when launching to give each run a unique file. |
