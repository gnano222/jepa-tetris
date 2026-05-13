---
name: runpod-training-workflow
description: Use when launching GPU training runs on RunPod for jepa-tetris, uploading data to the network volume, running parallel branch-based experiments, retrieving checkpoints, or diagnosing why a pod didn't train.
---

# RunPod Training Workflow

## Overview

Training runs on RunPod via a Docker image with pre-installed deps. Code is git-cloned at pod startup — you push a branch, pod pulls it, trains, saves checkpoint to the network volume, then self-stops. The network volume (`/workspace`) persists data and checkpoints across pod cycles.

## Architecture

```
Local machine                    RunPod pod
─────────────────               ──────────────────────────────
git push branch      ──────►    pod_startup.sh:
make train BRANCH=X             1. git clone/checkout branch
                                2. pip install -e .
                                3. symlink /workspace/{data,checkpoints,results}
                                4. python -m jepa_tetris.train ...
                                5. runpodctl pod stop (auto-stop)
```

Key files:
- `scripts/pod_startup.sh` — runs inside pod, handles git + training + stop
- `scripts/runpod_pod.py` — local SDK wrapper: create/stop/delete/status
- `Makefile` — one-command interface
- `.env.runpod` — secrets (gitignored): API key, volume ID, image, repo URL

## Single Experiment

```bash
git add -p && git commit -m "..." && git push
make train                          # RTX 4500, 50k steps, branch=main
make train STEPS=200000             # longer run
make train-a100 STEPS=200000        # A100 SXM
make status                         # list running pods
```

## Parallel Experiments (branch per experiment)

Each branch → separate pod → separate checkpoint file.

```bash
git checkout -b exp-my-idea
# ... edit code ...
git commit -am "try X" && git push origin exp-my-idea

make train BRANCH=exp-my-idea STEPS=100000
make train BRANCH=exp-other-idea STEPS=100000   # second pod simultaneously
```

Pod is named `jepa-<branch>` — visible in `make status`.
Checkpoint lands at `checkpoints/jepa-<branch>.pt` — no collisions.

> **Note:** `make stop` tracks only the last-launched pod. Stop others via the RunPod dashboard.

## Retrieving Results

**SSH connection string must be the direct TCP format** — NOT the `ssh.runpod.io` proxy (which requires an interactive PTY and blocks rsync/scp).

RunPod dashboard → pod → Connect → **SSH over exposed TCP** → looks like:
`root@174.x.x.x -p 22022`

```bash
# Pull checkpoints after pod auto-stops
make get-checkpoints POD_SSH="root@174.x.x.x -p 22022 -i ~/.ssh/id_ed25519"
make get-results     POD_SSH="root@174.x.x.x -p 22022 -i ~/.ssh/id_ed25519"

# Watch training in real time
make logs POD_SSH="root@174.x.x.x -p 22022 -i ~/.ssh/id_ed25519"
```

## Uploading Data (one-time per volume)

Spin up a cheap CPU pod with the network volume attached, get its direct TCP SSH string, then:

```bash
make upload-data POD_SSH="root@174.x.x.x -p 22022 -i ~/.ssh/id_ed25519"
```

## Common Issues

| Symptom | Fix |
|---|---|
| Pod starts but no training | `make logs POD_SSH=...` → check `/workspace/startup.log` |
| `data/buffer.npz` not found | Run `make upload-data` with a CPU pod attached to the volume |
| `ssh.runpod.io` gives "doesn't support PTY" | Use direct TCP from Connect tab, not the proxy |
| rsync fails with "unexpected tag" banner error | Same — use direct TCP only |
| `git pull` fails (private repo) | Use `https://TOKEN@github.com/USER/REPO.git` as `GITHUB_REPO` in `.env.runpod` |
| Pod won't auto-stop | `make stop` or stop via dashboard; check `RUNPOD_POD_ID` is set in pod env |
