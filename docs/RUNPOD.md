# RunPod GPU Training Setup

One-time setup (~15 min). After setup, training is `make train` + `make get-checkpoints`.

## Prerequisites

- RunPod account at runpod.io (add $20 credit to start)
- Docker installed locally (for building the image)
- Docker Hub account (free tier is fine)
- SSH key added to RunPod account: runpod.io → Settings → SSH Public Keys → Add

## Step 1: Install tools

```bash
# RunPod Python SDK (used by Makefile scripts locally)
pip install runpod
```

> **Note:** `runpodctl` is only needed *inside* pods (for auto-stop) — it comes pre-installed there. You do not need to install or configure it locally.

## Step 2: Build and push Docker image

```bash
# Build for linux/amd64 (RunPod's platform)
docker buildx build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/jepa-tetris-runpod:latest .
docker push YOUR_DOCKERHUB_USER/jepa-tetris-runpod:latest
```

## Step 3: Create Network Volume

In the RunPod dashboard:
1. Storage → Network Volumes → + New Network Volume
2. Name: `jepa-tetris-data`
3. Size: 50 GB (covers data + checkpoints + results with room to grow)
4. Region: pick one with RTX 4090 / A100 availability (US-TX or EU-SE are good)
5. Copy the Volume ID (looks like `abc123xyz`)

Cost: ~$3.50/month for 50 GB — billed whether the pod is running or not.

## Step 4: Upload data files to the volume

Spin up a cheap CPU pod with the volume to upload data:

1. In RunPod → Pods → + Deploy → search "Ubuntu" → pick cheapest CPU pod
2. Attach the volume at `/workspace`
3. Deploy, then get the SSH over exposed TCP connection string from Connect tab
4. Run locally:

```bash
POD_SSH="root@213.173.105.99 -p 49059 -i ~/.ssh/id_ed25519" make upload-data
```


5. Stop and delete the CPU pod (you just used it as a file upload gateway).

## Step 5: Configure .env.runpod

```bash
cp .env.runpod.example .env.runpod
# Edit .env.runpod and fill in:
#   RUNPOD_API_KEY    — from runpod.io → Settings → API Keys
#   RUNPOD_VOLUME_ID  — from Step 3
#   RUNPOD_IMAGE      — YOUR_DOCKERHUB_USER/jepa-tetris-runpod:latest
#   GITHUB_REPO       — your repo URL
```

## Daily workflow

```bash
# Push code changes
git push

# Start a training run (RTX 4090, 50k steps)
make train

# Or with more steps on an A100
make train-a100 STEPS=200000

# Watch what's happening
make status
make logs POD_SSH="root@<ip> -p <port> -i ~/.ssh/id_ed25519"

# After training completes (pod auto-stops), pull results
make get-checkpoints POD_SSH="root@<ip> -p <port> -i ~/.ssh/id_ed25519"
make get-results POD_SSH="..."
```

> **SSH connection string:** use the **direct TCP** format from RunPod Connect tab (not the `ssh.runpod.io` proxy). It looks like `root@174.x.x.x -p 22022`.

## Running parallel experiments

Each experiment lives on its own git branch. Push the branch, then launch a pod pointing at it. Multiple pods can run different branches simultaneously.

```bash
# 1. Create a branch for your experiment
git checkout -b exp-attention-pooling
# ... make your code changes ...
git add -p && git commit -m "try attention pooling in predictor"
git push origin exp-attention-pooling

# 2. Launch a pod on that branch (runs alongside any other running pods)
make train BRANCH=exp-attention-pooling STEPS=100000

# 3. Launch a second experiment on a different branch at the same time
make train BRANCH=exp-wider-predictor STEPS=100000
```

Each pod is named after its branch (`jepa-exp-attention-pooling`) so `make status` shows clearly what's running:

```
abc123  jepa-exp-attention-pooling   RUNNING
def456  jepa-exp-wider-predictor     RUNNING
```

Checkpoints land in separate files by branch — `checkpoints/jepa-exp-attention-pooling.pt` and `checkpoints/jepa-exp-wider-predictor.pt` — so they don't overwrite each other.

```bash
# Pull results for a specific experiment
make get-checkpoints POD_SSH="root@<ip> -p <port> -i ~/.ssh/id_ed25519"

# Compare runs locally
python -m jepa_tetris.eval --jepa checkpoints/jepa-exp-attention-pooling.pt ...
python -m jepa_tetris.eval --jepa checkpoints/jepa-exp-wider-predictor.pt ...
```

> **Stopping parallel pods:** `make stop` only tracks the most recently launched pod. To stop others, use the RunPod dashboard → Pods → Stop, or `make status` to get pod IDs then stop via the dashboard.

## GPU options and cost

| Command | GPU | VRAM | ~$/hr | 50k steps |
|---|---|---|---|---|
| `make train` | RTX 4090 | 24 GB | $0.34–$0.69 | ~$0.40 |
| `make train-a100` | A100 SXM | 80 GB | $1.39 | ~$0.80 |
| `make train-h100` | H100 SXM | 80 GB | $2.69 | ~$1.50 |

## Troubleshooting

**Pod starts but training doesn't run:** SSH in and check `/workspace/startup.log`.  
**`git pull` fails (private repo):** Add a GitHub Personal Access Token to RunPod Secrets, then use `https://TOKEN@github.com/USER/REPO.git` as `GITHUB_REPO`.  
**`data/buffer.npz` not found:** Run `make upload-data` to populate the network volume.  
**Pod won't stop:** `make stop` or go to RunPod dashboard → Pods → Stop.
