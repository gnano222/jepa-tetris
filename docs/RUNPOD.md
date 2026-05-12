# RunPod GPU Training Setup

One-time setup (~15 min). After setup, training is `make train` + `make get-checkpoints`.

## Prerequisites

- RunPod account at runpod.io (add $20 credit to start)
- Docker installed locally (for building the image)
- Docker Hub account (free tier is fine)
- SSH key added to RunPod: Settings → SSH Keys → Add

## Step 1: Install tools

```bash
# RunPod CLI
brew install runpod/runpodctl/runpodctl   # macOS

# RunPod Python SDK
pip install runpod

# Configure CLI with your API key (from runpod.io → Settings → API Keys)
runpodctl config --apiKey=YOUR_API_KEY
```

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
3. Deploy, then get the SSH connection string from Connect tab
4. Run locally:

```bash
POD_SSH="root@<ip> -p <port> -i ~/.ssh/id_rsa" make upload-data
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
make logs POD_SSH="root@<ip> -p <port> -i ~/.ssh/id_rsa"

# After training completes (pod auto-stops), pull results
make get-checkpoints POD_SSH="..."
make get-results POD_SSH="..."
```

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
