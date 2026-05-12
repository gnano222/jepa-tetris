# jepa-tetris RunPod operations
# Usage: make <target> [GPU=...] [STEPS=...]
#
# Prerequisites: pip install runpod && cp .env.runpod.example .env.runpod && fill in values

GPU    ?= NVIDIA RTX 4500
STEPS  ?= 50000
.PHONY: help train train-a100 train-h100 stop delete status upload-data get-checkpoints get-results ssh logs

help:
	@echo "RunPod targets:"
	@echo "  make train              Start RTX 4500 training pod (default GPU)"
	@echo "  make train GPU=...      Start pod with specific GPU type"
	@echo "  make train-a100         Start A100 SXM training pod"
	@echo "  make train-h100         Start H100 SXM training pod"
	@echo "  make stop               Stop the current pod (preserves /workspace)"
	@echo "  make delete             Permanently delete the current pod"
	@echo "  make status             List all pods and their state"
	@echo "  make upload-data        Upload local data/*.npz to network volume"
	@echo "  make get-checkpoints    Download checkpoints from pod to local ./checkpoints/"
	@echo "  make get-results        Download results/ from pod to local ./results/"
	@echo "  make ssh                SSH into the running pod"
	@echo "  make logs               Tail the startup log from the running pod"
	@echo ""
	@echo "Set GPU type:  make train GPU='NVIDIA A100 80GB SXM'"
	@echo "Set steps:     make train STEPS=200000"

train:
	JEPA_STEPS=$(STEPS) python scripts/runpod_pod.py create --gpu "$(GPU)"

train-a100:
	JEPA_STEPS=$(STEPS) python scripts/runpod_pod.py create --gpu "NVIDIA A100 80GB SXM"

train-h100:
	JEPA_STEPS=$(STEPS) python scripts/runpod_pod.py create --gpu "NVIDIA H100 80GB HBM3"

stop:
	python scripts/runpod_pod.py stop

delete:
	python scripts/runpod_pod.py delete

status:
	python scripts/runpod_pod.py status

# POD_SSH = "user@host [ssh-options]"
# Use the DIRECT TCP connection from RunPod Connect tab (not the ssh.runpod.io proxy).
# Example: POD_SSH="root@174.x.x.x -p 22022 -i ~/.ssh/id_ed25519"
# For rsync: host = $(firstword $(POD_SSH)), ssh_opts = $(wordlist 2,$(words $(POD_SSH)),$(POD_SSH))
upload-data:
	@[ -n "$(POD_SSH)" ] || (echo "Set POD_SSH='root@IP -p PORT -i ~/.ssh/id_ed25519' (use direct TCP from RunPod Connect tab)" && exit 1)
	rsync -avz --progress \
	  -e "ssh $(wordlist 2,$(words $(POD_SSH)),$(POD_SSH))" \
	  data/ $(firstword $(POD_SSH)):/workspace/data/

get-checkpoints:
	@[ -n "$(POD_SSH)" ] || (echo "Set POD_SSH='root@IP -p PORT -i ~/.ssh/id_ed25519' (use direct TCP from RunPod Connect tab)" && exit 1)
	rsync -avz --progress \
	  -e "ssh $(wordlist 2,$(words $(POD_SSH)),$(POD_SSH))" \
	  $(firstword $(POD_SSH)):/workspace/checkpoints/ ./checkpoints/

get-results:
	@[ -n "$(POD_SSH)" ] || (echo "Set POD_SSH='root@IP -p PORT -i ~/.ssh/id_ed25519' (use direct TCP from RunPod Connect tab)" && exit 1)
	rsync -avz --progress \
	  -e "ssh $(wordlist 2,$(words $(POD_SSH)),$(POD_SSH))" \
	  $(firstword $(POD_SSH)):/workspace/results/ ./results/

ssh:
	@[ -n "$(POD_SSH)" ] || (echo "Set POD_SSH='user@ssh.runpod.io -i ~/.ssh/id_ed25519'" && exit 1)
	ssh $(POD_SSH)

logs:
	@[ -n "$(POD_SSH)" ] || (echo "Set POD_SSH='user@ssh.runpod.io -i ~/.ssh/id_ed25519'" && exit 1)
	ssh $(POD_SSH) "tail -f /workspace/startup.log"
