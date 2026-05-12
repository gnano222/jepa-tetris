#!/usr/bin/env python3
"""RunPod pod lifecycle management for jepa-tetris training."""
import argparse
import os
import sys

import runpod

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POD_ID_FILE = os.path.join(_REPO_ROOT, ".pod_id")


def load_env():
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env.runpod")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def get_required(key):
    val = os.environ.get(key)
    if not val:
        sys.exit(f"Error: {key} not set. Copy .env.runpod.example to .env.runpod and fill in values.")
    return val


def cmd_create(args):
    load_env()
    runpod.api_key = get_required("RUNPOD_API_KEY")
    volume_id     = get_required("RUNPOD_VOLUME_ID")
    image         = get_required("RUNPOD_IMAGE")
    github_repo   = get_required("GITHUB_REPO")

    gpu_type = args.gpu or "NVIDIA RTX 4090"
    branch   = args.branch or os.environ.get("JEPA_BRANCH", "main")

    # Bootstrap: always pull the branch-specific code BEFORE running pod_startup.sh.
    # This avoids the stale-cache problem where the network volume's cached
    # pod_startup.sh lags behind the branch being trained.
    startup_cmd = (
        "bash -c 'set -euo pipefail; "
        "REPO_DIR=/workspace/jepa-tetris; "
        "if [ -d $REPO_DIR/.git ]; then "
        "  git -C $REPO_DIR fetch origin && "
        "  git -C $REPO_DIR checkout ${JEPA_BRANCH:-main} && "
        "  git -C $REPO_DIR pull --ff-only origin ${JEPA_BRANCH:-main}; "
        "else "
        "  git clone --branch ${JEPA_BRANCH:-main} $GITHUB_REPO $REPO_DIR; "
        "fi && "
        "exec bash $REPO_DIR/scripts/pod_startup.sh' 2>&1 | tee /workspace/startup.log"
    )

    env_vars = {
        "GITHUB_REPO":    github_repo,
        "JEPA_BRANCH":    branch,
        "JEPA_BUFFER":    os.environ.get("JEPA_BUFFER",     "data/buffer.npz"),
        "JEPA_STEPS":     os.environ.get("JEPA_STEPS",      "50000"),
        "JEPA_HORIZON":   os.environ.get("JEPA_HORIZON",    "4"),
        "JEPA_RUN":       os.environ.get("JEPA_RUN",        branch),
        "JEPA_OUT":       os.environ.get("JEPA_OUT",        f"checkpoints/jepa-{branch}.pt"),
        "JEPA_EXTRA_ARGS": os.environ.get("JEPA_EXTRA_ARGS", ""),
    }

    pod_name = f"jepa-{branch[:20]}"

    pod = runpod.create_pod(
        name=pod_name,
        image_name=image,
        gpu_type_id=gpu_type,
        cloud_type="SECURE",
        network_volume_id=volume_id,
        container_disk_in_gb=20,
        volume_in_gb=0,
        volume_mount_path="/workspace",
        docker_args=startup_cmd,
        env=env_vars,
        ports="22/tcp",
    )
    pod_id = pod["id"]
    with open(_POD_ID_FILE, "w") as f:
        f.write(pod_id)
    print(f"Pod created: {pod_id}  ({pod_name})")
    print(f"Branch: {branch} | GPU: {gpu_type} | Image: {image}")
    print(f"Checkpoint: checkpoints/jepa-{branch}.pt")
    print("Startup log: /workspace/startup.log")


def cmd_stop(args):
    load_env()
    runpod.api_key = get_required("RUNPOD_API_KEY")
    pod_id = _read_pod_id()
    runpod.stop_pod(pod_id)
    print(f"Pod {pod_id} stopped.")


def cmd_delete(args):
    load_env()
    runpod.api_key = get_required("RUNPOD_API_KEY")
    pod_id = _read_pod_id()
    confirm = input(f"Delete pod {pod_id}? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return
    runpod.terminate_pod(pod_id)
    os.remove(_POD_ID_FILE)
    print(f"Pod {pod_id} deleted.")


def cmd_status(args):
    load_env()
    runpod.api_key = get_required("RUNPOD_API_KEY")
    pods = runpod.get_pods()
    if not pods:
        print("No pods found.")
        return
    for p in pods:
        print(f"{p['id']}  {p.get('name','?'):20s}  {p.get('desiredStatus','?')}")


def _read_pod_id():
    if not os.path.exists(_POD_ID_FILE):
        sys.exit("No .pod_id file found. Run `make train` first.")
    with open(_POD_ID_FILE) as f:
        return f.read().strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    p_create = sub.add_parser("create")
    p_create.add_argument("--gpu", default=None, help="GPU type ID (default: NVIDIA RTX 4090)")
    p_create.add_argument("--branch", default=None, help="Git branch to run (default: main)")
    sub.add_parser("stop")
    sub.add_parser("delete")
    sub.add_parser("status")
    args = parser.parse_args()
    dispatch = {"create": cmd_create, "stop": cmd_stop, "delete": cmd_delete, "status": cmd_status}
    if args.cmd not in dispatch:
        parser.print_help()
        sys.exit(1)
    dispatch[args.cmd](args)
