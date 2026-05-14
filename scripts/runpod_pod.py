#!/usr/bin/env python3
"""RunPod pod lifecycle management for jepa-tetris training."""
import argparse
import os
import subprocess
import sys
import time

import runpod

_WATCHER_LOG = os.path.join(os.path.expanduser("~"), ".jepa_watcher.log")

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

    branch   = args.branch or os.environ.get("JEPA_BRANCH", "main")

    # Ordered GPU preference list — tried in order until one succeeds.
    # Blackwell (sm_100+) cards work fine; pod_startup.sh auto-upgrades PyTorch.
    _GPU_FALLBACKS = [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A4500",
        "NVIDIA RTX A5000",
        "NVIDIA L40S",
        "NVIDIA GeForce RTX 4080 SUPER",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX 5000 Ada Generation",
        "NVIDIA RTX A6000",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA RTX PRO 4500 Blackwell",
        "NVIDIA GeForce RTX 5080",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
    ]
    gpu_candidates = [args.gpu] if args.gpu else _GPU_FALLBACKS

    # Bootstrap: always pull the branch-specific code BEFORE running pod_startup.sh.
    # This avoids the stale-cache problem where the network volume's cached
    # pod_startup.sh lags behind the branch being trained.
    startup_cmd = (
        "bash -c 'set -euo pipefail; "
        "BRANCH=${JEPA_BRANCH:-main}; "
        "REPO_DIR=/workspace/jepa-${BRANCH}; "
        "if [ -d $REPO_DIR/.git ]; then "
        "  rm -f $REPO_DIR/.git/index.lock $REPO_DIR/.git/config.lock; "
        "  git -C $REPO_DIR fetch origin && "
        "  git -C $REPO_DIR checkout -f ${BRANCH} && "
        "  git -C $REPO_DIR reset --hard origin/${BRANCH}; "
        "else "
        "  git clone --branch ${BRANCH} $GITHUB_REPO $REPO_DIR; "
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

    pod = None
    for gpu_type in gpu_candidates:
        try:
            pod = runpod.create_pod(
                name=pod_name,
                image_name=image,
                gpu_type_id=gpu_type,
                cloud_type=os.environ.get("RUNPOD_CLOUD_TYPE", "SECURE"),
                network_volume_id=volume_id,
                container_disk_in_gb=20,
                volume_in_gb=0,
                volume_mount_path="/workspace",
                docker_args=startup_cmd,
                env=env_vars,
                ports="22/tcp",
            )
            print(f"  GPU: {gpu_type} — available")
            break
        except Exception as e:
            print(f"  GPU: {gpu_type} — unavailable ({e})")
    if pod is None:
        sys.exit("No GPU instances available. Try again later or check RunPod dashboard.")

    pod_id = pod["id"]
    with open(_POD_ID_FILE, "w") as f:
        f.write(pod_id)
    print(f"Pod created: {pod_id}  ({pod_name})")
    print(f"Branch: {branch} | GPU: {gpu_type} | Image: {image}")
    print(f"Checkpoint: checkpoints/jepa-{branch}.pt")
    print("Startup log: /workspace/startup.log")

    # Spawn a background watcher that auto-downloads when the pod exits.
    script = os.path.abspath(__file__)
    subprocess.Popen(
        [sys.executable, script, "watch", pod_id],
        start_new_session=True,
        stdout=open(_WATCHER_LOG, "a"),
        stderr=subprocess.STDOUT,
    )
    print(f"Auto-download watcher started (log: {_WATCHER_LOG})")


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


def cmd_download(args):
    """Spin up a cheap CPU pod, rsync results+checkpoints locally, then terminate it."""
    load_env()
    runpod.api_key = get_required("RUNPOD_API_KEY")
    volume_id = get_required("RUNPOD_VOLUME_ID")
    ssh_key   = os.path.expanduser(os.environ.get("SSH_KEY", "~/.ssh/id_ed25519"))
    pubkey    = open(ssh_key + ".pub").read().strip()

    print("Spinning up CPU pod to access network volume...")
    # cpu3c-2-4: 2 vCPU, 4 GB RAM — cheapest option, only needs to run rsync
    for instance_id in ("cpu3c-2-4", "cpu5c-2-4", "cpu3c-4-8"):
        try:
            pod = runpod.create_pod(
                name="jepa-download",
                image_name="runpod/base:1.0.3-ubuntu2204",
                gpu_type_id=None,
                instance_id=instance_id,
                network_volume_id=volume_id,
                volume_mount_path="/workspace",
                container_disk_in_gb=5,
                ports="22/tcp",
                env={"PUBLIC_KEY": pubkey},
            )
            print(f"CPU pod created ({instance_id}): {pod['id']}")
            break
        except Exception as e:
            print(f"  {instance_id} unavailable: {e}")
    else:
        sys.exit("No CPU pod instances available. Try again in a moment.")

    pod_id = pod["id"]
    try:
        # Wait for SSH to become available
        print("Waiting for SSH...", end="", flush=True)
        ip, port = None, None
        for _ in range(60):
            time.sleep(5)
            pods = runpod.get_pods()
            for p in pods:
                if p["id"] != pod_id:
                    continue
                for pt in ((p.get("runtime") or {}).get("ports") or []):
                    if pt.get("isIpPublic") and pt["privatePort"] == 22:
                        ip, port = pt["ip"], pt["publicPort"]
            if ip:
                # Test if SSH is actually accepting connections
                r = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     "-i", ssh_key, f"root@{ip}", "-p", str(port), "echo ok"],
                    capture_output=True
                )
                if r.returncode == 0:
                    print(" ready.")
                    break
            print(".", end="", flush=True)
        else:
            sys.exit("Timed out waiting for SSH.")

        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-i", ssh_key, "-p", str(port)]

        # Rsync results and checkpoints
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        for remote, local in [
            ("/workspace/results/",     "./results/"),
            ("/workspace/checkpoints/", "./checkpoints/"),
            ("/workspace/startup.log",  "./startup.log"),
        ]:
            print(f"Rsyncing {remote} -> {local}")
            subprocess.run([
                "rsync", "-avz", "--progress",
                "-e", f"ssh {' '.join(ssh_opts)}",
                f"root@{ip}:{remote}", local,
            ], check=True)

        print("Download complete.")
    finally:
        print(f"Terminating CPU pod {pod_id}...")
        runpod.terminate_pod(pod_id)
        print("Done.")


def cmd_watch(args):
    """Background watcher: poll until pod exits then auto-download. Called automatically by create."""
    load_env()
    runpod.api_key = get_required("RUNPOD_API_KEY")
    pod_id = args.pod_id

    print(f"[watcher] monitoring pod {pod_id}", flush=True)
    while True:
        time.sleep(30)
        try:
            pods = runpod.get_pods()
        except Exception as e:
            print(f"[watcher] poll error: {e}", flush=True)
            continue
        for p in pods:
            if p["id"] == pod_id and p.get("desiredStatus") == "EXITED":
                print(f"[watcher] pod {pod_id} exited — starting download", flush=True)
                cmd_download(args)
                _notify("jepa-tetris training complete", "Results downloaded to ./checkpoints/ and ./results/")
                print("[watcher] done.", flush=True)
                return
        print("[watcher] still running...", flush=True)


def _notify(title, message):
    """Fire a macOS notification (silent no-op on other platforms)."""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}"'],
            check=True, capture_output=True,
        )
    except Exception:
        pass


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
    sub.add_parser("download")
    p_watch = sub.add_parser("watch")
    p_watch.add_argument("pod_id", help="Pod ID to monitor")
    args = parser.parse_args()
    dispatch = {"create": cmd_create, "stop": cmd_stop, "delete": cmd_delete,
                "status": cmd_status, "download": cmd_download, "watch": cmd_watch}
    if args.cmd not in dispatch:
        parser.print_help()
        sys.exit(1)
    dispatch[args.cmd](args)
