#!/usr/bin/env python3
"""
GPU Queue Monitor - checks if GPUs are idle and queue is stuck, attempts recovery.

Run via cron every 5 minutes:
    */5 * * * * cd /Users/tarunchitra/repos/diffusion-markets/experiments && python3 scripts/gpu_queue_monitor.py >> logs/gpu_queue_monitor.log 2>&1
"""

import subprocess
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Configuration
REMOTE_HOST = "root@95.133.252.72"
REMOTE_PORT = "22"
REMOTE_WORKDIR = "/root/diffusion-markets/experiments"
LOCAL_WORKDIR = Path(__file__).parent.parent
QUEUE_FILE = LOCAL_WORKDIR / "remote_queue.jsonl"
STATE_FILE = LOCAL_WORKDIR / "remote_queue_state.json"
QUEUE_LOG = LOCAL_WORKDIR / "remote_logs" / "local_remote_gpu_queue.log"
QUEUE_PID_FILE = LOCAL_WORKDIR / "remote_logs" / "gpu_queue_monitor.pid"

# Thresholds
IDLE_THRESHOLD_MINUTES = 5  # Consider stuck if idle for this long
MAX_QUEUE_LOG_AGE_MINUTES = 3  # Queue log should update every ~30s


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}")


def run_ssh(cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Run command on remote host."""
    full_cmd = ["ssh", "-p", REMOTE_PORT, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", REMOTE_HOST, cmd]
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "SSH timeout"
    except Exception as e:
        return -1, str(e)


def get_gpu_status() -> dict:
    """Get GPU utilization from remote."""
    code, output = run_ssh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
    if code != 0:
        return {"error": output}
    
    gpus = []
    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({
                "index": int(parts[0]),
                "util": int(parts[1]),
                "mem_used": int(parts[2]),
                "mem_total": int(parts[3]),
            })
    return {"gpus": gpus}


def get_running_jobs() -> list[str]:
    """Get list of running forecastbench jobs."""
    code, output = run_ssh("ps aux | grep -E '\\.venv/bin/python.*forecastbench' | grep -v grep | grep -oE 'run-name [^ ]+'")
    if code != 0:
        return []
    return [line.replace("run-name ", "") for line in output.strip().split("\n") if line]


def check_queue_worker() -> dict:
    """Check if queue worker is running locally."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "remote_gpu_queue.py"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.strip().split("\n") if p]
        return {"running": len(pids) > 0, "pids": pids, "count": len(pids)}
    except Exception as e:
        return {"running": False, "error": str(e)}


def check_queue_log_freshness() -> dict:
    """Check if queue log is being updated."""
    if not QUEUE_LOG.exists():
        return {"fresh": False, "error": "Log file not found"}
    
    mtime = QUEUE_LOG.stat().st_mtime
    age_seconds = time.time() - mtime
    age_minutes = age_seconds / 60
    
    return {
        "fresh": age_minutes < MAX_QUEUE_LOG_AGE_MINUTES,
        "age_minutes": round(age_minutes, 1),
        "last_modified": datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
    }


def count_pending_jobs() -> dict:
    """Count jobs in queue that haven't been processed."""
    if not QUEUE_FILE.exists():
        return {"error": "Queue file not found"}
    if not STATE_FILE.exists():
        return {"error": "State file not found"}
    
    # Get all job IDs from queue
    queue_ids = set()
    with open(QUEUE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                job = json.loads(line)
                if "id" in job:
                    queue_ids.add(job["id"])
            except json.JSONDecodeError:
                pass
    
    # Get all job IDs from state
    state_ids = set()
    with open(STATE_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if "id" in entry:
                    state_ids.add(entry["id"])
            except json.JSONDecodeError:
                pass
    
    pending = queue_ids - state_ids
    return {
        "total_in_queue": len(queue_ids),
        "processed": len(state_ids),
        "pending": len(pending),
        "pending_ids": list(pending)[:10],  # First 10
    }


def restart_queue_worker():
    """Kill existing queue workers and start a fresh one."""
    log("Attempting to restart queue worker...")
    
    # Kill existing workers
    subprocess.run(["pkill", "-f", "remote_gpu_queue.py"], capture_output=True)
    time.sleep(2)
    
    # Start new worker
    cmd = [
        "nohup", "python3", "scripts/remote_gpu_queue.py",
        "--host", REMOTE_HOST,
        "--port", REMOTE_PORT,
        "--remote-workdir", REMOTE_WORKDIR,
        "--queue-file", str(QUEUE_FILE),
        "--state-file", str(STATE_FILE),
        "--poll-s", "30",
        "--max-procs-per-gpu", "2",
    ]
    
    log_file = open(QUEUE_LOG, "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        cwd=str(LOCAL_WORKDIR),
        start_new_session=True,
    )
    
    log(f"Started new queue worker with PID {proc.pid}")
    return proc.pid


def diagnose_and_fix():
    """Main monitoring logic."""
    log("=" * 60)
    log("GPU Queue Monitor Check")
    
    # 1. Check GPU status
    gpu_status = get_gpu_status()
    if "error" in gpu_status:
        log(f"ERROR getting GPU status: {gpu_status['error']}")
        return
    
    gpus = gpu_status["gpus"]
    all_idle = all(g["util"] == 0 and g["mem_used"] == 0 for g in gpus)
    gpu_strs = [f"GPU{g['index']}: {g['util']}% util, {g['mem_used']}MB" for g in gpus]
    log(f"GPU Status: {', '.join(gpu_strs)}")
    
    # 2. Check running jobs
    running_jobs = get_running_jobs()
    log(f"Running jobs: {len(running_jobs)} - {running_jobs[:5]}")
    
    # 3. Check queue worker
    queue_worker = check_queue_worker()
    log(f"Queue worker: {queue_worker}")
    
    # 4. Check queue log freshness
    log_freshness = check_queue_log_freshness()
    log(f"Queue log: {log_freshness}")
    
    # 5. Count pending jobs
    pending = count_pending_jobs()
    log(f"Pending jobs: {pending.get('pending', 'unknown')} of {pending.get('total_in_queue', 'unknown')}")
    
    # Diagnose issues
    issues = []
    
    if all_idle and len(running_jobs) == 0:
        issues.append("All GPUs idle with no running jobs")
    
    if not queue_worker.get("running"):
        issues.append("Queue worker not running")
    elif queue_worker.get("count", 0) > 1:
        issues.append(f"Multiple queue workers running: {queue_worker['pids']}")
    
    if not log_freshness.get("fresh"):
        issues.append(f"Queue log stale (age: {log_freshness.get('age_minutes', '?')} min)")
    
    if pending.get("pending", 0) > 0 and all_idle:
        issues.append(f"Pending jobs ({pending['pending']}) but GPUs idle")
    
    if issues:
        log(f"ISSUES DETECTED: {issues}")
        
        # Attempt recovery
        if not queue_worker.get("running") or queue_worker.get("count", 0) > 1:
            restart_queue_worker()
        elif not log_freshness.get("fresh") and pending.get("pending", 0) > 0:
            log("Queue log stale with pending jobs - restarting worker")
            restart_queue_worker()
        elif all_idle and pending.get("pending", 0) > 0:
            log("GPUs idle with pending jobs - checking if queue needs restart")
            # Check last few lines of queue log for "no task launched"
            if QUEUE_LOG.exists():
                with open(QUEUE_LOG) as f:
                    last_lines = f.readlines()[-10:]
                no_launch_count = sum(1 for l in last_lines if "no task launched" in l)
                if no_launch_count >= 3:
                    log(f"Queue reporting 'no task launched' {no_launch_count} times - restarting")
                    restart_queue_worker()
    else:
        log("All systems nominal")
    
    log("Check complete")


if __name__ == "__main__":
    os.chdir(LOCAL_WORKDIR)
    diagnose_and_fix()

