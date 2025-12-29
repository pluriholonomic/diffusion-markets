#!/usr/bin/env python3
"""
Background GPU utilization monitor.
Checks GPU usage and adds jobs when slots are available.
"""

import subprocess
import time
import json
import os
from datetime import datetime
from pathlib import Path

# Configuration
SSH_HOST = "root@95.133.252.72"
SSH_PORT = "22"
POLL_INTERVAL = 120  # seconds
QUEUE_FILE = Path(__file__).parent.parent / "remote_queue.jsonl"
LOG_FILE = Path(__file__).parent.parent / "remote_logs" / "gpu_monitor.log"

# Job templates for when we need to add more work
GPU0_JOB_TEMPLATES = [
    {
        "id": "pm_hybrid_train_seed{seed}",
        "cmd": """.venv/bin/python -m forecastbench pm_hybrid_train \
  --dataset-path /root/polymarket_data/derived/gamma_yesno_ready.parquet \
  --run-name pm_hybrid_train_seed{seed} \
  --max-rows 1500 \
  --text-cols question,description \
  --train-frac 0.8 \
  --seed {seed} \
  --ar-model Qwen/Qwen3-14B \
  --ar-K 5 \
  --ar-max-new-tokens 256 \
  --ar-temperature 0.7 \
  --ar-device cuda \
  --ar-device-map auto \
  --embed-model Qwen/Qwen3-14B \
  --embed-device cuda \
  --embed-dtype bfloat16 \
  --embed-device-map auto \
  --embed-batch-size 4 \
  --embed-max-length 512 \
  --diff-hidden-dim 256 \
  --diff-depth 4 \
  --diff-T 50 \
  --diff-train-steps 2000 \
  --diff-batch-size 256 \
  --diff-lr 1e-4 \
  --diff-samples 16 \
  --bins 20""",
    },
    {
        "id": "pm_difftrain_large_seed{seed}",
        "cmd": """.venv/bin/python -m forecastbench pm_difftrain \
  --dataset-path /root/polymarket_data/derived/gamma_yesno_ready.parquet \
  --run-name pm_difftrain_large_seed{seed} \
  --max-rows 5000 \
  --text-cols question,description \
  --pred-col pred_prob \
  --train-frac 0.8 \
  --seed {seed} \
  --bins 20 \
  --embed-model Qwen/Qwen3-14B \
  --embed-device cuda \
  --embed-dtype bfloat16 \
  --embed-device-map auto \
  --embed-batch-size 4 \
  --embed-max-length 512 \
  --device cuda \
  --train-steps 10000 \
  --batch-size 256 \
  --lr 1e-4 \
  --T 100 \
  --hidden-dim 512 \
  --depth 6 \
  --sample-steps 50 \
  --mc 32 \
  --agg mean \
  --log-every 500""",
    },
]

GPU1_JOB_TEMPLATES = [
    {
        "id": "pm_hybrid_train_gpu1_seed{seed}",
        "cmd": """.venv/bin/python -m forecastbench pm_hybrid_train \
  --dataset-path /root/polymarket_data/derived/gamma_yesno_ready.parquet \
  --run-name pm_hybrid_train_gpu1_seed{seed} \
  --max-rows 2000 \
  --text-cols question,description \
  --train-frac 0.8 \
  --seed {seed} \
  --ar-model Qwen/Qwen3-14B \
  --ar-K 3 \
  --ar-max-new-tokens 128 \
  --ar-temperature 0.7 \
  --ar-device cuda \
  --ar-device-map auto \
  --embed-model Qwen/Qwen3-14B \
  --embed-device cuda \
  --embed-dtype bfloat16 \
  --embed-device-map auto \
  --embed-batch-size 8 \
  --embed-max-length 256 \
  --diff-hidden-dim 512 \
  --diff-depth 6 \
  --diff-T 100 \
  --diff-train-steps 5000 \
  --diff-batch-size 512 \
  --diff-lr 5e-5 \
  --diff-samples 32 \
  --bins 20""",
    },
    {
        "id": "pm_difftrain_gpu1_seed{seed}",
        "cmd": """.venv/bin/python -m forecastbench pm_difftrain \
  --dataset-path /root/polymarket_data/derived/gamma_yesno_ready.parquet \
  --run-name pm_difftrain_gpu1_seed{seed} \
  --max-rows 3000 \
  --text-cols question,description \
  --pred-col pred_prob \
  --train-frac 0.8 \
  --seed {seed} \
  --bins 20 \
  --embed-model Qwen/Qwen3-14B \
  --embed-device cuda \
  --embed-dtype bfloat16 \
  --embed-device-map auto \
  --embed-batch-size 4 \
  --device cuda \
  --train-steps 8000 \
  --batch-size 256 \
  --lr 1e-4 \
  --T 100 \
  --hidden-dim 512 \
  --depth 6 \
  --sample-steps 50 \
  --mc 32 \
  --agg mean \
  --log-every 200""",
    },
]


def log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_gpu_status():
    """Get GPU utilization and process count from remote."""
    try:
        result = subprocess.run(
            ["ssh", "-p", SSH_PORT, "-o", "ConnectTimeout=10", SSH_HOST,
             "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "util": int(parts[1]),
                    "mem_used": int(parts[2]),
                    "mem_free": int(parts[3]),
                })
        return gpus
    except Exception as e:
        log(f"ERROR getting GPU status: {e}")
        return None


def get_gpu_procs():
    """Get number of processes per GPU."""
    try:
        result = subprocess.run(
            ["ssh", "-p", SSH_PORT, "-o", "ConnectTimeout=10", SSH_HOST,
             "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return {0: 0, 1: 0}
        
        # Count processes per GPU
        procs = {0: 0, 1: 0}
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                procs[0 if "GPU-0" in line or line.startswith("GPU 0") else 1] += 1
        
        # Simpler: just count lines and assume half per GPU if we can't parse
        total = len([l for l in result.stdout.strip().split("\n") if l.strip()])
        if sum(procs.values()) == 0 and total > 0:
            # Fallback: check which GPU has processes
            result2 = subprocess.run(
                ["ssh", "-p", SSH_PORT, "-o", "ConnectTimeout=10", SSH_HOST,
                 "nvidia-smi"],
                capture_output=True, text=True, timeout=30
            )
            # Just return rough estimate
            procs = {0: total // 2, 1: total - total // 2}
        
        return procs
    except Exception as e:
        log(f"ERROR getting GPU procs: {e}")
        return {0: 0, 1: 0}


def get_pending_jobs():
    """Count pending jobs in queue file."""
    try:
        with open(QUEUE_FILE, "r") as f:
            lines = f.readlines()
        
        pending = {"0": 0, "1": 0}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                job = json.loads(line)
                if not job.get("done"):
                    gpu = str(job.get("gpu", "0"))
                    pending[gpu] = pending.get(gpu, 0) + 1
            except:
                pass
        return pending
    except Exception as e:
        log(f"ERROR reading queue: {e}")
        return {"0": 0, "1": 0}


def enqueue_job(job_id: str, cmd: str, gpu: int):
    """Add a job to the queue."""
    try:
        result = subprocess.run(
            ["python3", str(Path(__file__).parent / "remote_gpu_enqueue.py"),
             "--queue-file", str(QUEUE_FILE),
             "--id", job_id,
             "--gpu", str(gpu),
             "--cmd", cmd],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            log(f"ENQUEUED: {job_id} -> GPU{gpu}")
            return True
        else:
            log(f"ENQUEUE FAILED: {job_id}: {result.stderr}")
            return False
    except Exception as e:
        log(f"ERROR enqueuing {job_id}: {e}")
        return False


def main():
    log("GPU utilization monitor started")
    log(f"Poll interval: {POLL_INTERVAL}s")
    
    seed_counter = 100  # Start with seed 100 to avoid conflicts
    job_template_idx = {0: 0, 1: 0}
    
    while True:
        try:
            gpus = get_gpu_status()
            if gpus is None:
                log("Could not get GPU status, retrying...")
                time.sleep(30)
                continue
            
            pending = get_pending_jobs()
            
            for gpu in gpus:
                idx = gpu["index"]
                util = gpu["util"]
                mem_free = gpu["mem_free"]
                pending_count = pending.get(str(idx), 0)
                
                # Consider underutilized if:
                # - Utilization < 50% AND
                # - At least 50GB free AND
                # - Fewer than 3 pending jobs for this GPU
                underutilized = util < 50 and mem_free > 50000 and pending_count < 3
                
                status = f"GPU{idx}: util={util}% mem_free={mem_free}MB pending={pending_count}"
                if underutilized:
                    status += " [UNDERUTILIZED - adding job]"
                log(status)
                
                if underutilized:
                    # Add a new job
                    templates = GPU0_JOB_TEMPLATES if idx == 0 else GPU1_JOB_TEMPLATES
                    template = templates[job_template_idx[idx] % len(templates)]
                    job_template_idx[idx] += 1
                    
                    job_id = template["id"].format(seed=seed_counter)
                    cmd = template["cmd"].format(seed=seed_counter)
                    seed_counter += 1
                    
                    enqueue_job(job_id, cmd, idx)
            
            log(f"--- Sleeping {POLL_INTERVAL}s ---")
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            log("Monitor stopped by user")
            break
        except Exception as e:
            log(f"ERROR in main loop: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()


