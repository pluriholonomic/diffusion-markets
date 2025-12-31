#!/usr/bin/env python3
"""
Monitor remote experiments and update EXPERIMENTS.md with results.

Run in background:
    nohup python3 scripts/monitor_and_update_experiments.py &

Or with specific interval:
    python3 scripts/monitor_and_update_experiments.py --interval 3600  # 1 hour
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REMOTE_HOST = "root@95.133.252.72"
REMOTE_BASE = "/root/diffusion-markets/experiments"
LOCAL_BASE = Path(__file__).parent.parent
EXPERIMENTS_MD = LOCAL_BASE / "EXPERIMENTS.md"


def ssh_cmd(cmd: str, timeout: int = 30) -> Tuple[int, str]:
    """Run command on remote via SSH."""
    try:
        result = subprocess.run(
            ["ssh", REMOTE_HOST, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "SSH timeout"
    except Exception as e:
        return -1, str(e)


def get_queue_status() -> Dict[str, Any]:
    """Get current queue status from remote."""
    code, output = ssh_cmd(
        f"cat {REMOTE_BASE}/remote_logs/remote_gpu_queue_state.jsonl 2>/dev/null | tail -5"
    )
    
    if code != 0:
        return {"error": output}
    
    try:
        lines = [l for l in output.strip().split("\n") if l.strip()]
        if lines:
            return json.loads(lines[-1])
    except Exception as e:
        pass
    
    return {"status": "unknown"}


def get_running_experiments() -> List[Dict[str, Any]]:
    """Get list of running experiments."""
    code, output = ssh_cmd(
        f"ps aux | grep -E 'forecastbench|grpo_train|difftrain' | grep -v grep"
    )
    
    running = []
    for line in output.strip().split("\n"):
        if line.strip():
            parts = line.split()
            if len(parts) > 10:
                running.append({
                    "pid": parts[1],
                    "cmd": " ".join(parts[10:])[:100],
                })
    
    return running


def get_completed_runs() -> List[Dict[str, Any]]:
    """Get list of completed run directories with metrics."""
    code, output = ssh_cmd(
        f"ls -1td {REMOTE_BASE}/runs/*/ 2>/dev/null | head -20"
    )
    
    if code != 0:
        return []
    
    runs = []
    for run_dir in output.strip().split("\n"):
        if not run_dir.strip():
            continue
        
        run_name = Path(run_dir).name
        
        # Check for metrics.json
        code2, metrics_out = ssh_cmd(
            f"cat {run_dir}/metrics.json 2>/dev/null | head -100"
        )
        
        metrics = {}
        if code2 == 0 and metrics_out.strip():
            try:
                metrics = json.loads(metrics_out)
            except Exception:
                pass
        
        runs.append({
            "name": run_name,
            "path": run_dir,
            "has_metrics": bool(metrics),
            "metrics": metrics,
        })
    
    return runs


def get_headline_fetch_status() -> Dict[str, Any]:
    """Check status of Exa headline fetch."""
    code, output = ssh_cmd(
        f"tail -20 {REMOTE_BASE}/logs/fetch_exa_headlines.log 2>/dev/null"
    )
    
    status = {"running": False, "log_tail": output if code == 0 else ""}
    
    # Check if process is still running
    code2, ps_out = ssh_cmd("pgrep -f fetch_exa_headlines")
    status["running"] = code2 == 0 and ps_out.strip() != ""
    
    # Check for completion
    if "Done!" in output or "Enriched data saved" in output:
        status["completed"] = True
    else:
        status["completed"] = False
    
    # Parse stats if available
    match = re.search(r"Coverage:\s*([\d.]+%)", output)
    if match:
        status["coverage"] = match.group(1)
    
    return status


def get_gpu_status() -> Dict[str, Any]:
    """Get GPU utilization."""
    code, output = ssh_cmd("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
    
    if code != 0:
        return {"error": output}
    
    gpus = []
    for i, line in enumerate(output.strip().split("\n")):
        if line.strip():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "id": i,
                    "utilization": f"{parts[0]}%",
                    "memory": f"{parts[1]}/{parts[2]} MB",
                })
    
    return {"gpus": gpus}


def format_metrics_table(runs: List[Dict[str, Any]]) -> str:
    """Format completed runs as a markdown table."""
    if not runs:
        return "| (No completed runs with metrics) |\n"
    
    lines = []
    lines.append("| Run | Brier | ECE | ROI | Status |")
    lines.append("|-----|-------|-----|-----|--------|")
    
    for run in runs[:10]:  # Last 10 runs
        name = run["name"][:40]
        metrics = run.get("metrics", {})
        
        brier = metrics.get("brier", metrics.get("brier_score", "—"))
        if isinstance(brier, float):
            brier = f"{brier:.4f}"
        
        ece = metrics.get("ece", metrics.get("expected_calibration_error", "—"))
        if isinstance(ece, float):
            ece = f"{ece:.4f}"
        
        roi = metrics.get("roi", metrics.get("trading_roi", "—"))
        if isinstance(roi, float):
            roi = f"{roi:.1%}"
        
        status = "✅" if run["has_metrics"] else "⏳"
        
        lines.append(f"| {name} | {brier} | {ece} | {roi} | {status} |")
    
    return "\n".join(lines)


def update_experiments_md(
    queue_status: Dict[str, Any],
    running: List[Dict[str, Any]],
    completed: List[Dict[str, Any]],
    headlines: Dict[str, Any],
    gpu: Dict[str, Any],
) -> None:
    """Update EXPERIMENTS.md with current status."""
    
    if not EXPERIMENTS_MD.exists():
        print(f"[monitor] {EXPERIMENTS_MD} not found")
        return
    
    content = EXPERIMENTS_MD.read_text()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Update timestamp
    content = re.sub(
        r"\*This document is auto-updated\. Last modified:.*\*",
        f"*This document is auto-updated. Last modified: {now}*",
        content,
    )
    
    # Update queue status in header
    if "error" not in queue_status:
        current_task = queue_status.get("current", {}).get("id", "idle")
        content = re.sub(
            r"> \*\*Queue status\*\*:.*",
            f"> **Queue status**: `{current_task}` on remote H200",
            content,
        )
    
    # Build status update section
    status_section = f"""
### Auto-Updated Status ({now})

**Headlines Fetch**: {"🔄 Running" if headlines.get("running") else ("✅ Complete" if headlines.get("completed") else "⏳ Pending")}
{f'- Coverage: {headlines["coverage"]}' if headlines.get("coverage") else ""}

**GPU Status**:
"""
    
    for gpu_info in gpu.get("gpus", []):
        status_section += f"- GPU {gpu_info['id']}: {gpu_info['utilization']} util, {gpu_info['memory']}\n"
    
    if running:
        status_section += "\n**Running Experiments**:\n"
        for exp in running[:5]:
            status_section += f"- PID {exp['pid']}: `{exp['cmd'][:60]}...`\n"
    
    # Update or insert the auto-status section
    marker_start = "### Auto-Updated Status"
    marker_end = "---\n\n## "
    
    if marker_start in content:
        # Replace existing section
        pattern = r"### Auto-Updated Status.*?(?=---\n\n## )"
        content = re.sub(pattern, status_section.strip() + "\n\n", content, flags=re.DOTALL)
    else:
        # Insert before "## 6. Monitoring"
        insert_point = content.find("## 6. Monitoring")
        if insert_point > 0:
            content = content[:insert_point] + status_section + "\n---\n\n" + content[insert_point:]
    
    # Update results table if we have completed runs with metrics
    runs_with_metrics = [r for r in completed if r.get("has_metrics")]
    if runs_with_metrics:
        new_table = format_metrics_table(runs_with_metrics)
        
        # Find and update the preliminary numbers section
        prelim_pattern = r"### 5\.3 Preliminary Numbers.*?```\n┌.*?└.*?```"
        if re.search(prelim_pattern, content, re.DOTALL):
            # Keep the section but add a note about auto-updated results
            pass  # Don't overwrite manually curated data
    
    EXPERIMENTS_MD.write_text(content)
    print(f"[monitor] Updated {EXPERIMENTS_MD}")


def main_loop(interval_s: int = 3600, once: bool = False) -> None:
    """Main monitoring loop."""
    print(f"[monitor] Starting experiment monitor (interval: {interval_s}s)")
    
    while True:
        try:
            print(f"[monitor] Checking status at {datetime.now()}")
            
            queue_status = get_queue_status()
            running = get_running_experiments()
            completed = get_completed_runs()
            headlines = get_headline_fetch_status()
            gpu = get_gpu_status()
            
            print(f"[monitor] Queue: {queue_status.get('current', {}).get('id', 'unknown')}")
            print(f"[monitor] Running: {len(running)} experiments")
            print(f"[monitor] Completed: {len(completed)} runs")
            print(f"[monitor] Headlines: {'running' if headlines.get('running') else 'done/idle'}")
            
            update_experiments_md(queue_status, running, completed, headlines, gpu)
            
            if once:
                break
            
            print(f"[monitor] Sleeping {interval_s}s until next check...")
            time.sleep(interval_s)
            
        except KeyboardInterrupt:
            print("[monitor] Interrupted")
            break
        except Exception as e:
            print(f"[monitor] Error: {e}")
            if once:
                break
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=3600, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()
    
    main_loop(interval_s=args.interval, once=args.once)



