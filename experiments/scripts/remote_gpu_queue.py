#!/usr/bin/env python3
"""
Remote GPU queue worker.

Runs *locally* and dispatches jobs to a remote GPU box over SSH when GPUs are idle.

Queue format: JSONL (one JSON object per line). Example:

  {"id":"ar_eval_scalar_n512","gpu":"1","cmd":".venv/bin/python -m forecastbench pm_eval ..."}

Fields:
  - id (str, required): unique task id
  - gpu (str|int, required): "0"|"1"|"any" (or integer 0/1)
  - cmd (str, required): command to run on remote *inside* remote_workdir
  - log (str, optional): remote log path (default: remote_logs/<id>.log)
  - ready_if (str, optional): remote shell snippet; task is eligible only if it exits 0

The worker appends launched tasks to a local state JSONL so tasks are not launched twice.
To retry a task, delete its record from the state file.
"""

from __future__ import annotations

import argparse
import errno
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GPUStat:
    index: int
    util: int
    mem_used_mib: int
    mem_total_mib: int


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: List[str], *, timeout_s: int) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout_s))
    return int(p.returncode), str(p.stdout), str(p.stderr)


def ssh(host: str, port: int, remote_cmd: str, *, timeout_s: int = 30) -> Tuple[int, str, str]:
    return _run(["ssh", "-p", str(port), host, remote_cmd], timeout_s=timeout_s)


def _flock(f, lock_flag: int, *, timeout_s: float = 5.0) -> None:
    """
    Cross-process file lock using fcntl.flock (Unix).
    Uses a bounded wait so the worker can't hang forever on a stuck writer.
    """
    import fcntl

    deadline = time.time() + float(timeout_s)
    while True:
        try:
            fcntl.flock(f.fileno(), lock_flag | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.time() >= deadline:
                raise TimeoutError("Timed out waiting for file lock")
            time.sleep(0.05)
        except OSError as e:
            # Some filesystems may not support flock; treat as fatal (better than corrupt reads).
            raise RuntimeError(f"flock failed: {e}") from e


def _read_text_locked(path: Path, *, timeout_s: float = 5.0) -> str:
    import fcntl

    if not path.exists():
        return ""
    with path.open("r") as f:
        _flock(f, fcntl.LOCK_SH, timeout_s=timeout_s)
        return f.read()


def _append_line_locked(path: Path, line: str, *, timeout_s: float = 5.0) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        _flock(f, fcntl.LOCK_EX, timeout_s=timeout_s)
        f.write(line)
        f.flush()


def _parse_gpu_stats(out: str) -> List[GPUStat]:
    rows: List[GPUStat] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            util = int(float(parts[1]))
            mem_used = int(float(parts[2]))
            mem_total = int(float(parts[3]))
        except Exception:
            continue
        rows.append(GPUStat(index=idx, util=util, mem_used_mib=mem_used, mem_total_mib=mem_total))
    return rows


def get_gpu_stats(host: str, port: int) -> List[GPUStat]:
    code, out, err = ssh(
        host,
        port,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
        timeout_s=20,
    )
    if code != 0:
        raise RuntimeError(f"nvidia-smi failed (code={code}): {err.strip()}")
    stats = _parse_gpu_stats(out)
    if not stats:
        raise RuntimeError(f"Failed to parse nvidia-smi output: {out!r}")
    return stats


def _parse_index_uuid_map(out: str) -> Dict[str, int]:
    """
    Parse:
      index, uuid
    """
    m: Dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except Exception:
            continue
        uuid = parts[1]
        if uuid:
            m[str(uuid)] = int(idx)
    return m


def _parse_compute_apps(out: str) -> List[Tuple[str, int]]:
    """
    Parse:
      gpu_uuid, pid
    """
    rows: List[Tuple[str, int]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running processes"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        uuid = parts[0]
        try:
            pid = int(parts[1])
        except Exception:
            continue
        if uuid:
            rows.append((str(uuid), int(pid)))
    return rows


def get_gpu_proc_counts(host: str, port: int) -> Dict[int, int]:
    """
    Return a map {gpu_index: num_compute_processes}.
    """
    code1, out1, err1 = ssh(
        host,
        port,
        "nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits",
        timeout_s=20,
    )
    if code1 != 0:
        raise RuntimeError(f"nvidia-smi index/uuid failed (code={code1}): {err1.strip()}")
    uuid_to_idx = _parse_index_uuid_map(out1)

    code2, out2, _err2 = ssh(
        host,
        port,
        "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits || true",
        timeout_s=20,
    )
    if code2 != 0:
        # tolerate; treat as no processes
        return {int(i): 0 for i in uuid_to_idx.values()}
    apps = _parse_compute_apps(out2)

    counts: Dict[int, int] = {int(i): 0 for i in uuid_to_idx.values()}
    for uuid, _pid in apps:
        idx = uuid_to_idx.get(uuid)
        if idx is None:
            continue
        counts[int(idx)] = int(counts.get(int(idx), 0)) + 1
    return counts


def gpu_slots_free(
    *,
    stats: List[GPUStat],
    proc_counts: Dict[int, int],
    max_procs_per_gpu: int,
    max_util_for_launch: int,
    min_free_mem_mib: int,
) -> Dict[int, int]:
    """
    Compute how many additional processes we are willing to launch per GPU.
    """
    out: Dict[int, int] = {}
    for s in stats:
        idx = int(s.index)
        procs = int(proc_counts.get(idx, 0))
        free_mem = int(s.mem_total_mib) - int(s.mem_used_mib)
        slots = max(0, int(max_procs_per_gpu) - procs)
        if int(s.util) > int(max_util_for_launch):
            slots = 0
        if free_mem < int(min_free_mem_mib):
            slots = 0
        out[idx] = int(slots)
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    raw_text = _read_text_locked(path)
    if not raw_text:
        return []
    for raw in raw_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError:
            # Most common cause: another process appended an incomplete line.
            # Skip rather than crashing; the next poll will likely see the completed line.
            continue
    return tasks


def read_launched_ids(state_path: Path) -> set[str]:
    launched: set[str] = set()
    if not state_path.exists():
        return launched
    raw_text = _read_text_locked(state_path)
    if not raw_text:
        return launched
    for raw in raw_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            tid = rec.get("id")
            if isinstance(tid, str) and tid:
                launched.add(tid)
        except Exception:
            continue
    return launched


def append_state(state_path: Path, rec: Dict[str, Any]) -> None:
    _append_line_locked(state_path, json.dumps(rec, sort_keys=True) + "\n")


def append_failure(fail_path: Path, rec: Dict[str, Any]) -> None:
    """
    Convenience: keep a compact failures-only log for quick grepping.
    """
    _append_line_locked(fail_path, json.dumps(rec, sort_keys=True) + "\n")


def _task_gpu(task: Dict[str, Any]) -> str:
    g = task.get("gpu")
    if isinstance(g, int):
        return str(g)
    if isinstance(g, str):
        return g.strip()
    return ""


def _task_id(task: Dict[str, Any]) -> str:
    tid = task.get("id")
    if not isinstance(tid, str) or not tid.strip():
        raise ValueError(f"Task missing string id: {task!r}")
    return tid.strip()


def _task_cmd(task: Dict[str, Any]) -> str:
    cmd = task.get("cmd")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError(f"Task missing string cmd: {task!r}")
    return cmd.strip()


def _task_log(task: Dict[str, Any], *, default_id: str) -> str:
    log = task.get("log")
    if isinstance(log, str) and log.strip():
        return log.strip()
    return f"remote_logs/{default_id}.log"


def is_ready(task: Dict[str, Any], *, host: str, port: int, remote_workdir: str) -> bool:
    ready_if = task.get("ready_if")
    if not isinstance(ready_if, str) or not ready_if.strip():
        return True

    snippet = ready_if.strip()
    remote = f"bash -lc {shlex.quote(f'cd {shlex.quote(remote_workdir)} && {snippet}')}"
    code, _out, _err = ssh(host, port, remote, timeout_s=20)
    return code == 0


def launch_task(
    task: Dict[str, Any],
    *,
    host: str,
    port: int,
    remote_workdir: str,
    gpu_index: int,
    dry_run: bool,
) -> Dict[str, Any]:
    tid = _task_id(task)
    cmd = _task_cmd(task)
    log_path = _task_log(task, default_id=tid)
    pid_path = f"{log_path}.pid"
    exit_path = f"{log_path}.exitcode"
    done_path = f"{log_path}.done"
    cmd_path = f"{log_path}.cmd.sh"
    runner_path = f"{log_path}.runner.sh"

    if dry_run:
        return {
            "id": tid,
            "gpu": gpu_index,
            "log": log_path,
            "pid": None,
            "exit_path": exit_path,
            "done_path": done_path,
            "cmd_path": cmd_path,
            "runner_path": runner_path,
            "dry_run": True,
        }

    # Robustly preserve cmd text without worrying about shell quoting inside cmd.
    # We write a command file and a small runner that records the exit code + done timestamp.
    remote_script = f"""
set -euo pipefail
cd {shlex.quote(remote_workdir)}
export PYTHONPATH=src
mkdir -p {shlex.quote(str(Path(log_path).parent))}

# write command file
cat > {shlex.quote(cmd_path)} <<'CMD_EOF'
{cmd}
CMD_EOF
chmod +x {shlex.quote(cmd_path)}

# write runner
cat > {shlex.quote(runner_path)} <<'RUN_EOF'
#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(remote_workdir)}
export PYTHONPATH=src
set +e
bash {shlex.quote(cmd_path)}
ec=$?
set -e
echo "$ec" > {shlex.quote(exit_path)}
date -u +%FT%TZ > {shlex.quote(done_path)}
exit "$ec"
RUN_EOF
chmod +x {shlex.quote(runner_path)}

{{ nohup env CUDA_VISIBLE_DEVICES={int(gpu_index)} bash {shlex.quote(runner_path)} > {shlex.quote(log_path)} 2>&1 & echo $! > {shlex.quote(pid_path)}; }}
echo "__LAUNCHED__ id={tid} gpu={int(gpu_index)} pid=$(cat {shlex.quote(pid_path)}) log={log_path}"
""".strip()

    remote = f"bash -lc {shlex.quote(remote_script)}"
    code, out, err = ssh(host, port, remote, timeout_s=60)
    if code != 0:
        raise RuntimeError(f"Launch failed for {tid} (code={code}): {err.strip()}")

    pid: Optional[int] = None
    for line in out.splitlines():
        if "__LAUNCHED__" in line and "pid=" in line:
            try:
                # naive parse: pid=<int>
                for tok in line.split():
                    if tok.startswith("pid="):
                        pid = int(tok.split("=", 1)[1])
            except Exception:
                pid = None
    return {
        "id": tid,
        "gpu": int(gpu_index),
        "log": log_path,
        "pid": pid,
        "exit_path": exit_path,
        "done_path": done_path,
        "cmd_path": cmd_path,
        "runner_path": runner_path,
        "stdout": out.strip(),
    }


def _load_state_records(state_path: Path) -> List[Dict[str, Any]]:
    raw_text = _read_text_locked(state_path)
    if not raw_text:
        return []
    recs = []
    for raw in raw_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            continue
    return recs


def _state_index(state_path: Path) -> Tuple[Dict[str, Dict[str, Any]], set[str]]:
    """
    Returns:
      launched: {id -> launched_record}
      done: set(ids) which have a terminal event
    """
    launched: Dict[str, Dict[str, Any]] = {}
    done: set[str] = set()
    for rec in _load_state_records(state_path):
        tid = rec.get("id")
        if not isinstance(tid, str) or not tid:
            continue
        ev = rec.get("event")
        if ev == "launched":
            launched[tid] = rec
        elif ev in {"finished", "failed", "finished_legacy", "failed_legacy", "unknown_exit"}:
            done.add(tid)
    return launched, done


def _ssh_read_file_tail(host: str, port: int, remote_workdir: str, path: str, *, n: int = 200) -> str:
    # Keep quoting simple: these log paths are expected to be simple relative paths like
    # remote_logs/<task>.log; using bash -lc + nested shlex.quote can become fragile.
    cmd = f"cd {shlex.quote(remote_workdir)} && tail -n {int(n)} {shlex.quote(path)} 2>/dev/null || true"
    code, out, err = ssh(host, port, f"bash -lc {shlex.quote(cmd)}", timeout_s=30)
    if code != 0 and not out:
        return f"[tail_error code={code}] {err.strip()}\n"
    return out


def _ssh_read_file(host: str, port: int, remote_workdir: str, path: str) -> Optional[str]:
    cmd = f"cd {shlex.quote(remote_workdir)} && test -f {shlex.quote(path)} && cat {shlex.quote(path)}"
    code, out, _err = ssh(host, port, f"bash -lc {shlex.quote(cmd)}", timeout_s=20)
    if code != 0:
        return None
    return out


def _ssh_pid_running(host: str, port: int, pid: int) -> bool:
    cmd = f"ps -p {int(pid)} >/dev/null 2>&1"
    code, _out, _err = ssh(host, port, cmd, timeout_s=20)
    return code == 0


def poll_and_log_completions(
    *,
    host: str,
    port: int,
    remote_workdir: str,
    state_path: Path,
    failures_path: Path,
    tail_lines: int = 200,
) -> None:
    """
    For launched tasks with no terminal event yet, check remote exitcode/done markers and log failures.
    """
    launched, done = _state_index(state_path)
    for tid, rec in launched.items():
        if tid in done:
            continue
        log_path = rec.get("log")
        if not isinstance(log_path, str) or not log_path:
            continue
        exit_path = rec.get("exit_path")
        if not isinstance(exit_path, str) or not exit_path:
            exit_path = f"{log_path}.exitcode"
        done_path = rec.get("done_path")
        if not isinstance(done_path, str) or not done_path:
            done_path = f"{log_path}.done"
        pid = rec.get("pid")

        exit_txt = _ssh_read_file(host, port, str(remote_workdir), str(exit_path))
        if exit_txt is not None and exit_txt.strip():
            try:
                ec = int(exit_txt.strip().splitlines()[-1].strip())
            except Exception:
                ec = None
            done_txt = _ssh_read_file(host, port, str(remote_workdir), str(done_path))
            done_ts = done_txt.strip().splitlines()[-1].strip() if done_txt else None
            tail = _ssh_read_file_tail(host, port, str(remote_workdir), str(log_path), n=int(tail_lines))
            ev = "finished" if ec == 0 else "failed"
            out_rec = {
                "ts": _utc_ts(),
                "event": ev,
                "id": tid,
                "gpu": rec.get("gpu"),
                "pid": pid,
                "log": log_path,
                "exit_code": ec,
                "done_ts": done_ts,
                "tail": tail if ec != 0 else None,
            }
            append_state(state_path, out_rec)
            if ev == "failed":
                append_failure(failures_path, out_rec)
            continue

        # Legacy / unexpected: no exitcode marker yet. If PID is gone, infer from log tail.
        pid_i = None
        try:
            pid_i = int(pid) if pid is not None else None
        except Exception:
            pid_i = None

        if pid_i is not None and _ssh_pid_running(host, port, pid_i):
            continue

        # PID not running and no exitcode marker -> treat as legacy completion.
        tail = _ssh_read_file_tail(host, port, str(remote_workdir), str(log_path), n=int(tail_lines))
        inferred_ok = ("Artifacts:" in tail) or ("Artifacts :" in tail)
        ev = "finished_legacy" if inferred_ok else "failed_legacy"
        out_rec = {
            "ts": _utc_ts(),
            "event": ev,
            "id": tid,
            "gpu": rec.get("gpu"),
            "pid": pid_i,
            "log": log_path,
            "exit_code": None,
            "done_ts": None,
            "tail": tail,
        }
        append_state(state_path, out_rec)
        if ev != "finished_legacy":
            append_failure(failures_path, out_rec)


def pick_and_launch(
    tasks: List[Dict[str, Any]],
    *,
    launched_ids: set[str],
    slots_free: Dict[int, int],
    stats_by_idx: Dict[int, GPUStat],
    host: str,
    port: int,
    remote_workdir: str,
    state_path: Path,
    failures_path: Path,
    dry_run: bool,
) -> int:
    """
    Try to launch up to the available slot capacity across GPUs. Returns number of tasks launched.
    """
    launched_count = 0
    slots = dict(slots_free)

    # First pass: tasks pinned to a specific GPU.
    for task in tasks:
        tid = _task_id(task)
        if tid in launched_ids:
            continue
        g = _task_gpu(task)
        if g in {"any", ""}:
            continue
        try:
            gi = int(g)
        except Exception:
            continue
        if int(slots.get(gi, 0)) <= 0:
            continue
        if not is_ready(task, host=host, port=port, remote_workdir=remote_workdir):
            continue
        try:
            rec = launch_task(
                task,
                host=host,
                port=port,
                remote_workdir=remote_workdir,
                gpu_index=gi,
                dry_run=dry_run,
            )
            append_state(
                state_path,
                {"ts": _utc_ts(), "event": "launched", **rec},
            )
            launched_ids.add(tid)
            slots[gi] = int(slots.get(gi, 0)) - 1
            launched_count += 1
        except Exception as e:
            fail_rec = {"ts": _utc_ts(), "event": "launch_failed", "id": tid, "gpu": gi, "error": str(e)}
            append_state(state_path, fail_rec)
            append_failure(failures_path, fail_rec)
            continue

    # Second pass: gpu="any" tasks.
    for task in tasks:
        tid = _task_id(task)
        if tid in launched_ids:
            continue
        g = _task_gpu(task)
        if g not in {"any", ""}:
            continue
        free = [gi for gi, n in slots.items() if int(n) > 0]
        if not free:
            break
        if not is_ready(task, host=host, port=port, remote_workdir=remote_workdir):
            continue
        # Prefer the GPU with the most free memory to reduce OOM risk.
        def _free_mem(i: int) -> int:
            s = stats_by_idx.get(int(i))
            if s is None:
                return -1
            return int(s.mem_total_mib) - int(s.mem_used_mib)

        gi = int(max(free, key=_free_mem))
        try:
            rec = launch_task(
                task,
                host=host,
                port=port,
                remote_workdir=remote_workdir,
                gpu_index=gi,
                dry_run=dry_run,
            )
            append_state(
                state_path,
                {"ts": _utc_ts(), "event": "launched", **rec},
            )
            launched_ids.add(tid)
            slots[gi] = int(slots.get(gi, 0)) - 1
            launched_count += 1
        except Exception as e:
            fail_rec = {"ts": _utc_ts(), "event": "launch_failed", "id": tid, "gpu": gi, "error": str(e)}
            append_state(state_path, fail_rec)
            append_failure(failures_path, fail_rec)
            continue

    return launched_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="root@95.133.252.72")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--remote-workdir", type=str, default="/root/diffusion-markets/experiments")
    ap.add_argument("--queue-file", type=str, required=True)
    ap.add_argument("--state-file", type=str, required=True)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument(
        "--max-procs-per-gpu",
        type=int,
        default=1,
        help="Allow up to this many concurrent processes per GPU (set to 2 to overfill underutilized GPUs).",
    )
    ap.add_argument(
        "--max-util-for-launch",
        type=int,
        default=85,
        help="Only launch additional tasks onto a GPU if its current utilization is <= this threshold.",
    )
    ap.add_argument(
        "--min-free-mem-mib",
        type=int,
        default=20000,
        help="Only launch additional tasks onto a GPU if it has at least this much free VRAM.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Ensure logs show up promptly under nohup (stdout is block-buffered when redirected).
    try:
        import sys

        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    queue_path = Path(args.queue_file)
    state_path = Path(args.state_file)
    failures_path = state_path.parent / "remote_gpu_queue_failures.jsonl"

    print(f"[{_utc_ts()}] remote_gpu_queue start host={args.host} port={args.port} poll_s={args.poll_s}")
    print(f"[{_utc_ts()}] queue_file={queue_path} state_file={state_path} remote_workdir={args.remote_workdir}")
    print(
        f"[{_utc_ts()}] launch policy: max_procs_per_gpu={int(args.max_procs_per_gpu)} "
        f"max_util_for_launch={int(args.max_util_for_launch)} min_free_mem_mib={int(args.min_free_mem_mib)} "
        f"dry_run={bool(args.dry_run)}"
    )

    while True:
        try:
            # 0) First, record any tasks that have already completed/failed since last poll.
            # This gives us a "failed tasks" audit trail with log tails (stderr/stdout).
            poll_and_log_completions(
                host=str(args.host),
                port=int(args.port),
                remote_workdir=str(args.remote_workdir),
                state_path=state_path,
                failures_path=failures_path,
                tail_lines=200,
            )

            launched_ids = read_launched_ids(state_path)
            tasks = read_jsonl(queue_path)
            # Validate tasks early to surface errors in logs.
            valid_tasks = []
            for t in tasks:
                _task_id(t)
                _task_cmd(t)
                if _task_gpu(t) not in {"", "any", "0", "1"}:
                    # allow "0"/"1"/"any"; ignore unknown.
                    raise ValueError(f"Task {t.get('id')!r} has invalid gpu={t.get('gpu')!r}")
                valid_tasks.append(t)
            tasks = valid_tasks

            if not tasks:
                print(f"[{_utc_ts()}] queue empty; sleeping")
                time.sleep(int(args.poll_s))
                continue

            stats = get_gpu_stats(args.host, int(args.port))
            stats_by_idx = {int(s.index): s for s in stats}
            proc_counts = get_gpu_proc_counts(args.host, int(args.port))
            slots_free = gpu_slots_free(
                stats=stats,
                proc_counts=proc_counts,
                max_procs_per_gpu=int(args.max_procs_per_gpu),
                max_util_for_launch=int(args.max_util_for_launch),
                min_free_mem_mib=int(args.min_free_mem_mib),
            )

            # Pretty status line
            parts = []
            for idx in sorted(stats_by_idx.keys()):
                s = stats_by_idx[idx]
                procs = int(proc_counts.get(idx, 0))
                free_mem = int(s.mem_total_mib) - int(s.mem_used_mib)
                slots = int(slots_free.get(idx, 0))
                parts.append(
                    f"gpu{idx}:util={s.util}% mem={s.mem_used_mib}/{s.mem_total_mib}MiB free={free_mem}MiB procs={procs} slots={slots}"
                )
            any_slots = any(int(v) > 0 for v in slots_free.values())
            print(f"[{_utc_ts()}] status: " + " | ".join(parts))

            if not any_slots:
                time.sleep(int(args.poll_s))
                continue

            n = pick_and_launch(
                tasks,
                launched_ids=launched_ids,
                slots_free=slots_free,
                stats_by_idx=stats_by_idx,
                host=str(args.host),
                port=int(args.port),
                remote_workdir=str(args.remote_workdir),
                state_path=state_path,
                failures_path=failures_path,
                dry_run=bool(args.dry_run),
            )
            if n > 0:
                print(f"[{_utc_ts()}] launched {n} task(s)")
            elif any_slots:
                # Helpful debug: this usually means the remaining tasks are gated by ready_if
                # (or pinned to GPUs with 0 slots), so the worker will spin "idle" with free GPUs.
                try:
                    candidates = []
                    for t in tasks:
                        tid = _task_id(t)
                        if tid in launched_ids:
                            continue
                        candidates.append(t)
                    gated = [
                        t
                        for t in candidates
                        if isinstance(t.get("ready_if"), str) and str(t.get("ready_if")).strip()
                    ]
                    nongated = len(candidates) - len(gated)
                    print(
                        f"[{_utc_ts()}] no task launched this poll "
                        f"(candidates={len(candidates)} gated_by_ready_if={len(gated)} ungated={nongated})"
                    )
                    for t in gated[:5]:
                        rid = str(t.get("id"))
                        snippet = str(t.get("ready_if", "")).strip().replace("\n", " ")[:180]
                        print(f"[{_utc_ts()}] blocked: id={rid} ready_if='{snippet}...'")
                except Exception:
                    # Never crash the worker on debug printing.
                    pass
            time.sleep(int(args.poll_s))
        except KeyboardInterrupt:
            print(f"[{_utc_ts()}] interrupted; exiting")
            return
        except Exception as e:
            print(f"[{_utc_ts()}] ERROR: {e}")
            time.sleep(int(args.poll_s))


if __name__ == "__main__":
    main()


