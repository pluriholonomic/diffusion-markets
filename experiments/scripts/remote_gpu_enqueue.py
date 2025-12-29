#!/usr/bin/env python3
"""
Safe queue writer for experiments/remote_queue.jsonl using an exclusive file lock.

Example:
  python3 scripts/remote_gpu_enqueue.py \
    --queue-file experiments/remote_queue.jsonl \
    --id my_task --gpu any \
    --cmd '.venv/bin/python -m forecastbench parity --d 24 --k 10 --n 50000 --run-name parity_k10'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _flock_exclusive(f, *, timeout_s: float = 10.0) -> None:
    import fcntl

    deadline = time.time() + float(timeout_s)
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.time() >= deadline:
                raise TimeoutError("Timed out waiting for exclusive queue lock")
            time.sleep(0.05)


def append_task(queue_path: Path, task: Dict[str, Any]) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(task, sort_keys=True) + "\n"
    with queue_path.open("a") as f:
        _flock_exclusive(f, timeout_s=10.0)
        f.write(line)
        f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-file", type=str, required=True)
    ap.add_argument("--id", type=str, required=True)
    ap.add_argument("--gpu", type=str, default="any", help='0|1|any (default: "any")')
    ap.add_argument("--cmd", type=str, required=True)
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--ready-if", type=str, default=None)
    args = ap.parse_args()

    task: Dict[str, Any] = {"id": str(args.id), "gpu": str(args.gpu), "cmd": str(args.cmd)}
    if args.log:
        task["log"] = str(args.log)
    if args.ready_if:
        task["ready_if"] = str(args.ready_if)

    append_task(Path(args.queue_file), task)
    print(f"enqueued id={task['id']} gpu={task['gpu']}")


if __name__ == "__main__":
    main()



