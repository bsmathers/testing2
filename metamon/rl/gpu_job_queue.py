"""Cross-process FIFO admission control for shared CUDA memory.

Jobs declare a conservative VRAM budget before starting CUDA work.  A request is
granted only when (1) the sum of active reservations fits below total VRAM minus
a safety margin and (2) nvidia-smi reports enough currently-free memory for the
new reservation.  Otherwise the job remains queued until memory is released.

The CLI form is intended for long-lived learner/collector subprocesses:

    python -m metamon.rl.gpu_job_queue run --kind learner --budget-mb 7200 -- cmd ...

The context-manager form is used for incremental jobs such as tournaments that
run inside the learner process.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DEFAULT_SAFETY_MB = int(os.environ.get("METAMON_GPU_QUEUE_SAFETY_MB", "1536"))
_DEFAULT_POLL_SECONDS = float(os.environ.get("METAMON_GPU_QUEUE_POLL_SECONDS", "2"))
_DEFAULT_LOG_SECONDS = float(os.environ.get("METAMON_GPU_QUEUE_LOG_SECONDS", "30"))


def _root() -> Path:
    base = os.environ.get(
        "METAMON_GPU_QUEUE_DIR",
        f"/tmp/metamon-gpu-queue-{os.getuid()}",
    )
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _paths(device: int) -> tuple[Path, Path]:
    root = _root()
    return root / f"cuda{device}.json", root / f"cuda{device}.lock"


def _proc_start(pid: int) -> Optional[str]:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except Exception:
        return None


def _alive(entry: dict) -> bool:
    pid = int(entry["pid"])
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    recorded = entry.get("proc_start")
    current = _proc_start(pid)
    return recorded is None or current is None or recorded == current


def _query_nvidia_smi(device: int) -> tuple[int, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()[0]
    total, free = [int(x.strip()) for x in out.split(",")[:2]]
    return total, free


def _load(path: Path) -> dict:
    if not path.exists():
        return {"active": [], "pending": []}
    try:
        state = json.loads(path.read_text())
    except Exception:
        state = {"active": [], "pending": []}
    state.setdefault("active", [])
    state.setdefault("pending", [])
    state["active"] = [e for e in state["active"] if _alive(e)]
    state["pending"] = [e for e in state["pending"] if _alive(e)]
    return state


def _save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, path)


@contextlib.contextmanager
def _locked(device: int):
    state_path, lock_path = _paths(device)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _load(state_path)
        try:
            yield state_path, state
        finally:
            _save(state_path, state)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@dataclass
class GPULease:
    kind: str
    budget_mb: int
    device: int = 0
    safety_mb: int = _DEFAULT_SAFETY_MB
    poll_seconds: float = _DEFAULT_POLL_SECONDS

    def __post_init__(self):
        self.budget_mb = int(self.budget_mb)
        self.device = int(self.device)
        self.safety_mb = int(self.safety_mb)
        self._id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._held = False

    def acquire(self) -> "GPULease":
        if self.budget_mb <= 0:
            self._held = True
            return self

        entry = {
            "id": self._id,
            "pid": os.getpid(),
            "proc_start": _proc_start(os.getpid()),
            "kind": self.kind,
            "budget_mb": self.budget_mb,
            "created": time.time(),
        }
        last_log = 0.0
        while True:
            total_mb, free_mb = _query_nvidia_smi(self.device)
            granted = False
            reason = ""
            with _locked(self.device) as (_path, state):
                ids = {e["id"] for e in state["active"] + state["pending"]}
                if self._id not in ids:
                    state["pending"].append(entry)

                state["pending"].sort(key=lambda e: (e["created"], e["id"]))
                active_reserved = sum(int(e["budget_mb"]) for e in state["active"])
                usable_mb = total_mb - self.safety_mb
                mine_is_head = bool(state["pending"]) and state["pending"][0]["id"] == self._id
                reservation_fits = active_reserved + self.budget_mb <= usable_mb
                physical_fits = free_mb >= self.budget_mb + self.safety_mb

                if mine_is_head and reservation_fits and physical_fits:
                    state["pending"] = [e for e in state["pending"] if e["id"] != self._id]
                    state["active"].append(entry)
                    granted = True
                else:
                    if not mine_is_head:
                        reason = "waiting behind earlier GPU job"
                    elif not reservation_fits:
                        reason = (
                            f"reserved {active_reserved} + request {self.budget_mb} > "
                            f"usable {usable_mb} MiB"
                        )
                    else:
                        reason = (
                            f"physical free {free_mb} < request+safety "
                            f"{self.budget_mb + self.safety_mb} MiB"
                        )

            if granted:
                self._held = True
                print(
                    f"[gpu-queue] admitted {self.kind}: {self.budget_mb} MiB "
                    f"on cuda:{self.device} (free={free_mb} MiB, total={total_mb} MiB)",
                    flush=True,
                )
                return self

            now = time.time()
            if now - last_log >= _DEFAULT_LOG_SECONDS:
                print(
                    f"[gpu-queue] queued {self.kind}: {self.budget_mb} MiB "
                    f"on cuda:{self.device}; {reason}",
                    flush=True,
                )
                last_log = now
            time.sleep(self.poll_seconds)

    def release(self) -> None:
        if not self._held:
            return
        if self.budget_mb > 0:
            with _locked(self.device) as (_path, state):
                state["active"] = [e for e in state["active"] if e["id"] != self._id]
                state["pending"] = [e for e in state["pending"] if e["id"] != self._id]
        self._held = False
        print(f"[gpu-queue] released {self.kind} on cuda:{self.device}", flush=True)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def gpu_job_lease(kind: str, budget_mb: int, device: int = 0) -> GPULease:
    return GPULease(kind=kind, budget_mb=budget_mb, device=device)


def _run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("gpu_job_queue run requires a command after --")

    with GPULease(
        kind=args.kind,
        budget_mb=args.budget_mb,
        device=args.device,
        safety_mb=args.safety_mb,
    ):
        child = subprocess.Popen(command)

        def forward(sig, _frame):
            if child.poll() is None:
                child.send_signal(sig)

        old_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers[sig] = signal.signal(sig, forward)
        try:
            return child.wait()
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            if child.poll() is None:
                child.terminate()
                child.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="FIFO CUDA-memory admission queue")
    sub = parser.add_subparsers(dest="subcmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--kind", required=True)
    run.add_argument("--budget-mb", type=int, required=True)
    run.add_argument("--device", type=int, default=int(os.environ.get("METAMON_GPU_DEVICE", "0")))
    run.add_argument("--safety-mb", type=int, default=_DEFAULT_SAFETY_MB)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.subcmd == "run":
        raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
