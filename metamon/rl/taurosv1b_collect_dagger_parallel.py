"""Run TaurosV1B DAgger collection across independent GPU-backed shards.

A single collector is coordinator-bound in Python. This launcher runs a small
number of collectors concurrently, each with its own replay directory, then
hard-links completed replays into the canonical DAgger directory. TaurosV1A
uses FlashAttention and therefore must keep CUDA visible during collection.
Shards are count-resumable. If the source actor/config changes, stale DAgger
data is discarded automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_replays(root: Path) -> int:
    fmt = root / "gen1ou"
    if not fmt.is_dir():
        return 0
    return sum(
        1
        for p in fmt.iterdir()
        if p.is_file() and (p.name.endswith(".json") or p.name.endswith(".json.lz4"))
    )


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _reset_if_stale(root: Path, marker: Path, signature: dict) -> None:
    old = _read_json(marker)
    if old == signature:
        return
    if root.exists():
        print(f"Discarding stale DAgger data: {root}", flush=True)
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(marker, signature)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Parallel TaurosV1B DAgger collector")
    p.add_argument("--weights", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_games", type=int, default=75_000)
    p.add_argument("--shards", type=int, default=min(os.cpu_count() or 1, 4))
    p.add_argument("--lanes_per_shard", type=int, default=16)
    p.add_argument("--workers_per_shard", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--train_pool",
        default="metamon/rl/configs/opponent_pools/hl_gen1ou_taurosv1b.yaml",
    )
    p.add_argument(
        "--train_team_set",
        default="metamon/rl/configs/team_sets/taurosv1b_public_train.yaml",
    )
    p.add_argument("--temp_low", type=float, default=1.0)
    p.add_argument("--temp_high", type=float, default=1.5)
    args = p.parse_args()

    if args.target_games <= 0:
        raise ValueError("--target_games must be positive")
    if args.shards <= 0:
        raise ValueError("--shards must be positive")
    if args.lanes_per_shard <= 0 or args.workers_per_shard <= 0:
        raise ValueError("lanes/workers per shard must be positive")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(args.weights)

    weights = os.path.abspath(args.weights)
    output = Path(args.output_dir).resolve()
    shard_root = Path(f"{output}_shards")
    signature = {
        "version": 2,
        "weights": weights,
        "weights_sha256": _sha256(weights),
        "target_games": args.target_games,
        "shards": args.shards,
        "lanes_per_shard": args.lanes_per_shard,
        "workers_per_shard": args.workers_per_shard,
        "train_pool": args.train_pool,
        "train_team_set": args.train_team_set,
        "temp_low": args.temp_low,
        "temp_high": args.temp_high,
        "cuda_required": True,
    }

    output_marker = output / ".taurosv1b_parallel_dagger_source.json"
    shard_marker = shard_root / ".taurosv1b_parallel_dagger_source.json"
    _reset_if_stale(output, output_marker, signature)
    _reset_if_stale(shard_root, shard_marker, signature)

    if _count_replays(output) >= args.target_games:
        print(
            f"[skip] matching DAgger set already has {_count_replays(output):,} "
            f"replays in {output}",
            flush=True,
        )
        return

    base = args.target_games // args.shards
    remainder = args.target_games % args.shards
    targets = [base + (1 if i < remainder else 0) for i in range(args.shards)]

    children = []
    log_handles = []
    # TaurosV1A uses FlashAttention, whose inference path is CUDA-only in this
    # environment.  Do not hide CUDA from collectors.  Keep the default shard
    # count intentionally small to avoid replicating too many policies/KV caches
    # on a 16 GB training GPU.
    child_env = os.environ.copy()

    try:
        for i, target in enumerate(targets):
            shard = shard_root / f"shard_{i:02d}"
            shard.mkdir(parents=True, exist_ok=True)
            log_path = shard_root / f"shard_{i:02d}.log"
            log = log_path.open("a", encoding="utf-8")
            log_handles.append(log)
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "metamon.rl.taurosv1b_collect_dagger",
                "--weights",
                weights,
                "--output_dir",
                str(shard),
                "--target_games",
                str(target),
                "--train_pool",
                args.train_pool,
                "--train_team_set",
                args.train_team_set,
                "--lanes",
                str(args.lanes_per_shard),
                "--n_workers",
                str(args.workers_per_shard),
                "--temp_low",
                str(args.temp_low),
                "--temp_high",
                str(args.temp_high),
                "--seed",
                str(args.seed + i),
            ]
            print(
                f"Starting DAgger shard {i + 1}/{args.shards}: "
                f"target={target:,}, existing={_count_replays(shard):,}",
                flush=True,
            )
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_env,
            )
            children.append((i, proc, log_path))

        failures = []
        for i, proc, log_path in children:
            code = proc.wait()
            if code != 0:
                failures.append((i, code, log_path))
            else:
                print(f"DAgger shard {i + 1}/{args.shards} complete.", flush=True)

        if failures:
            details = ", ".join(
                f"shard {i} exit={code} log={path}"
                for i, code, path in failures
            )
            raise RuntimeError(f"parallel DAgger failed: {details}")
    except BaseException:
        for _, proc, _ in children:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, _ in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        raise
    finally:
        for log in log_handles:
            log.close()

    # Rebuild the canonical replay directory deterministically from completed shards.
    fmt = output / "gen1ou"
    if fmt.exists():
        shutil.rmtree(fmt)
    fmt.mkdir(parents=True, exist_ok=True)

    for i in range(args.shards):
        src_fmt = shard_root / f"shard_{i:02d}" / "gen1ou"
        if not src_fmt.is_dir():
            raise RuntimeError(f"missing completed shard directory: {src_fmt}")
        for src in src_fmt.iterdir():
            if not src.is_file() or not (
                src.name.endswith(".json") or src.name.endswith(".json.lz4")
            ):
                continue
            dst = fmt / src.name
            if dst.exists():
                dst = fmt / f"shard{i:02d}_{src.name}"
            _link_or_copy(src, dst)

    merged = _count_replays(output)
    if merged < args.target_games:
        raise RuntimeError(
            f"merged DAgger set has only {merged:,}/{args.target_games:,} replays"
        )
    _write_json(output_marker, signature)
    print(f"Parallel DAgger complete: {merged:,} replays in {output}", flush=True)


if __name__ == "__main__":
    main()
