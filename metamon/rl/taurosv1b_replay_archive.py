"""Archive phase-F online replays without changing the learner's 150k FIFO.

The online collector writes completed ``.json.lz4`` trajectories into the normal
FIFO.  This process periodically snapshots newly visible FIFO files into a
separate archive, using hard links when both directories share a filesystem and
falling back to copies otherwise.  FIFO eviction can then unlink its pathname
without deleting the archived replay.

Only atomically-completed replay filenames are considered (never ``.tmp`` files).
The archive is capped at ``--max_files`` (2,000,000 by default) and is restartable.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import signal
import time
from pathlib import Path


_VALID_SUFFIXES = (".json", ".json.lz4")
_STOP = False


def _handle_stop(_signum, _frame):
    global _STOP
    _STOP = True


def _is_replay_name(name: str) -> bool:
    return name.endswith(_VALID_SUFFIXES) and not name.endswith(".tmp")


def _count_replays(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for e in os.scandir(directory) if e.is_file() and _is_replay_name(e.name))


def _write_state(path: Path, count: int, max_files: int) -> None:
    payload = {
        "count": int(count),
        "max_files": int(max_files),
        "updated_unix": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    os.replace(tmp, path)


def _load_count(archive_dir: Path, state_path: Path, max_files: int) -> int:
    """Trust the saved count only if no archive-directory mutation followed it."""
    if state_path.is_file() and archive_dir.is_dir():
        try:
            state = json.loads(state_path.read_text())
            state_count = int(state["count"])
            if int(state.get("max_files", max_files)) == max_files:
                # A link/copy updates the directory mtime.  State is written after
                # each scan, so an archive newer than state indicates an interrupted
                # scan and requires a recount.
                if archive_dir.stat().st_mtime_ns <= state_path.stat().st_mtime_ns:
                    return state_count
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    return _count_replays(archive_dir)


def _link_or_copy(src: Path, dst: Path) -> bool:
    """Return True only when a new archive entry was created."""
    try:
        os.link(src, dst)
        return True
    except FileExistsError:
        return False
    except FileNotFoundError:
        # FIFO eviction can race a scan; the next replay is still safe to process.
        return False
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    # Different filesystems: copy to a temporary pathname, then publish atomically.
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    try:
        shutil.copy2(src, tmp)
        try:
            os.link(tmp, dst)
            created = True
        except FileExistsError:
            created = False
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        return created
    except FileNotFoundError:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        return False


def archive_once(
    source_dir: Path,
    archive_dir: Path,
    *,
    max_files: int,
    count: int,
    previous_source_names: set[str] | None,
) -> tuple[int, set[str], int]:
    """Archive files newly appearing in the FIFO since the previous scan."""
    current_names: set[str] = set()
    if not source_dir.is_dir():
        return count, current_names, 0

    entries = []
    for entry in os.scandir(source_dir):
        if not entry.is_file() or not _is_replay_name(entry.name):
            continue
        current_names.add(entry.name)
        if previous_source_names is None or entry.name not in previous_source_names:
            entries.append(entry.name)

    # Stable ordering makes the cap deterministic if it is reached mid-scan.
    entries.sort()
    added = 0
    for name in entries:
        if count >= max_files:
            break
        src = source_dir / name
        dst = archive_dir / name
        if _link_or_copy(src, dst):
            count += 1
            added += 1
    return count, current_names, added


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive TaurosV1B phase-F replays")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--max_files", type=int, default=2_000_000)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.max_files <= 0:
        raise ValueError("--max_files must be positive")
    if args.poll_seconds <= 0 and not args.once:
        raise ValueError("--poll_seconds must be positive")

    source_dir = Path(args.source_dir).resolve()
    archive_dir = Path(args.archive_dir).resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)
    state_path = archive_dir.parent / ".taurosv1b_archive_state.json"

    count = _load_count(archive_dir, state_path, args.max_files)
    if count > args.max_files:
        # Never delete user data automatically.  Treat an already-oversized archive
        # as full and leave its contents untouched.
        print(
            f"Archive already contains {count:,} replays (> cap {args.max_files:,}); "
            "no new files will be added.",
            flush=True,
        )
        return

    print(
        f"Phase-F replay archive: {archive_dir} ({count:,}/{args.max_files:,}); "
        f"source={source_dir}",
        flush=True,
    )

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    previous_names: set[str] | None = None
    while True:
        count, previous_names, added = archive_once(
            source_dir,
            archive_dir,
            max_files=args.max_files,
            count=count,
            previous_source_names=previous_names,
        )
        _write_state(state_path, count, args.max_files)
        if added:
            print(
                f"Archived +{added:,} phase-F replays ({count:,}/{args.max_files:,})",
                flush=True,
            )

        if args.once or count >= args.max_files or _STOP:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
