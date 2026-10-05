"""W&B wrapper for TaurosV1B phases A-D.

This intentionally leaves ``taurosv1b_pretrain`` optimization semantics untouched.
It launches the existing trainer as a child process, mirrors stdout/stderr to the
terminal, parses its once-per-epoch summaries, and records them to Weights & Biases.

A-D are stage-resumable rather than mid-stage-resumable. Therefore each retry is
logged as a new W&B run (under the same experiment group) instead of attempting to
resume an earlier W&B step sequence while the underlying optimizer restarts.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Optional

import wandb


_PHASE_EPOCHS = {"a": 150, "b1": 50, "b2": 50, "c": 50, "d": 25}
_PHASE_OFFSETS = {"a": 0, "b1": 150, "b2": 200, "c": 250, "d": 300}

_KD_RE = re.compile(
    r"^KD epoch\s+(?P<epoch>\d+)/(?P<total>\d+):\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+lr=(?P<lr>[-+0-9.eE]+)"
)
_CRITIC_RE = re.compile(
    r"^Critic epoch\s+(?P<epoch>\d+)/(?P<total>\d+):\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)"
)
_BRIDGE_RE = re.compile(
    r"^Bridge epoch\s+(?P<epoch>\d+)/(?P<total>\d+):\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+kd=(?P<kd>[-+0-9.eE]+)\s+"
    r"lambda=(?P<lam>[-+0-9.eE]+)"
)


def _metric_from_line(line: str, phase: str) -> Optional[tuple[int, dict]]:
    match = _KD_RE.match(line)
    if match:
        epoch = int(match.group("epoch"))
        return epoch, {
            "pretrain/loss": float(match.group("loss")),
            "pretrain/kd_loss": float(match.group("loss")),
            "pretrain/lr": float(match.group("lr")),
        }

    match = _CRITIC_RE.match(line)
    if match:
        epoch = int(match.group("epoch"))
        return epoch, {
            "pretrain/loss": float(match.group("loss")),
            "pretrain/critic_loss": float(match.group("loss")),
        }

    match = _BRIDGE_RE.match(line)
    if match:
        epoch = int(match.group("epoch"))
        total_loss = float(match.group("loss"))
        kd_loss = float(match.group("kd"))
        lam = float(match.group("lam"))
        return epoch, {
            "pretrain/loss": total_loss,
            "pretrain/kd_loss": kd_loss,
            "pretrain/kd_lambda": lam,
            "pretrain/critic_component": total_loss - lam * kd_loss,
        }

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a TaurosV1B A-D pretraining phase with W&B logging.",
        add_help=False,
    )
    parser.add_argument("--phase", choices=_PHASE_EPOCHS, required=True)
    parser.add_argument(
        "--wandb_project",
        default=os.environ.get("METAMON_WANDB_PROJECT", "taurosv1b"),
    )
    parser.add_argument(
        "--wandb_entity",
        default=os.environ.get("METAMON_WANDB_ENTITY") or None,
    )
    known, forwarded = parser.parse_known_args()

    phase = known.phase
    group = os.environ.get("WANDB_RUN_GROUP")
    tags = [
        tag
        for tag in os.environ.get(
            "WANDB_TAGS", "taurosv1b,distilled-public"
        ).split(",")
        if tag
    ]

    run = wandb.init(
        project=known.wandb_project,
        entity=known.wandb_entity,
        group=group,
        name=f"taurosv1b-pretrain-{phase}",
        job_type="pretrain",
        tags=tags,
        config={
            "phase": phase,
            "phase_epochs": _PHASE_EPOCHS[phase],
            "global_epoch_offset": _PHASE_OFFSETS[phase],
            "trainer_module": "metamon.rl.taurosv1b_pretrain",
        },
    )
    wandb.define_metric("pretrain/global_epoch")
    wandb.define_metric("pretrain/*", step_metric="pretrain/global_epoch")

    # -u is important because stdout is piped through this process; without it,
    # child epoch summaries can be block-buffered and appear in W&B much later.
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "metamon.rl.taurosv1b_pretrain",
        "--phase",
        phase,
        *forwarded,
    ]
    print("[W&B wrapper] launching:", " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    exit_code = 1
    try:
        for raw_line in proc.stdout:
            print(raw_line, end="", flush=True)
            line = raw_line.rstrip("\n")
            parsed = _metric_from_line(line, phase)
            if parsed is None:
                continue
            phase_epoch, metrics = parsed
            global_epoch = _PHASE_OFFSETS[phase] + phase_epoch
            metrics.update(
                {
                    "pretrain/global_epoch": global_epoch,
                    "pretrain/phase_epoch": phase_epoch,
                    "pretrain/phase_progress": phase_epoch / _PHASE_EPOCHS[phase],
                }
            )
            run.log(metrics)

        exit_code = proc.wait()
        run.log(
            {
                "pretrain/exit_code": exit_code,
                "pretrain/completed": int(exit_code == 0),
            }
        )
        if exit_code != 0:
            raise SystemExit(exit_code)
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
