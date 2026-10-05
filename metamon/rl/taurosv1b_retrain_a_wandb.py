"""W&B wrapper for the fixed-LR second Phase-A distillation pass."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import wandb

_EPOCHS = 150
_OFFSET = 150
_KD_RE = re.compile(
    r"^KD epoch\s+(?P<epoch>\d+)/(?P<total>\d+):\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+lr=(?P<lr>[-+0-9.eE]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--wandb_project",
        default=os.environ.get("METAMON_WANDB_PROJECT", "taurosv1b"),
    )
    parser.add_argument(
        "--wandb_entity",
        default=os.environ.get("METAMON_WANDB_ENTITY") or None,
    )
    known, forwarded = parser.parse_known_args()

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
        name="taurosv1b-pretrain-a-fixed-1e-5",
        job_type="pretrain",
        tags=tags,
        config={
            "phase": "a_fixed_1e-5",
            "phase_epochs": _EPOCHS,
            "global_epoch_offset": _OFFSET,
            "learning_rate": 1.0e-5,
            "warmup_epochs": 0,
            "optimizer_state_restored": False,
            "trainer_module": "metamon.rl.taurosv1b_retrain_a",
        },
    )
    wandb.define_metric("pretrain/global_epoch")
    wandb.define_metric("pretrain/*", step_metric="pretrain/global_epoch")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "metamon.rl.taurosv1b_retrain_a",
        "--phase",
        "a",
        *forwarded,
    ]
    print("[W&B wrapper] launching:", " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    exit_code = 1
    try:
        for raw_line in proc.stdout:
            print(raw_line, end="", flush=True)
            match = _KD_RE.match(raw_line.rstrip("\n"))
            if match is None:
                continue
            epoch = int(match.group("epoch"))
            loss = float(match.group("loss"))
            lr = float(match.group("lr"))
            run.log(
                {
                    "pretrain/global_epoch": _OFFSET + epoch,
                    "pretrain/phase_epoch": epoch,
                    "pretrain/phase_progress": epoch / _EPOCHS,
                    "pretrain/loss": loss,
                    "pretrain/kd_loss": loss,
                    "pretrain/lr": lr,
                }
            )

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
