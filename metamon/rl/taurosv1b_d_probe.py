"""Short Phase-D probe from a completed TaurosV1B Phase-C checkpoint.

The probe trains the ordinary Phase-D bridge for only a few epochs on a
deterministic train split and evaluates C vs D on the corresponding held-out
test split. The evaluation reports:
  * weighted AMAGO critic loss (actor coefficients forced to zero),
  * V0 policy KD loss,
  * their deltas.

This is intended to answer whether Phase D is improving the critic enough to
justify a full bridge before entering online Phase E.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import torch

from metamon.rl.dataset_config import (
    CustomReplaySource,
    DatasetConfig,
    build_dataset,
    flatten_config,
    load_dataset_config,
)
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.taurosv1b_pretrain import (
    PHASES,
    _build_teacher,
    _configure_dataset,
    _extract_policy,
    _loader,
    policy_kd_loss,
    train_bridge,
)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scaled_probe_config(public_config: str, dagger1_dir: str) -> DatasetConfig:
    """Match Phase D's 70% public / 30% DAgger1 mixture."""
    public = flatten_config(load_dataset_config(public_config))
    public_custom = list(public.custom_replays or [])
    public_total = (
        float(public.replay_weight)
        + sum((public.self_play or {}).values())
        + sum(cr.weight for cr in public_custom)
    )
    if public_total <= 0:
        raise ValueError(f"Public dataset config has zero total weight: {public_config}")

    scale = 0.70 / public_total
    self_play = {
        name: weight * scale for name, weight in (public.self_play or {}).items()
    }
    custom = [
        CustomReplaySource(dir=cr.dir, weight=cr.weight * scale)
        for cr in public_custom
    ]
    custom.append(CustomReplaySource(dir=os.path.abspath(dagger1_dir), weight=0.30))

    return DatasetConfig(
        replay_weight=float(public.replay_weight) * scale,
        self_play=self_play or None,
        custom_replays=custom,
        formats=public.formats,
    )


def _make_probe_dataset(
    config: DatasetConfig,
    split: str,
    test_fraction: float,
    split_seed: int,
    steps: int,
    batch_size: int,
    max_seq_len: int,
    verbose: bool,
):
    spec = get_pretrained_model("TaurosV1A")
    dataset = build_dataset(
        config=config,
        obs_space=spec.observation_space,
        action_space=spec.action_space,
        reward_function=spec.reward_function,
        verbose=verbose,
        split=split,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )
    return _configure_dataset(dataset, steps, batch_size, max_seq_len)


@torch.no_grad()
def _evaluate(
    student,
    teacher,
    dataset,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Evaluate on a fixed RNG replay stream without mutating PopArt statistics."""
    _seed_all(seed)
    loader = _loader(dataset, batch_size, workers=0)

    old_online = student.online_coeff
    old_offline = student.offline_coeff
    was_training = student.training
    student.online_coeff = 0.0
    student.offline_coeff = 0.0
    student.eval()
    teacher.eval()

    critic_sum = 0.0
    kd_sum = 0.0
    seen = 0

    # Agent.forward updates PopArt statistics even under no_grad. Freeze that
    # update so evaluation cannot alter the checkpoint being compared/saved.
    with patch.object(student.popart, "update_stats", lambda *args, **kwargs: None):
        for step, batch in enumerate(loader):
            if step >= steps:
                break
            batch = batch.to(device)
            critic = student(batch, log_step=False)
            kd = policy_kd_loss(student, teacher, batch)
            critic_sum += float(critic.detach())
            kd_sum += float(kd.detach())
            seen += 1

    student.online_coeff = old_online
    student.offline_coeff = old_offline
    student.train(was_training)

    if seen == 0:
        raise RuntimeError("Held-out probe loader produced zero batches")
    return {
        "critic_loss": critic_sum / seen,
        "kd_loss": kd_sum / seen,
        "batches": seen,
    }


def _atomic_json(path: str, payload: dict) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Probe whether TaurosV1B Phase D helps")
    p.add_argument("--input_weights", required=True)
    p.add_argument("--output_weights", required=True)
    p.add_argument("--output_json", default=None)
    p.add_argument("--dagger1_dir", required=True)
    p.add_argument("--public_config", default="online_selfplay_taurosv1b.yaml")
    p.add_argument("--teacher_checkpoint", type=int, default=62)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--eval_batches", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dloader_workers", type=int, default=0)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--eval_seed", type=int, default=20261006)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.eval_batches <= 0:
        raise ValueError("--eval_batches must be positive")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test_fraction must be in (0, 1)")
    if not os.path.isfile(args.input_weights):
        raise FileNotFoundError(args.input_weights)

    device = torch.device(args.device)
    config = _scaled_probe_config(args.public_config, args.dagger1_dir)

    train_dataset = _make_probe_dataset(
        config,
        "train",
        args.test_fraction,
        args.split_seed,
        args.steps_per_epoch,
        args.batch_size,
        args.max_seq_len,
        verbose=True,
    )
    eval_dataset = _make_probe_dataset(
        config,
        "test",
        args.test_fraction,
        args.split_seed,
        args.eval_batches,
        args.batch_size,
        args.max_seq_len,
        verbose=True,
    )

    student = _extract_policy(
        "TaurosV1A", checkpoint=0, weights=args.input_weights
    ).to(device)
    teacher = _build_teacher(args.teacher_checkpoint, device)

    # Match the exact start-of-D target semantics. Phase D hard-syncs before its
    # first update, so compare the C baseline after the same hard sync.
    student.hard_sync_targets()
    baseline = _evaluate(
        student,
        teacher,
        eval_dataset,
        steps=args.eval_batches,
        batch_size=args.batch_size,
        device=device,
        seed=args.eval_seed,
    )
    print(
        f"Probe baseline C: critic={baseline['critic_loss']:.6f} "
        f"kd={baseline['kd_loss']:.6f} batches={baseline['batches']}",
        flush=True,
    )

    probe_spec = replace(PHASES["d"], epochs=args.epochs)
    train_bridge(
        student,
        teacher,
        train_dataset,
        probe_spec,
        args.steps_per_epoch,
        args.batch_size,
        args.dloader_workers,
        device,
        early_stop_min_epochs=args.epochs + 1,
        early_stop_patience=0,
    )

    post = _evaluate(
        student,
        teacher,
        eval_dataset,
        steps=args.eval_batches,
        batch_size=args.batch_size,
        device=device,
        seed=args.eval_seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_weights)), exist_ok=True)
    torch.save(student.state_dict(), args.output_weights)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_weights": os.path.abspath(args.input_weights),
        "output_weights": os.path.abspath(args.output_weights),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "eval_batches": args.eval_batches,
        "batch_size": args.batch_size,
        "test_fraction": args.test_fraction,
        "split_seed": args.split_seed,
        "eval_seed": args.eval_seed,
        "phase_d_lr": probe_spec.lr,
        "kd_lambda": 1.0,
        "baseline_c": baseline,
        "post_d_probe": post,
        "delta": {
            "critic_loss": post["critic_loss"] - baseline["critic_loss"],
            "critic_loss_fraction": (
                post["critic_loss"] / baseline["critic_loss"] - 1.0
                if baseline["critic_loss"] != 0
                else None
            ),
            "kd_loss": post["kd_loss"] - baseline["kd_loss"],
        },
    }

    output_json = args.output_json or f"{args.output_weights}.probe.json"
    _atomic_json(output_json, payload)

    print("\n======================================")
    print(f"D probe epochs:          {args.epochs}")
    print(f"C held-out critic loss:  {baseline['critic_loss']:.6f}")
    print(f"D held-out critic loss:  {post['critic_loss']:.6f}")
    print(
        "Critic relative change:  "
        f"{100.0 * payload['delta']['critic_loss_fraction']:+.2f}%"
    )
    print(f"C held-out KD:           {baseline['kd_loss']:.6f}")
    print(f"D held-out KD:           {post['kd_loss']:.6f}")
    print(f"Saved weights:           {os.path.abspath(args.output_weights)}")
    print(f"Saved metrics:           {os.path.abspath(output_json)}")
    print("======================================")


if __name__ == "__main__":
    main()
