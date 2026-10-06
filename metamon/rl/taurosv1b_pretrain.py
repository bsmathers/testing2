"""Policy-only TaurosV0 -> 35M TaurosV1B bootstrap.

The pre-online schedule is deliberately split into explicit phases so no dense
teacher critic tensors are ever written to disk:

A   150 epochs: public-data policy distillation from TaurosV0@62
B1   50 epochs: 75% public / 25% first student-occupancy (DAgger) pile
B2   50 epochs: 50% public / 25% DAgger-1 / 25% DAgger-2 (legacy path)
C    50 epochs: critic-only warmup from B1, using public + DAgger-1 states
D    25 epochs: joint critic + policy-KL bridge from C, using public + DAgger-1

One epoch is 1000 minibatches by default.  Teacher policy probabilities are
computed on the fly, so persistent teacher-label storage is zero.
"""

from __future__ import annotations

import gc
import itertools
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Optional

import amago
import torch
import torch.nn as nn
from amago.loading import MAGIC_PAD_VAL, RLData_pad_collate
from torch.utils.data import DataLoader

from metamon.rl.dataset_config import (
    CustomReplaySource,
    DatasetConfig,
    build_dataset,
    load_dataset_config,
)
from metamon.rl.pretrained import get_pretrained_model


@dataclass(frozen=True)
class PhaseSpec:
    epochs: int
    lr: float
    warmup_epochs: int
    public_weight: float
    dagger1_weight: float
    dagger2_weight: float


# Project-wide rule: every optimization phase uses a fixed 1e-5 step size.
# No learning-rate warmup/ramp is used anywhere in A-D.
PHASES = {
    "a": PhaseSpec(150, 1.0e-5, 0, 1.00, 0.00, 0.00),
    "b1": PhaseSpec(50, 1.0e-5, 0, 0.75, 0.25, 0.00),
    "b2": PhaseSpec(50, 1.0e-5, 0, 0.50, 0.25, 0.25),
    "c": PhaseSpec(50, 1.0e-5, 0, 0.70, 0.30, 0.00),
    "d": PhaseSpec(25, 1.0e-5, 0, 0.70, 0.30, 0.00),
}


class PolicyOnlyTeacher(nn.Module):
    """Keep only the V0 modules needed to produce policy distributions."""

    def __init__(self, policy):
        super().__init__()
        self.tstep_encoder = policy.tstep_encoder
        self.traj_encoder = policy.traj_encoder
        self.actor = policy.actor
        self.register_buffer("gammas", policy.gammas.detach().clone())
        self.pass_obs_keys_to_actor = tuple(policy.pass_obs_keys_to_actor)

    def probs(self, batch) -> torch.Tensor:
        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        s, _ = self.traj_encoder(
            seq=o, time_idxs=batch.time_idxs, hidden_state=None
        )
        straight = {k: batch.obs[k] for k in self.pass_obs_keys_to_actor}
        return self.actor(s, straight_from_obs=straight).probs


def _student_probs(student, batch) -> torch.Tensor:
    o = student.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
    s, _ = student.traj_encoder(seq=o, time_idxs=batch.time_idxs, hidden_state=None)
    straight = {k: batch.obs[k] for k in student.pass_obs_keys_to_actor}
    return student.actor(s, straight_from_obs=straight).probs


def _shared_gamma_pairs(student_gammas: torch.Tensor, teacher_gammas: torch.Tensor):
    pairs = []
    sg = student_gammas.detach().cpu()
    for t_idx, gamma in enumerate(teacher_gammas.detach().cpu()):
        diffs = (sg - gamma).abs()
        s_idx = int(diffs.argmin())
        if float(diffs[s_idx]) > 1e-5:
            raise RuntimeError(
                f"Teacher gamma {float(gamma):.6f} has no matching student gamma; "
                f"student={sg.tolist()}"
            )
        pairs.append((s_idx, t_idx, float(gamma)))
    return pairs


def _action_mask(batch) -> torch.Tensor:
    # The actor distribution itself already normalizes over legal actions.  This
    # mask removes padding, terminal observations, and replay steps whose action
    # is unknown/missing.
    valid = ~((batch.rl2s == MAGIC_PAD_VAL).all(-1))
    valid = valid[:, :-1]
    missing = batch.obs["missing_action_mask"][:, :-1]
    if missing.ndim == 3:
        missing = missing.squeeze(-1)
    return valid & (~missing.bool())


def policy_kd_loss(student, teacher: PolicyOnlyTeacher, batch) -> torch.Tensor:
    with torch.no_grad():
        teacher_p = teacher.probs(batch)
    student_p = _student_probs(student, batch)
    pairs = _shared_gamma_pairs(student.gammas, teacher.gammas)
    mask = _action_mask(batch)
    denom = mask.sum().clamp(min=1)

    loss = torch.zeros((), device=student_p.device, dtype=torch.float32)
    eps = 1e-8
    for s_idx, t_idx, _gamma in pairs:
        p = teacher_p[:, :-1, t_idx, :].float()
        q = student_p[:, :-1, s_idx, :].float()
        # Illegal actions have exact probability zero in both distributions.
        # xlogy-style masking avoids 0 * log(0).
        positive = p > 0
        log_ratio = torch.zeros_like(p)
        log_ratio[positive] = torch.log(p[positive].clamp_min(eps)) - torch.log(
            q[positive].clamp_min(eps)
        )
        kl = (p * log_ratio).sum(-1)
        loss = loss + (kl * mask.float()).sum() / denom
    return loss / len(pairs)


def _extract_policy(model_name: str, checkpoint: int, weights: Optional[str] = None):
    spec = get_pretrained_model(model_name)
    exp = spec.initialize_agent(checkpoint=checkpoint, log=False)
    policy = exp.policy
    if weights is not None:
        state = torch.load(weights, map_location="cpu")
        policy.load_state_dict(state, strict=True)
        policy.on_checkpoint_loaded(is_resume=False)
    # The placeholder Experiment/optimizer are not used by this trainer.  The
    # policy remains a normal nn.Module after unwrapping.
    del exp
    gc.collect()
    return policy


def _build_teacher(checkpoint: int, device: torch.device) -> PolicyOnlyTeacher:
    full = _extract_policy("TaurosV0", checkpoint)
    teacher = PolicyOnlyTeacher(full).to(device)
    teacher.eval().requires_grad_(False)
    # Drop V0 critics/targets/maximized critics: they are the storage/VRAM object
    # this methodology intentionally avoids distilling.
    del full
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return teacher


def _configure_dataset(dataset, steps_per_epoch: int, batch_size: int, max_seq_len: int):
    stub = SimpleNamespace(
        train_batches_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        accelerator=SimpleNamespace(num_processes=1),
        max_seq_len=max_seq_len,
        padded_sampling="none",
        has_dset_edit_rights=False,
    )
    dataset.configure_from_experiment(stub)
    return dataset


def _one_custom_dataset(path: str, student_spec):
    cfg = DatasetConfig(
        replay_weight=0.0,
        custom_replays=[CustomReplaySource(dir=os.path.abspath(path), weight=1.0)],
        formats=["gen1ou"],
    )
    return build_dataset(
        config=cfg,
        obs_space=student_spec.observation_space,
        action_space=student_spec.action_space,
        reward_function=student_spec.reward_function,
        verbose=False,
    )


def make_phase_dataset(
    phase: str,
    public_config: str,
    dagger1_dir: Optional[str],
    dagger2_dir: Optional[str],
    steps_per_epoch: int,
    batch_size: int,
    max_seq_len: int,
):
    spec = PHASES[phase]
    student_spec = get_pretrained_model("TaurosV1A")
    public = build_dataset(
        config=load_dataset_config(public_config),
        obs_space=student_spec.observation_space,
        action_space=student_spec.action_space,
        reward_function=student_spec.reward_function,
        verbose=True,
    )
    datasets = [public]
    weights = [spec.public_weight]
    if spec.dagger1_weight:
        if not dagger1_dir:
            raise ValueError(f"phase {phase} requires --dagger1_dir")
        datasets.append(_one_custom_dataset(dagger1_dir, student_spec))
        weights.append(spec.dagger1_weight)
    if spec.dagger2_weight:
        if not dagger2_dir:
            raise ValueError(f"phase {phase} requires --dagger2_dir")
        datasets.append(_one_custom_dataset(dagger2_dir, student_spec))
        weights.append(spec.dagger2_weight)

    if len(datasets) == 1:
        mixed = datasets[0]
    else:
        mixed = amago.loading.MixtureOfDatasets(
            datasets=datasets,
            sampling_weights=weights,
            dset_name=f"TaurosV1B pretrain phase {phase}",
        )
    return _configure_dataset(mixed, steps_per_epoch, batch_size, max_seq_len)


def _loader(dataset, batch_size: int, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=RLData_pad_collate,
        pin_memory=torch.cuda.is_available(),
    )


def _optimizer(params: Iterable[torch.nn.Parameter], lr: float):
    params = [p for p in params if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)


def _scheduler(optimizer, warmup_steps: int):
    def scale(step: int):
        if warmup_steps <= 0:
            return 1.0
        return min((step + 1) / warmup_steps, 1.0)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _clip(params, max_norm: float = 1.0):
    return torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], max_norm)


def _save(policy, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(policy.state_dict(), path)
    print(f"Saved: {path}")


def train_kd(
    student,
    teacher,
    dataset,
    phase_spec: PhaseSpec,
    steps_per_epoch: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    student.train()
    teacher.eval()
    params = itertools.chain(
        student.tstep_encoder.parameters(),
        student.traj_encoder.parameters(),
        student.actor.parameters(),
    )
    params = list(params)
    opt = _optimizer(params, phase_spec.lr)
    sched = _scheduler(opt, phase_spec.warmup_epochs * steps_per_epoch)
    loader = _loader(dataset, batch_size, workers)

    global_step = 0
    for epoch in range(phase_spec.epochs):
        running = 0.0
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            loss = policy_kd_loss(student, teacher, batch)
            loss.backward()
            _clip(params)
            opt.step()
            sched.step()
            running += float(loss.detach())
            global_step += 1
        print(
            f"KD epoch {epoch + 1:03d}/{phase_spec.epochs}: "
            f"loss={running / max(steps_per_epoch, 1):.6f} "
            f"lr={sched.get_last_lr()[0]:.3g}"
        )


def _set_requires_grad(module: nn.Module, value: bool):
    for p in module.parameters():
        p.requires_grad_(value)


def train_critic_only(
    student,
    dataset,
    phase_spec: PhaseSpec,
    steps_per_epoch: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    # Critical phase boundary: KD only updates the online encoder/actor.  Never
    # bootstrap a critic from the untouched target_actor/target_critics.
    student.hard_sync_targets()
    _set_requires_grad(student.tstep_encoder, False)
    _set_requires_grad(student.traj_encoder, False)
    _set_requires_grad(student.actor, False)
    student.online_coeff = 0.0
    student.offline_coeff = 0.0
    student.train()

    params = list(student.critics.parameters())
    opt = _optimizer(params, phase_spec.lr)
    sched = _scheduler(opt, phase_spec.warmup_epochs * steps_per_epoch)
    loader = _loader(dataset, batch_size, workers)

    for epoch in range(phase_spec.epochs):
        running = 0.0
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            loss = student(batch, log_step=False)
            loss.backward()
            _clip(params)
            opt.step()
            sched.step()
            student.soft_sync_targets()
            running += float(loss.detach())
        print(
            f"Critic epoch {epoch + 1:03d}/{phase_spec.epochs}: "
            f"loss={running / max(steps_per_epoch, 1):.6f}"
        )


def _bridge_lambda(epoch: int) -> float:
    if epoch < 5:
        return 1.0
    if epoch < 15:
        return 0.5
    return 0.25


def train_bridge(
    student,
    teacher,
    dataset,
    phase_spec: PhaseSpec,
    steps_per_epoch: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    _set_requires_grad(student.tstep_encoder, True)
    _set_requires_grad(student.traj_encoder, True)
    _set_requires_grad(student.actor, True)
    _set_requires_grad(student.critics, True)
    student.online_coeff = 0.0
    student.offline_coeff = 0.0
    student.hard_sync_targets()
    student.train()
    teacher.eval()

    params = list(student.trainable_params)
    opt = _optimizer(params, phase_spec.lr)
    sched = _scheduler(opt, phase_spec.warmup_epochs * steps_per_epoch)
    loader = _loader(dataset, batch_size, workers)

    for epoch in range(phase_spec.epochs):
        lam = _bridge_lambda(epoch)
        running = 0.0
        running_kd = 0.0
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            critic_loss = student(batch, log_step=False)
            kd = policy_kd_loss(student, teacher, batch)
            loss = critic_loss + lam * kd
            loss.backward()
            _clip(params)
            opt.step()
            sched.step()
            student.soft_sync_targets()
            running += float(loss.detach())
            running_kd += float(kd.detach())
        print(
            f"Bridge epoch {epoch + 1:03d}/{phase_spec.epochs}: "
            f"loss={running / max(steps_per_epoch, 1):.6f} "
            f"kd={running_kd / max(steps_per_epoch, 1):.6f} lambda={lam:.2f}"
        )

    # Phase E loads a raw policy state_dict.  End with internally consistent
    # targets rather than carrying an arbitrary EMA lag across the run boundary.
    student.hard_sync_targets()


def main():
    p = ArgumentParser(description="TaurosV1B V0->35M staged policy distillation")
    p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--input_weights", default=None)
    p.add_argument("--output_weights", required=True)
    p.add_argument("--teacher_checkpoint", type=int, default=62)
    p.add_argument("--public_config", default="online_selfplay_taurosv1b.yaml")
    p.add_argument("--dagger1_dir", default=None)
    p.add_argument("--dagger2_dir", default=None)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dloader_workers", type=int, default=0)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    phase_spec = PHASES[args.phase]
    if args.phase != "a" and not args.input_weights:
        raise ValueError(f"phase {args.phase} requires --input_weights")

    device = torch.device(args.device)
    student = _extract_policy(
        "TaurosV1A",
        checkpoint=0,
        weights=args.input_weights,
    ).to(device)

    dataset = make_phase_dataset(
        phase=args.phase,
        public_config=args.public_config,
        dagger1_dir=args.dagger1_dir,
        dagger2_dir=args.dagger2_dir,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    teacher = None
    if args.phase in {"a", "b1", "b2", "d"}:
        teacher = _build_teacher(args.teacher_checkpoint, device)

    if args.phase in {"a", "b1", "b2"}:
        assert teacher is not None
        train_kd(
            student, teacher, dataset, phase_spec,
            args.steps_per_epoch, args.batch_size, args.dloader_workers, device,
        )
    elif args.phase == "c":
        train_critic_only(
            student, dataset, phase_spec,
            args.steps_per_epoch, args.batch_size, args.dloader_workers, device,
        )
    elif args.phase == "d":
        assert teacher is not None
        train_bridge(
            student, teacher, dataset, phase_spec,
            args.steps_per_epoch, args.batch_size, args.dloader_workers, device,
        )

    _save(student, args.output_weights)


if __name__ == "__main__":
    main()
