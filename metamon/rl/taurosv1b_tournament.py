"""Periodic TaurosV1B vs TaurosV0@62 tournament hook.

Imported by the V1B E/F training gin files.  When the V1B online runner is the
entrypoint, this module patches its checkpoint hook so every five learner epochs
(after epoch 0) it pauses learning, plays exactly 50 single-lane games against
TaurosV0@62, and logs the win rate to the same W&B/Accelerate tracker.

The tournament uses the in-memory trainee.  It does not create a temporary policy
checkpoint, so the 5-epoch monitoring cadence adds no persistent model storage.
Persistent raw policies are still kept every 25 epochs and full Accelerate states
every 100 epochs.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import amago
import gin
import torch

from metamon.rl import online_rl as legacy
from metamon.rl.evaluate.opponent_pool import load_simple_opponent_pool
from metamon.rl.metamon_to_amago import mirror_online_experiment_gin_bindings
from metamon.rl.pretrained import get_pretrained_model

TOURNAMENT_INTERVAL = 5
TOURNAMENT_GAMES = 50
TOURNAMENT_V0_CHECKPOINT = 62
TOURNAMENT_TEAM_SET = "modern_replays_v2"
PERSISTENT_POLICY_INTERVAL = 25
TOURNAMENT_MAX_TIMESTEPS = 100_000


def _force_reload_training_gin(experiment) -> None:
    """Restore trainee gin after the tournament loads the V0 opponent.

    ``MetamonOnlineExperiment._reload_gin`` intentionally no-ops in learn-only
    mode because its ordinary placeholder envs do not load opponents.  The
    periodic tournament *does* load an opponent, so we need the same restoration
    logic without that learn-only guard.
    """
    config = getattr(experiment, "_gin_config", None)
    files = getattr(experiment, "_gin_config_files", None)
    if config is None or files is None:
        return
    gin.clear_config()
    amago.cli_utils.use_config(config, files, finalize=False)
    for name, value in getattr(experiment, "_gin_extra_bindings", {}).items():
        try:
            gin.bind_parameter(name, value)
        except ValueError:
            pass
    mirror_online_experiment_gin_bindings()
    gin.finalize()


def _run_v0_tournament(experiment) -> float:
    """Run exactly 50 sequential games and return V1B's win rate."""
    if experiment.accelerator.num_processes != 1:
        raise RuntimeError(
            "The periodic V1B tournament currently requires a single learner "
            "process. Launch V1B with one GPU/process or disable the hook."
        )

    trainee_spec = get_pretrained_model("TaurosV1A")
    opponent = load_simple_opponent_pool(
        opponent_agent="TaurosV0",
        battle_format="gen1ou",
        team_set=TOURNAMENT_TEAM_SET,
        checkpoint=TOURNAMENT_V0_CHECKPOINT,
        temperature=1.0,
        battle_backend="metamon",
    )
    make_env = legacy._make_val_env(
        trainee_spec,
        battle_format="gen1ou",
        reward_function=trainee_spec.reward_function,
        val_opponent_kwargs={"opponent_config": opponent},
        # One lane is intentional: AMAGO's episode stopping condition is global
        # across lanes, so one lane guarantees exactly 50 completed games rather
        # than potentially overshooting by several simultaneous terminations.
        lanes=1,
        n_workers=1,
        seed=10_000 + int(experiment.epoch),
        team_set_name=TOURNAMENT_TEAM_SET,
    )

    try:
        metrics = experiment.evaluate_test(
            make_env,
            timesteps=TOURNAMENT_MAX_TIMESTEPS,
            episodes=TOURNAMENT_GAMES,
        )
    finally:
        _force_reload_training_gin(experiment)

    win_values = [
        float(v)
        for k, v in metrics.items()
        if k.startswith("Average Win Rate in ")
    ]
    if not win_values:
        raise RuntimeError(
            "50-game V0 tournament completed without an 'Average Win Rate' metric."
        )
    # There is one single-lane test env, but averaging is harmless if its naming
    # changes and more than one equivalent metric is emitted later.
    return sum(win_values) / len(win_values)


def _patched_save_checkpoint(self) -> None:
    """Policy/full-state checkpointing plus the periodic V0 tournament."""
    epoch = int(self.epoch)
    main = self.accelerator.is_main_process

    # Persistent policies: 0, 25, 50, ...  The tournament itself evaluates the
    # live model and therefore needs no extra checkpoint at epochs 5/10/15/...
    if main and epoch % PERSISTENT_POLICY_INTERVAL == 0:
        path = os.path.join(
            self.ckpt_dir,
            "policy_weights",
            f"policy_epoch_{epoch}.pt",
        )
        torch.save(self.policy.state_dict(), path)

    full_interval: Optional[int] = getattr(self, "full_state_ckpt_interval", 100)
    if (
        full_interval is not None
        and full_interval > 0
        and epoch % int(full_interval) == 0
    ):
        ckpt_name = f"{self.run_name}_epoch_{epoch}"
        self.accelerator.save_state(
            os.path.join(self.ckpt_dir, "training_states", ckpt_name),
            safe_serialization=True,
        )

    # Epoch 0 is useful as a checkpoint but is not "five epochs of E/F" yet.
    tournament_due = epoch > 0 and epoch % TOURNAMENT_INTERVAL == 0
    if not tournament_due:
        return

    # The tracked V1B launcher is single-GPU.  Make the synchronization explicit
    # so an accidental distributed launch fails cleanly instead of desynchronizing.
    self.accelerator.wait_for_everyone()
    if main:
        win_rate = _run_v0_tournament(self)
        stderr = math.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / TOURNAMENT_GAMES)
        self.log(
            {
                "v0_62_win_rate": win_rate,
                "v0_62_games": TOURNAMENT_GAMES,
                "v0_62_binomial_stderr": stderr,
            },
            key="tournament",
        )
        print(
            f"[V1B tournament] epoch={epoch}  vs TaurosV0@62: "
            f"{win_rate:.1%} over {TOURNAMENT_GAMES} games",
            flush=True,
        )
    self.accelerator.wait_for_everyone()


def _install_patch() -> None:
    # ``python -m metamon.rl.taurosv1b_online`` executes that file as __main__.
    # When imported normally (e.g. by the DAgger collector), use its canonical
    # module name.  Avoid importing the runner here: doing so from a gin import
    # during ``-m`` execution would execute the runner a second time.
    for module_name in ("__main__", "metamon.rl.taurosv1b_online"):
        module = sys.modules.get(module_name)
        cls = getattr(module, "TaurosV1BOnlineExperiment", None) if module else None
        if cls is not None:
            cls.save_checkpoint = _patched_save_checkpoint
            return


_install_patch()
