"""Periodic TaurosV1B vs TaurosV0@62 tournament and resume hooks.

Imported by the V1B E/F training gin files. Every five learner epochs (after
epoch 0) the current in-memory policy plays exactly 50 single-lane games against
TaurosV0@62 and logs the result to the same W&B/Accelerate tracker.

The hooks also restore non-tensor training counters that AMAGO's Accelerate state
does not serialize. Evaluation RNG state is restored afterwards so monitoring is
observational rather than an intervention in training.
"""

from __future__ import annotations

import gc
import math
import os
import random
import sys
from typing import Optional

import amago
import gin
import numpy as np
import torch

from metamon.rl import online_rl as legacy
from metamon.rl.gpu_job_queue import gpu_job_lease
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


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_v0_tournament(experiment) -> float:
    """Run exactly 50 sequential games and return V1B's win rate."""
    if experiment.accelerator.num_processes != 1:
        raise RuntimeError(
            "The periodic V1B tournament requires a single learner process."
        )

    rng_state = _capture_rng_state()
    was_training = bool(experiment.policy.training)
    metrics = None
    try:
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
            lanes=1,
            n_workers=1,
            seed=10_000 + int(experiment.epoch),
            team_set_name=TOURNAMENT_TEAM_SET,
        )
        metrics = experiment.evaluate_test(
            make_env,
            timesteps=TOURNAMENT_MAX_TIMESTEPS,
            episodes=TOURNAMENT_GAMES,
        )
    finally:
        # Opponent initialization clears/rebinds global gin. Restore trainee
        # configuration, RNG, and module mode before the next optimization step.
        _force_reload_training_gin(experiment)
        _restore_rng_state(rng_state)
        experiment.policy.train(was_training)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if metrics is None:
        raise RuntimeError("V0 tournament failed before producing metrics.")
    win_values = [
        float(v)
        for k, v in metrics.items()
        if k.startswith("Average Win Rate in ")
    ]
    if not win_values:
        raise RuntimeError(
            "50-game V0 tournament completed without an 'Average Win Rate' metric."
        )
    return sum(win_values) / len(win_values)


def _patched_save_checkpoint(self) -> None:
    """Policy/full-state checkpointing plus the periodic V0 tournament."""
    epoch = int(self.epoch)
    main = self.accelerator.is_main_process

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

    if epoch <= 0 or epoch % TOURNAMENT_INTERVAL != 0:
        return

    self.accelerator.wait_for_everyone()
    if main:
        budget_mb = int(os.environ.get("TOURNAMENT_GPU_BUDGET_MB", "3000"))
        device = int(os.environ.get("METAMON_GPU_DEVICE", "0"))
        # The learner already owns its base GPU reservation.  Tournament is an
        # incremental reservation for the extra opponent policy/KV cache.  If a
        # collector burst currently occupies that headroom, block here; FIFO
        # ordering guarantees the tournament runs before the collector reacquires.
        with gpu_job_lease("tournament", budget_mb=budget_mb, device=device):
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


def _make_patched_load_checkpoint(original):
    def _patched_load_checkpoint(self, epoch: int, resume_training_state: bool = False):
        result = original(self, epoch, resume_training_state=resume_training_state)
        if resume_training_state:
            completed_epochs = int(epoch) + 1
            batches_per_epoch = int(getattr(self, "train_batches_per_epoch", 0))
            accum = max(int(getattr(self, "batches_per_update", 1)), 1)
            completed_forward_calls = completed_epochs * batches_per_epoch
            completed_updates = completed_forward_calls // accum
            self.grad_update_counter = completed_updates

            filt = getattr(self.policy, "fbc_filter_func", None)
            if filt is not None and hasattr(filt, "_seq_step"):
                # The 10k-entry percentile history is deliberately not serialized.
                # It repopulates for seq_warmup (~200) calls, but the long floor
                # warmup must not restart after every process failure.
                filt._seq_step = max(
                    int(getattr(filt, "_seq_step", 0)), completed_forward_calls
                )
        return result

    return _patched_load_checkpoint


def _install_patch() -> None:
    # ``python -m metamon.rl.taurosv1b_online`` executes the runner as __main__.
    # Avoid importing it here, which would execute/register the runner twice.
    for module_name in ("__main__", "metamon.rl.taurosv1b_online"):
        module = sys.modules.get(module_name)
        cls = getattr(module, "TaurosV1BOnlineExperiment", None) if module else None
        if cls is not None:
            cls.save_checkpoint = _patched_save_checkpoint
            if not getattr(cls, "_v1b_load_checkpoint_patched", False):
                cls.load_checkpoint = _make_patched_load_checkpoint(cls.load_checkpoint)
                cls._v1b_load_checkpoint_patched = True
            return


_install_patch()
