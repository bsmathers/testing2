"""Collect student-occupancy trajectories for TaurosV1B DAgger distillation.

The student plays the games.  TaurosV0 is *not* queried during collection; the
pretrainer labels these histories on the fly later, which keeps teacher-label
disk usage at zero.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser

import torch

from metamon.data import MetamonDataset
from metamon.rl import online_rl as legacy
from metamon.rl.metamon_to_amago import MetamonFIFODataset
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.taurosv1b_online import _create_experiment_with_safe_class


def _count_replays(root: str) -> int:
    fmt = os.path.join(root, "gen1ou")
    if not os.path.isdir(fmt):
        return 0
    return sum(
        1
        for name in os.listdir(fmt)
        if name.endswith(".json") or name.endswith(".json.lz4")
    )


def main():
    p = ArgumentParser(description="Collect V1B student-occupancy DAgger battles")
    p.add_argument("--weights", required=True, help="35M student policy state_dict")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_games", type=int, default=75_000)
    p.add_argument(
        "--train_pool",
        default="metamon/rl/configs/opponent_pools/hl_gen1ou_taurosv1b.yaml",
    )
    p.add_argument(
        "--train_team_set",
        default="metamon/rl/configs/team_sets/mediumg1_train.yaml",
    )
    p.add_argument("--lanes", type=int, default=128)
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--timesteps_per_round", type=int, default=500)
    p.add_argument("--temp_low", type=float, default=1.0)
    p.add_argument("--temp_high", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(args.weights)
    os.makedirs(os.path.join(args.output_dir, "gen1ou"), exist_ok=True)

    pretrained = get_pretrained_model("TaurosV1A")
    fifo_metamon = MetamonDataset(
        dset_root=os.path.abspath(args.output_dir),
        observation_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=pretrained.reward_function,
        formats=["gen1ou"],
        shuffle=True,
        verbose=False,
        write_index_cache=False,
    )
    fifo = MetamonFIFODataset(
        parsed_replay_dset=fifo_metamon,
        dset_max_size=max(args.target_games * 2, args.target_games + 10_000),
        dset_min_size=0,
        dset_name="V1B DAgger collector",
    )

    experiment = _create_experiment_with_safe_class(
        mixed_precision="no",
        full_state_interval=0,
        mode="collect",
        run_name="taurosv1b_dagger_collect",
        save_dir=os.path.abspath(os.path.join(args.output_dir, "_collector_state")),
        pretrained=pretrained,
        train_gin_config_path=pretrained.train_gin_config_path,
        amago_dataset=fifo,
        battle_format="gen1ou",
        reward_function=pretrained.reward_function,
        opponent_config_path=args.train_pool,
        val_opponent_kwargs={},
        buffer_dir=os.path.abspath(args.output_dir),
        save_results_to=None,
        lanes=args.lanes,
        n_workers=args.n_workers,
        train_team_set=args.train_team_set,
        val_team_set="modern_replays_v2",
        temp_low=args.temp_low,
        temp_high=args.temp_high,
        epochs=1,
        train_timesteps_per_epoch=args.timesteps_per_round,
        steps_per_epoch=0,
        batch_size_per_gpu=1,
        grad_accum=1,
        learning_rate=None,
        lr_warmup_epochs=0,
        seq_floor_warmup_epochs=0,
        val_timesteps=0,
        val_interval=1,
        ckpt_interval=1,
        dloader_workers=0,
        seed=args.seed,
        log=False,
    )
    experiment.start()
    experiment.load_checkpoint_from_path(args.weights, is_accelerate_state=False)

    round_idx = 0
    count = _count_replays(args.output_dir)
    print(f"Starting with {count:,}/{args.target_games:,} battles")
    while count < args.target_games:
        experiment.epoch = round_idx
        experiment.collect_new_training_data()
        fifo.on_end_of_collection(experiment)
        count = _count_replays(args.output_dir)
        round_idx += 1
        print(f"DAgger battles: {count:,}/{args.target_games:,}", flush=True)

    print(f"Collection complete: {count:,} battles in {args.output_dir}")


if __name__ == "__main__":
    main()
