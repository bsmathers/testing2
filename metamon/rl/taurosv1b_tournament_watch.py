"""Periodic TaurosV1B-vs-TaurosV0 tournament evaluation with W&B logging.

The online learner writes a numbered raw policy every ``eval_every`` epochs.
This watcher notices those immutable checkpoints, plays a 50-game H2H against
TaurosV0@62, logs V1B win rate to Weights & Biases, tracks/copies the best model,
and optionally deletes non-retention checkpoints after successful evaluation so
5-epoch evaluation does not multiply long-term checkpoint storage.

This is intentionally a separate process from the learner. It gives tournament
evaluation its own W&B run (same group as the learner) and avoids concurrent writes
to the learner's W&B run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import wandb

# Import registers the local ``TaurosV1B`` model used by the H2H subprocesses.
import metamon.rl.taurosv1b_online  # noqa: F401
from metamon.rl.evaluate.common import MatchupSpec, PolicySpec, run_matchup_pair
from metamon.rl.evaluate.results import ResultsTracker


_CKPT_RE = re.compile(r"^policy_epoch_(\d+)\.pt$")


def _available_epochs(policy_dir: Path, every: int, min_epoch: int, max_epoch: int):
    epochs = []
    if not policy_dir.is_dir():
        return epochs
    for p in policy_dir.iterdir():
        m = _CKPT_RE.match(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if min_epoch <= epoch <= max_epoch and epoch % every == 0:
            epochs.append(epoch)
    return sorted(epochs)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"completed_epochs": [], "best_epoch": None, "best_winrate": None}
    try:
        raw = json.loads(path.read_text())
        raw.setdefault("completed_epochs", [])
        raw.setdefault("best_epoch", None)
        raw.setdefault("best_winrate", None)
        return raw
    except Exception:
        return {"completed_epochs": [], "best_epoch": None, "best_winrate": None}


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _run_one(
    *,
    epoch: int,
    run_name: str,
    save_dir: Path,
    output_root: Path,
    gpu: int,
    games: int,
    team_set: str,
    timeout: int,
    startup_delay: float,
):
    # H2H subprocesses inherit these and therefore resolve TaurosV1B to this
    # exact local training run/checkpoint directory.
    os.environ["METAMON_SAVE_DIR"] = str(save_dir)
    os.environ["TAUROSV1B_RUN_NAME"] = run_name

    v1b = PolicySpec(
        name=f"TaurosV1B-{run_name}-e{epoch}",
        model_name="TaurosV1B",
        checkpoint=epoch,
        temperature=1.0,
        team_set=team_set,
        battle_backend="metamon",
    )
    v0 = PolicySpec(
        name="TaurosV0-e62",
        model_name="TaurosV0",
        checkpoint=62,
        temperature=1.0,
        team_set=team_set,
        battle_backend="metamon",
    )
    matchup = MatchupSpec(
        policy_a=v1b,
        policy_b=v0,
        n_battles=games,
        battle_format="gen1ou",
    )

    epoch_dir = output_root / f"epoch_{epoch:04d}"
    if epoch_dir.exists():
        shutil.rmtree(epoch_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    pair = run_matchup_pair(
        matchup=matchup,
        gpu_a=gpu,
        gpu_b=gpu,
        output_dir=str(epoch_dir),
        timeout=timeout,
        acceptor_startup_delay=startup_delay,
        verbose=False,
        save_trajectories=False,
    )
    tracker = ResultsTracker(str(epoch_dir))
    result = tracker.record_from_results_dir(
        matchup_id=matchup.matchup_id,
        policy_a_name=v1b.short_label,
        policy_b_name=v0.short_label,
        results_dir=os.path.join(pair.matchup_dir, "results"),
        challenger_username=pair.challenger_username,
    )
    if result is None:
        raise RuntimeError(f"No tournament result was produced for epoch {epoch}")
    if result.total_battles != games:
        raise RuntimeError(
            f"Tournament at epoch {epoch} produced {result.total_battles}/{games} battles; "
            "refusing to log a partial win rate."
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Watch V1B checkpoints and log V0 H2H win rates")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--phase", choices=["e", "f"], required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--retain_every", type=int, default=25)
    p.add_argument("--min_epoch", type=int, default=5)
    p.add_argument("--max_epoch", type=int, default=799)
    p.add_argument("--team_set", default="modern_replays_v2")
    p.add_argument("--poll_seconds", type=float, default=10.0)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--acceptor_startup_delay", type=float, default=10.0)
    p.add_argument("--learner_pid", type=int, default=None)
    p.add_argument("--wandb_project", default=os.environ.get("METAMON_WANDB_PROJECT", "online-metamon"))
    p.add_argument("--wandb_entity", default=os.environ.get("METAMON_WANDB_ENTITY"))
    args = p.parse_args()

    if args.eval_every <= 0 or args.games <= 0:
        raise ValueError("eval_every and games must be positive")
    if args.retain_every > 0 and args.retain_every % args.eval_every != 0:
        raise ValueError("retain_every must be a multiple of eval_every")

    save_dir = Path(args.save_dir).resolve()
    ckpt_root = save_dir / args.run_name / "ckpts"
    policy_dir = ckpt_root / "policy_weights"
    output_root = save_dir / args.run_name / "tournaments_vs_taurosv0_62"
    state_path = output_root / "state.json"
    best_path = output_root / "best_policy.pt"
    state = _load_state(state_path)
    completed = {int(x) for x in state["completed_epochs"]}

    phase_offset = 325 if args.phase == "e" else 1125
    wb = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.run_name}-tournaments",
        group=args.run_name,
        job_type="evaluation",
        id=f"{args.run_name}-tournaments",
        resume="allow",
        config={
            "opponent": "TaurosV0@62",
            "games_per_eval": args.games,
            "eval_every_epochs": args.eval_every,
            "team_set": args.team_set,
            "phase": args.phase.upper(),
        },
    )
    wandb.define_metric("training_epoch")
    wandb.define_metric("tournament/*", step_metric="training_epoch")

    try:
        while True:
            epochs = _available_epochs(
                policy_dir, args.eval_every, args.min_epoch, args.max_epoch
            )
            pending = [e for e in epochs if e not in completed]
            if pending:
                epoch = pending[0]
                ckpt = policy_dir / f"policy_epoch_{epoch}.pt"
                try:
                    result = _run_one(
                        epoch=epoch,
                        run_name=args.run_name,
                        save_dir=save_dir,
                        output_root=output_root,
                        gpu=args.gpu,
                        games=args.games,
                        team_set=args.team_set,
                        timeout=args.timeout,
                        startup_delay=args.acceptor_startup_delay,
                    )
                except Exception as exc:
                    print(f"Tournament epoch {epoch} failed: {exc}", flush=True)
                    time.sleep(args.poll_seconds)
                    continue

                winrate = result.policy_a_wins / result.total_battles
                global_epoch = phase_offset + epoch
                wandb.log(
                    {
                        "training_epoch": global_epoch,
                        "phase_epoch": epoch,
                        "tournament/winrate_vs_taurosv0_62": winrate,
                        "tournament/wins": result.policy_a_wins,
                        "tournament/losses": result.policy_b_wins,
                        "tournament/games": result.total_battles,
                    },
                    step=global_epoch,
                )
                print(
                    f"Tournament epoch {epoch}: {result.policy_a_wins}-{result.policy_b_wins} "
                    f"({winrate:.1%})",
                    flush=True,
                )

                best_wr = state.get("best_winrate")
                if best_wr is None or winrate > float(best_wr):
                    shutil.copy2(ckpt, best_path)
                    state["best_epoch"] = epoch
                    state["best_winrate"] = winrate
                    wandb.log(
                        {
                            "training_epoch": global_epoch,
                            "tournament/best_winrate_vs_taurosv0_62": winrate,
                            "tournament/best_phase_epoch": epoch,
                        },
                        step=global_epoch,
                    )

                completed.add(epoch)
                state["completed_epochs"] = sorted(completed)
                _write_state(state_path, state)

                # Evaluation requires 5-epoch numbered checkpoints, but long-term
                # training only needs the 25-epoch lineage plus rolling latest.
                # Delete the extra checkpoint only *after* a complete logged result
                # and (if it was best) after copying it to best_policy.pt.
                if args.retain_every > 0 and epoch % args.retain_every != 0:
                    try:
                        ckpt.unlink()
                    except FileNotFoundError:
                        pass
                continue

            learner_alive = True
            if args.learner_pid is not None:
                try:
                    os.kill(args.learner_pid, 0)
                except OSError:
                    learner_alive = False
            if not learner_alive:
                remaining = [
                    e
                    for e in _available_epochs(
                        policy_dir, args.eval_every, args.min_epoch, args.max_epoch
                    )
                    if e not in completed
                ]
                if not remaining:
                    break
            time.sleep(args.poll_seconds)
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
