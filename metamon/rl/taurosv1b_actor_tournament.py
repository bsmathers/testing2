"""Evaluate a saved TaurosV1A/V1B actor against TaurosV0@62 and persist results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import torch

from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.taurosv1b_tournament import (
    TOURNAMENT_GAMES,
    TOURNAMENT_TEAM_SET,
    TOURNAMENT_V0_CHECKPOINT,
    _run_v0_tournament,
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _matching_result(path: str, weights_sha256: str, games: int) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        old.get("weights_sha256") == weights_sha256
        and int(old.get("games", -1)) == games
        and int(old.get("opponent_checkpoint", -1)) == TOURNAMENT_V0_CHECKPOINT
        and old.get("team_set") == TOURNAMENT_TEAM_SET
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--output", required=True)
    p.add_argument("--seed_epoch_base", type=int, default=20_000)
    args = p.parse_args()

    if args.games <= 0 or args.games % TOURNAMENT_GAMES != 0:
        raise ValueError(
            f"--games must be a positive multiple of {TOURNAMENT_GAMES}; "
            f"got {args.games}"
        )
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(args.weights)

    weights = os.path.abspath(args.weights)
    digest = _sha256(weights)
    if _matching_result(args.output, digest, args.games):
        print(f"[skip] matching tournament result already exists: {args.output}")
        with open(args.output, "r", encoding="utf-8") as f:
            old = json.load(f)
        print(
            f"TaurosV1B vs TaurosV0@{TOURNAMENT_V0_CHECKPOINT}: "
            f"{old['wins']}/{old['games']} = {old['win_rate']:.2%}"
        )
        return

    spec = get_pretrained_model("TaurosV1A")
    exp = spec.initialize_agent(checkpoint=0, log=False)
    state = torch.load(weights, map_location="cpu")
    exp.policy.load_state_dict(state, strict=True)
    exp.policy.on_checkpoint_loaded(is_resume=False)
    exp.policy.eval()

    total_wins = 0
    rounds = args.games // TOURNAMENT_GAMES
    round_results = []
    for i in range(rounds):
        exp.epoch = args.seed_epoch_base + i
        wr = _run_v0_tournament(exp)
        wins = int(round(wr * TOURNAMENT_GAMES))
        total_wins += wins
        round_results.append(
            {
                "round": i + 1,
                "seed_epoch": int(exp.epoch),
                "games": TOURNAMENT_GAMES,
                "wins": wins,
                "win_rate": wr,
            }
        )
        played = (i + 1) * TOURNAMENT_GAMES
        print(
            f"[{i + 1}/{rounds}] round={wins}/{TOURNAMENT_GAMES} "
            f"({wr:.1%}) cumulative={total_wins}/{played} "
            f"({total_wins / played:.2%})",
            flush=True,
        )

    p_hat = total_wins / args.games
    se = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / args.games)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": weights,
        "weights_sha256": digest,
        "opponent": "TaurosV0",
        "opponent_checkpoint": TOURNAMENT_V0_CHECKPOINT,
        "team_set": TOURNAMENT_TEAM_SET,
        "temperature": 1.0,
        "games": args.games,
        "wins": total_wins,
        "losses": args.games - total_wins,
        "win_rate": p_hat,
        "binomial_stderr": se,
        "normal_95_ci": [p_hat - 1.96 * se, p_hat + 1.96 * se],
        "rounds": round_results,
    }
    _atomic_json(args.output, payload)

    print("\n======================================")
    print(f"Actor vs TaurosV0@{TOURNAMENT_V0_CHECKPOINT}")
    print(f"Record:   {total_wins}-{args.games - total_wins}")
    print(f"Win rate: {p_hat:.2%}")
    print(f"SE:       {se:.2%}")
    print(f"95% CI:   [{p_hat - 1.96 * se:.2%}, {p_hat + 1.96 * se:.2%}]")
    print(f"Saved:    {os.path.abspath(args.output)}")
    print("======================================")


if __name__ == "__main__":
    main()
