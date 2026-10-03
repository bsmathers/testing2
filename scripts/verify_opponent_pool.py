#!/usr/bin/env python3
"""Verify the current opponent pool configuration and active agents for TaurosV1A."""

import os
import sys

# Ensure repo root is on PYTHONPATH
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from metamon.rl.evaluate.opponent_pool import load_opponent_pool


def main():
    config_path = os.path.join(
        repo_dir, "metamon/rl/configs/opponent_pools/hl_gen1ou_taurosv1a.yaml"
    )
    print(f"Loading opponent pool from: {config_path}\n")
    pool = load_opponent_pool(config_path, "gen1ou")

    if pool.discover_specs:
        base_name, merged, spec = pool.discover_specs[0]
        print(f"Self-Play Model:        {merged.get('model_name')}")
        print(f"Configured min_epoch:   {spec.get('min_epoch')}")
        print(f"Recency rho:            {spec.get('recency_rho')}")
        print(f"Total self agent slots: {spec.get('total_num_agents')}")
    else:
        print("No discover specs found.")

    pool._maybe_refresh_discovered(force=True)
    self_play_active = any("TaurosSelf" in name for name, _ in pool.agents)
    print(f"\nAre TaurosSelf checkpoints active in the pool right now?: {self_play_active}")
    print(f"Total active pool rows: {len(pool.agents)}")

    print("\nCurrent active pool roster:")
    for name, agent_spec in pool.agents:
        model = agent_spec.get("model_name")
        ckpts = agent_spec.get("checkpoints")
        print(f"  - {name:<28} (model: {model}, ckpts: {ckpts})")


if __name__ == "__main__":
    main()
