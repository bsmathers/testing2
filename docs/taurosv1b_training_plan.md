# TaurosV1B staged training plan

TaurosV1B keeps the TaurosV1/V1A `grouped_v2_medium.gin` architecture (~35M)
but avoids asking it to bootstrap from random weights through online RL. TaurosV0
is a frozen teacher for policy-only distillation; its dense critics are never
stored or distilled.

## One-command run

The tracked end-to-end launcher runs every stage in order and is resumable:

```bash
bash scripts/train_taurosv1b_all.sh
```

Sequence:

```text
A -> DAgger-1 -> B1 -> DAgger-2 -> B2 -> C -> D -> E -> F
```

Useful paths can be moved to a fast NVMe drive:

```bash
V1B_WORK_DIR=/fast/v1b \
METAMON_SAVE_DIR=/fast/v1b/checkpoints \
METAMON_CACHE_DIR=/fast/metamon_cache \
METAMON_WANDB_PROJECT=taurosv1b \
bash scripts/train_taurosv1b_all.sh
```

The script skips completed A-D checkpoints, resumes partially collected DAgger
piles, and resumes E/F from the newest full optimizer state. A persistent local
experiment ID gives E and F stable W&B run IDs (`WANDB_RESUME=allow`) across
restarts.

## Fixed schedule

One learner epoch is 1,000 minibatch updates.

| Phase | Epochs | Updates | Training |
| --- | ---: | ---: | --- |
| A | 150 | 150k | V0 policy KL on public data |
| B1 | 50 | 50k | 75% public + 25% DAgger-1 |
| B2 | 50 | 50k | 50% public + 25% DAgger-1 + 25% DAgger-2 |
| C | 50 | 50k | critic only; actor + both encoders frozen |
| D | 25 | 25k | critic + V0 policy-KL bridge |
| E | **800** | **800k** | public-opponent online RL |
| F | **800** | **800k** | public anchors + recency-weighted V1B self-play |
| Total | **1,925** | **1.925M** | plus 150k DAgger battles |

The total learner-update budget is deliberately close to V1A's 2M updates, but
the first 325k establish a competent compact actor and critic instead of spending
the entire budget on cold-start RL.

## A/B: policy-only distillation

Teacher: `TaurosV0@62`. Student: random exact V1/V1A 35M architecture.
Teacher policy probabilities are generated on the fly. No teacher critic or
teacher-policy label sidecars are persisted.

Only gamma conditions shared with V0 are supervised. Gamma heads are matched by
numeric value rather than tensor position. The KL excludes padding,
missing-action states, and impossible actions.

Two DAgger rounds each collect 75,000 complete battles with the current student.
V0 does not play those games; it labels the resulting histories on the fly during
the next distillation stage.

Individual commands remain available through:

```bash
python -m metamon.rl.taurosv1b_pretrain --help
python -m metamon.rl.taurosv1b_collect_dagger --help
```

## C: critic-only warmup

Phase C calls `hard_sync_targets()` before the first TD update, explicitly freezes
the timestep encoder, temporal encoder and actor with `requires_grad_(False)`, and
constructs a critic-only optimizer. Merely zeroing the actor loss is not used as
a substitute for freezing.

The critic learns from real trajectory returns rather than stored V0 critic
predictions.

## D: shared-representation bridge

Everything is unfrozen. The ordinary AWR actor loss remains disabled while the
critic trains and the V0 policy KL anchors the actor/backbone. The KL coefficient
is 1.0 for bridge epochs 0-4, 0.5 for 5-14, and 0.25 for 15-24. Targets are hard
synced again before the phase-D checkpoint is written.

## E: 800 epochs public-only online RL

The E recipe uses the audited V1B runner, a beta-2 normalized exponential
AWR/CRR-style regression objective, LR `5e-5`, and collection temperature
`U(1.0, 2.25)`.

The FIFO is prefilled beyond 5,000 battles. Its training weight ramps from 20% to
60% over epochs 0-400 and remains 60% for epochs 400-799. The dataset itself also
enforces FIFO readiness, so an unready FIFO cannot be sampled.

## F: 800 epochs self-play refinement

The online share remains 60%. Static public anchors carry about 65% of opponent
row mass; discovered V1B checkpoints carry about 35%, with `recency_rho=1.5` and
rolling `latest/policy.pt` included. AWR beta is 3 and LR is `3e-5`.

Scientifically, phase E should be checked against the frozen external opponent
matrix before trusting F as an improvement experiment. The one-command launcher
runs all requested stages automatically; stop after E if you want to enforce a
manual aggregate gate before F.

## W&B and the five-epoch V0 tournament

E and F launch the learner with `--log`; the collector deliberately does not open
a second W&B run. Standard AMAGO metrics therefore appear normally in the phase
run.

At learner epochs 5, 10, 15, ... the checkpoint callback additionally pauses the
learner and evaluates the **current in-memory V1B policy** for exactly 50
single-lane games against `TaurosV0@62`, on `modern_replays_v2`, temperature 1.
It logs:

```text
tournament/v0_62_win_rate
tournament/v0_62_games
tournament/v0_62_binomial_stderr
```

The tournament does not write a temporary model checkpoint. Python, NumPy and
Torch CPU/CUDA RNG states and the model train/eval mode are restored afterwards,
so monitoring does not change subsequent learner randomness. Loading V0 clears
global gin state, so the tournament hook also explicitly restores the trainee gin
configuration before optimization resumes.

## Resume semantics

Raw policy checkpoints are retained every 25 epochs. Full Accelerate states are
retained every 100 epochs. The callback is invoked every 5 epochs only to run the
tournament.

On an E/F restart, `scripts/train_taurosv1b.sh`:

1. finds the newest full optimizer state;
2. finds the raw policy from that same epoch;
3. removes numbered policies newer than the optimizer state so phase-F
   `discover:true` cannot see abandoned "future" checkpoints;
4. rolls `latest/policy.pt` back to the exact resume policy before starting the
   collector;
5. resumes the learner at the following epoch.

The online FIFO is retained intentionally. E/F are off-policy replay methods, so
those trajectories remain valid behavior data even if the optimizer rolls back.

## Disk use

There are **zero persistent teacher critic labels** and **zero persistent teacher
policy labels**. The five-epoch tournaments also add zero model-checkpoint disk
use.

For the ~35M FP32 policy, a useful engineering estimate is ~140 MB per raw state
(`35M * 4 bytes`). With the tracked retention policy:

- E + F numbered raw policies: 64 files, ~9.0 GB;
- rolling `latest` for E/F plus A/B1/B2/C/D and E/F final copies: ~1.2 GB;
- 16 full states (epochs 0,100,...,700 in each online phase): at most roughly
  ~6.7 GB if AdamW held two FP32 moments for all 35M policy parameters; in practice
  it should be somewhat smaller because target-network parameters are not optimized;
- 150k DAgger battles: roughly ~1.5-3 GB;
- two independent 150k online FIFO caps (E and F): roughly ~3-6 GB total;
- TaurosV0 teacher checkpoint: ~0.25 GB;
- W&B files, indexes, logs and filesystem overhead: budget ~1-2 GB.

That puts the expected **incremental V1B working set, excluding the existing
public dataset cache, around 22-27 GB**. Treat **30 GB as tight and 40 GB as the
recommended free-space budget**.

A fresh machine additionally needs the public Gen1 data cache. V1B downloads only
Gen1OU from `pac-base`, `pac-exploratory`, and `pac-tauros`, plus Gen1OU human
replays, not the entire multi-format 220 GB parsed-pile repository. However the
self-play loader keeps decompressed `.tar` archives and Hugging Face may retain
cache blobs as well, so provision substantially more space on a cold machine.
**~200-250 GB free total is a conservative fresh-machine allocation; 40 GB is the
relevant incremental requirement when the V1A/public data cache already exists.**

Measure the actual machine after caches/checkpoints are created with:

```bash
du -sh "$METAMON_CACHE_DIR" "$V1B_WORK_DIR" "$METAMON_SAVE_DIR"
```

## Hardware

A 5090 is unnecessary. The most memory-intensive stage is policy distillation,
where the V0 teacher encoder/actor and trainable 35M student coexist. The
pretrainer drops teacher critics/target critics after loading them.

Recommended target: **RTX 5070 Ti 16 GB**. Distillation defaults to batch 8.
Student-only online RL can use a larger batch if profiling permits. The audited
runner exposes mixed precision, but the tracked recipes remain FP32 (`"no"`) until
a BF16 smoke test confirms PopArt/two-hot critic stability.

## Validation caveat

The ordinary `val_timesteps` setting is a health check, not a balanced matchup
matrix. Final checkpoint selection should still use explicit H2H sweeps across a
frozen opponent matrix. The every-five-epoch V0 tournament is a high-frequency
single-anchor diagnostic, not a replacement for that matrix.
