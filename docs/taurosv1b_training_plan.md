# TaurosV1B staged training plan

TaurosV1B keeps the TaurosV1/V1A `grouped_v2_medium.gin` architecture (~35M)
but avoids asking it to bootstrap from random weights through online RL.  TaurosV0
is a frozen teacher for policy-only distillation; its dense critics are never
stored or distilled.

## Fixed schedule

One learner epoch below is 1,000 minibatch updates.

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

The 1.925M-update total is deliberately close to the 2M updates in the V1A
recipe, but the first 325k updates are used to bootstrap policy/value competence
instead of spending the entire budget on cold-start RL.

## A: initial policy distillation

```bash
python -m metamon.rl.taurosv1b_pretrain \
  --phase a \
  --output_weights checkpoints/v1b_pretrain/phase_a.pt
```

Teacher: `TaurosV0@62`.  Student: random `TaurosV1A` architecture only.  Teacher
policy probabilities are produced on the fly; no teacher critic or policy label
sidecars are written.

Only gamma conditions shared with V0 are supervised.  The implementation matches
by numeric gamma value instead of assuming tensor positions are the same.

## B1: first student-occupancy round

Collect 75k complete battles with the phase-A student:

```bash
python -m metamon.rl.taurosv1b_collect_dagger \
  --weights checkpoints/v1b_pretrain/phase_a.pt \
  --output_dir buffer/v1b_dagger1 \
  --target_games 75000
```

Then distill another 50 epochs:

```bash
python -m metamon.rl.taurosv1b_pretrain \
  --phase b1 \
  --input_weights checkpoints/v1b_pretrain/phase_a.pt \
  --dagger1_dir buffer/v1b_dagger1 \
  --output_weights checkpoints/v1b_pretrain/phase_b1.pt
```

## B2: second student-occupancy round

```bash
python -m metamon.rl.taurosv1b_collect_dagger \
  --weights checkpoints/v1b_pretrain/phase_b1.pt \
  --output_dir buffer/v1b_dagger2 \
  --target_games 75000

python -m metamon.rl.taurosv1b_pretrain \
  --phase b2 \
  --input_weights checkpoints/v1b_pretrain/phase_b1.pt \
  --dagger1_dir buffer/v1b_dagger1 \
  --dagger2_dir buffer/v1b_dagger2 \
  --output_weights checkpoints/v1b_pretrain/phase_b2.pt
```

At this point run the external H2H matrix.  If the student is still roughly
100+ Elo below V0 despite low policy KL, stop and investigate capacity/history
mismatch rather than blindly proceeding.  The target is within ~20-50 Elo.

## C: critic-only warmup

```bash
python -m metamon.rl.taurosv1b_pretrain \
  --phase c \
  --input_weights checkpoints/v1b_pretrain/phase_b2.pt \
  --dagger1_dir buffer/v1b_dagger1 \
  --dagger2_dir buffer/v1b_dagger2 \
  --output_weights checkpoints/v1b_pretrain/phase_c.pt
```

The code explicitly calls `hard_sync_targets()` first, freezes the timestep
encoder, temporal encoder, and actor with `requires_grad_(False)`, and constructs
a critic-only optimizer.  Zeroing actor-loss coefficients alone is not used as a
substitute for freezing.

## D: shared-representation bridge

```bash
python -m metamon.rl.taurosv1b_pretrain \
  --phase d \
  --input_weights checkpoints/v1b_pretrain/phase_c.pt \
  --dagger1_dir buffer/v1b_dagger1 \
  --dagger2_dir buffer/v1b_dagger2 \
  --output_weights checkpoints/v1b_pretrain/phase_d.pt
```

The V0 KL coefficient is 1.0 for epochs 0-4, 0.5 for 5-14, and 0.25 for 15-24.
The ordinary advantage-weighted actor loss remains off.  Targets are hard-synced
again before the phase-D checkpoint is written.

## E: 800 epochs public-only online RL

Use the audited runner.  It fixes FIFO readiness, epoch-zero annealing, resume,
checkpoint-zero handling, sparse full-state checkpointing, and filter NaNs.

```bash
BASE_WEIGHTS=checkpoints/v1b_pretrain/phase_d.pt \
  bash scripts/train_taurosv1b.sh e --log
```

Data mix: online FIFO ramps from 20% to 60% during epochs 0-400 and remains 60%
for epochs 400-799.  The collector is prefilled beyond 5,000 battles before the
learner starts, and the dataset also enforces readiness internally.

Training uses normalized exponential AWR/CRR-style regression with beta=2.  It is
not described as importance sampling because no behavior-policy log-ratio is
injected in this run.

At the end of phase E, select a checkpoint with a balanced external H2H sweep
against at least TaurosV0@62, Kakuna@34, V2ADataAblation@90,
SyntheticRLV2@48, and Superkazam@50 on `modern_replays_v2` (and ideally a second
team distribution).  Start F only after at least V0 parity on the aggregate
matrix with no severe matchup regression.

## F: 800 epochs recency-weighted self-play

```bash
BASE_WEIGHTS=/path/to/selected_phase_e_policy.pt \
  bash scripts/train_taurosv1b.sh f --log
```

The online share stays at 60%.  Static public anchors carry about 65% of opponent
row mass; discovered V1B checkpoints carry about 35%, with `recency_rho=1.5` and
rolling `latest/policy.pt` included.  The AWR beta increases to 3 only in this
post-gate phase.

## Checkpoints and disk

Raw policy checkpoints are saved every 25 epochs; full Accelerate states are
saved every 100.  This avoids the old behavior of writing a full optimizer state
at every policy checkpoint.

Incremental storage beyond the already-downloaded public dataset is expected to
be roughly:

- DAgger-1 + DAgger-2 (150k compressed battles): ~1.5-3 GB;
- online FIFO (150k cap): ~1.5-3 GB, independent of E/F run length;
- policy/full-state checkpoints for E/F with the sparse cadence: roughly 10-20 GB;
- logs/indices/miscellaneous: a few GB;
- teacher labels / dense critics: **0 GB**.

Provision ~30-40 GB of free incremental disk if the public dataset is already
cached.  A fresh machine needs substantially more for dataset caches.

## Hardware

A 5090 is unnecessary.  The most memory-intensive stage is A/B/D, where the
50M V0 teacher actor/encoder and 35M trainable student coexist.  The pretrainer
drops the teacher critics/target critics after loading them.

Recommended target: **RTX 5070 Ti 16 GB**.  Start distillation at batch 8; try
larger batches only after profiling.  Student-only online RL can usually use a
larger batch.  The audited runner exposes `mixed_precision`, but the tracked
recipes stay at FP32 (`"no"`) until a BF16 smoke test confirms PopArt/two-hot
critic stability.

## Validation caveat

`val_timesteps` is environment timesteps per lane, not a requested number of
completed games and not a balanced matchup matrix.  It is a health check only;
explicit H2H sweeps should choose checkpoints.
