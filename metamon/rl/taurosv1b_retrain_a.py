"""Second TaurosV1B Phase-A pass: 150 epochs at fixed eta=1e-5.

This intentionally resets optimizer state.  It starts from the saved 150-epoch
Phase-A policy, keeps the same public-data KD objective, disables warmup, and
runs another 150 epochs with a constant 1e-5 learning rate.
"""

from metamon.rl import taurosv1b_pretrain as core
from metamon.rl import taurosv1b_pretrain_accel as accel


def main() -> None:
    core.PHASES["a"] = core.PhaseSpec(
        epochs=150,
        lr=1.0e-5,
        warmup_epochs=0,
        public_weight=1.0,
        dagger1_weight=0.0,
        dagger2_weight=0.0,
    )
    accel.main()


if __name__ == "__main__":
    main()
