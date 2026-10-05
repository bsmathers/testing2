"""Throughput/progress shim for TaurosV1B phases A-D.

This module deliberately does not reimplement the optimization.  It imports the
canonical :mod:`taurosv1b_pretrain` trainer, replaces only its DataLoader factory,
and then calls the original ``main()``.  The replacement loader:

* uses persistent multiprocessing workers and prefetching when workers > 0;
* pins host memory on CUDA, matching the original loader;
* exposes one live tqdm bar per learner epoch; and
* slices the iterator to exactly ``train_batches_per_epoch`` batches, avoiding the
  original loop's otherwise-unused (N+1)-th fetched batch.

All losses, optimizer/scheduler steps, target-network updates, and phase schedules
remain in ``taurosv1b_pretrain.py`` unchanged.
"""

from __future__ import annotations

import itertools

import torch
from amago.loading import RLData_pad_collate
from torch.utils.data import DataLoader
from tqdm import tqdm

from metamon.rl import taurosv1b_pretrain as core


class _EpochProgressLoader:
    def __init__(self, loader: DataLoader, steps_per_epoch: int):
        self.loader = loader
        self.steps_per_epoch = int(steps_per_epoch)
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        # islice prevents the outer trainer from requesting an unnecessary
        # (steps_per_epoch + 1)-th batch before its existing break condition.
        batches = itertools.islice(iter(self.loader), self.steps_per_epoch)
        return iter(
            tqdm(
                batches,
                total=self.steps_per_epoch,
                desc=f"epoch {self.epoch:03d}",
                unit="batch",
                dynamic_ncols=True,
                mininterval=0.5,
                smoothing=0.1,
            )
        )


def _fast_loader(dataset, batch_size: int, workers: int):
    workers = max(int(workers), 0)
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=RLData_pad_collate,
        pin_memory=torch.cuda.is_available(),
    )
    if workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
        )

    loader = DataLoader(**kwargs)
    experiment = getattr(dataset, "experiment", None)
    steps_per_epoch = getattr(experiment, "train_batches_per_epoch", None)
    if steps_per_epoch is None:
        raise RuntimeError(
            "TaurosV1B pretraining dataset was not configured with "
            "train_batches_per_epoch"
        )
    return _EpochProgressLoader(loader, steps_per_epoch)


def main() -> None:
    core._loader = _fast_loader
    core.main()


if __name__ == "__main__":
    main()
