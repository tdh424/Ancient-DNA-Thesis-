"""
length_bucketed_sampler.py — Length-bucketed, class-balanced batch sampler.

Motivation:
  - All three simulated sources (bacterial, human, environmental) use the
    same fragment length distribution (Section 3.2.1 of the thesis), so length
    no longer serves as a discriminative shortcut for the classifier. However,
    short and long reads still differ in how much padding the model sees, and
    purely random shuffle batches mix very short (~30 bp) and very long
    (~100 bp) reads together, wasting compute on padded positions.

  - To use compute more efficiently AND keep each mini-batch a fair
    representation of the population in the class balance sense, we group
    reads by length into a small number of buckets, and each mini-batch is
    drawn from one bucket while preserving the dataset's class prevalence
    (e.g. 25 % ancient + 75 % modern for the classifier).

Design:
  1. Sort reads into `n_buckets` length quantiles (so each bucket has roughly
     the same number of reads). The bucket boundaries are derived from the
     observed length distribution at construction time, not hard-coded.
  2. Within each bucket, split indices by class.
  3. At each iteration:
       a. pick a bucket at random (weighted by bucket size, so the natural
          length distribution is preserved across batches);
       b. draw `n_anc` ancient indices and `n_mod` modern indices from that
          bucket, with replacement if necessary (small buckets that run out
          of ancient examples reuse them);
       c. yield the combined batch.
  4. One epoch = `N / batch_size` batches.

Why balanced but not flat in a distribution:
  - Balanced  : every batch has a fixed fraction of ancient examples
                (parameter `ancient_frac`), regardless of where in the length
                distribution the bucket sits.
  - Not flat  : within a batch the read lengths follow the natural variation
                inside the bucket (a few-bp window), not an artificial uniform
                distribution across the full 30–100 bp range.

The sampler is a drop-in replacement for `shuffle=True` in a DataLoader.
For validation / test splits, use the standard sequential loader.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler


class LengthBucketedBatchSampler(Sampler[list[int]]):
    """Yield batches of indices grouped by similar read length.

    Args:
        lengths        : 1-D array of read lengths, one per example.
        labels         : 1-D binary array (0 = modern, 1 = ancient). Used for
                         the class balance within a batch.
        batch_size     : Total number of indices per batch.
        n_buckets      : Number of length quantile buckets (default: 10).
        ancient_frac   : Target fraction of ancient examples per batch.
                         Default: None → match the natural prevalence of
                         `labels`. Set to 0.5 for fully balanced batches.
        drop_last      : Whether to drop the final partial pass over buckets.
                         Default: False, since we sample with replacement.
        seed           : Base RNG seed; the epoch counter is mixed in so
                         each epoch sees a different ordering.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        n_buckets: int = 10,
        ancient_frac: float | None = None,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        lengths = np.asarray(lengths).reshape(-1)
        labels  = np.asarray(labels).reshape(-1)
        if lengths.shape != labels.shape:
            raise ValueError(
                f'lengths and labels shape mismatch: '
                f'{lengths.shape} vs {labels.shape}'
            )

        self.N           = len(lengths)
        self.batch_size  = int(batch_size)
        self.n_buckets   = int(n_buckets)
        self.drop_last   = bool(drop_last)
        self.seed        = int(seed)
        self._epoch      = 0

        natural_anc = float(labels.mean()) if self.N > 0 else 0.0
        self.ancient_frac = float(ancient_frac) if ancient_frac is not None \
            else natural_anc

        # Bucket boundaries from length quantiles. Use `np.quantile` rather
        # than fixed bp ranges so buckets adapt to whatever distribution the
        # current dataset has.
        quantiles = np.linspace(0.0, 1.0, n_buckets + 1)
        edges = np.quantile(lengths, quantiles)
        # `np.digitize` returns 1..n; subtract 1 and clip so we get 0..n-1.
        bucket_id = np.clip(
            np.digitize(lengths, edges[1:-1], right=False), 0, n_buckets - 1
        )

        self.bucket_anc: list[np.ndarray] = []
        self.bucket_mod: list[np.ndarray] = []
        for b in range(n_buckets):
            mask = bucket_id == b
            self.bucket_anc.append(np.where(mask & (labels == 1))[0])
            self.bucket_mod.append(np.where(mask & (labels == 0))[0])

        bucket_sizes = np.array(
            [len(a) + len(m) for a, m in zip(self.bucket_anc, self.bucket_mod)]
        )
        # Bucket selection weights: proportional to bucket size, so the
        # epoch-level length distribution matches the data, not flat.
        self.bucket_weights = bucket_sizes / max(bucket_sizes.sum(), 1)

        # Per-batch class counts (deterministic, balanced as requested)
        self.n_anc = max(1, int(round(self.batch_size * self.ancient_frac)))
        self.n_mod = self.batch_size - self.n_anc
        if self.n_anc < 0 or self.n_mod < 0:
            raise ValueError('batch_size too small for the requested split.')

        # Sanity check: every bucket should have at least one read in each
        # class, otherwise we will be sampling the same one repeatedly.
        empty = []
        for b in range(n_buckets):
            if (self.n_anc > 0 and len(self.bucket_anc[b]) == 0) or \
               (self.n_mod > 0 and len(self.bucket_mod[b]) == 0):
                empty.append(b)
        if empty:
            print(f'[LengthBucketedBatchSampler] warning: bucket(s) {empty} '
                  f'are missing one class; samples will be reused.')

    # ---- protocol -------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Set epoch counter so the RNG advances between epochs."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        # Total batches per epoch — one full pass over the dataset.
        n_batches = self.N // self.batch_size
        return n_batches if self.drop_last else max(1, n_batches)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch * 100003)
        n_batches = len(self)
        # Pre-pick the bucket for each batch in this epoch (weighted by size).
        bucket_picks = rng.choice(
            self.n_buckets, size=n_batches, replace=True, p=self.bucket_weights
        )
        for b in bucket_picks:
            anc_pool = self.bucket_anc[b]
            mod_pool = self.bucket_mod[b]

            if self.n_anc > 0:
                if len(anc_pool) == 0:
                    # Fallback to a global pool if a bucket has no ancient
                    # examples (rare with default quantile bucketing).
                    anc_pool = np.concatenate(
                        [p for p in self.bucket_anc if len(p) > 0]
                    )
                anc_idx = rng.choice(
                    anc_pool, size=self.n_anc,
                    replace=len(anc_pool) < self.n_anc,
                )
            else:
                anc_idx = np.empty(0, dtype=np.int64)

            if self.n_mod > 0:
                if len(mod_pool) == 0:
                    mod_pool = np.concatenate(
                        [p for p in self.bucket_mod if len(p) > 0]
                    )
                mod_idx = rng.choice(
                    mod_pool, size=self.n_mod,
                    replace=len(mod_pool) < self.n_mod,
                )
            else:
                mod_idx = np.empty(0, dtype=np.int64)

            batch = np.concatenate([anc_idx, mod_idx])
            rng.shuffle(batch)
            yield batch.tolist()


def build_loader(
    dataset,
    lengths,
    labels,
    batch_size,
    *,
    shuffle: bool = True,
    n_buckets: int = 10,
    ancient_frac: float | None = None,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """Convenience wrapper that builds a `DataLoader` with the bucketed
    sampler when `shuffle=True`, and a plain sequential loader otherwise.

    Use this in place of `torch.utils.data.DataLoader(...)` in the training
    scripts to enable the new sampling regime.
    """
    if shuffle:
        sampler = LengthBucketedBatchSampler(
            lengths=lengths,
            labels=labels,
            batch_size=batch_size,
            n_buckets=n_buckets,
            ancient_frac=ancient_frac,
            seed=seed,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
