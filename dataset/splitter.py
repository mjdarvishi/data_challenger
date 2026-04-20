from dataclasses import dataclass

import torch
from core.config import Config, SplitMode


@dataclass
class SplitResult:
    X_train: torch.Tensor
    Y_train: torch.Tensor
    X_val: torch.Tensor
    Y_val: torch.Tensor
    X_test: torch.Tensor
    Y_test: torch.Tensor


class TimeSeriesSplitter:
    def __init__(self):
        self.config = Config()
        self.train_ratio = self.config.train_ratio
        self.val_ratio = self.config.val_ratio
        self.split_mode = self.config.split_mode
        self.split_seed = self.config.split_seed

    def split(self, X: torch.Tensor, Y: torch.Tensor):
        return self.split_train_val_test(X, Y)

    def split_train_val_test(self, X: torch.Tensor, Y: torch.Tensor) -> SplitResult:
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")

        n = X.shape[0]
        if n < 3:
            raise ValueError("Need at least 3 samples for train/val/test split")

        if self.split_mode == SplitMode.CHRONOLOGICAL:
            idx_train, idx_val, idx_test = self._chronological_indices(n)
        elif self.split_mode == SplitMode.WEEKLY_BLOCK:
            idx_train, idx_val, idx_test = self._blockwise_random_indices(
                n=n,
                block_size=self.config.hours_per_week(),
            )
        else:
            raise ValueError(
                f"Unsupported split_mode={self.split_mode}. Use SplitMode enum values."
            )

        return SplitResult(
            X_train=X[idx_train],
            Y_train=Y[idx_train],
            X_val=X[idx_val],
            Y_val=Y[idx_val],
            X_test=X[idx_test],
            Y_test=Y[idx_test],
        )

    def _counts(self, n: int) -> tuple[int, int, int]:
        train_count = int(n * self.train_ratio)
        val_count = int(n * self.val_ratio)

        # Ensure each split has at least one sample and total stays n.
        train_count = max(1, min(train_count, n - 2))
        val_count = max(1, min(val_count, n - train_count - 1))
        test_count = n - train_count - val_count

        if test_count < 1:
            val_count = max(1, val_count - 1)
            test_count = n - train_count - val_count

        return train_count, val_count, test_count

    def _chronological_indices(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_count, val_count, _ = self._counts(n)

        train_end = train_count
        val_end = train_count + val_count

        idx_train = torch.arange(0, train_end)
        idx_val = torch.arange(train_end, val_end)
        idx_test = torch.arange(val_end, n)

        return idx_train, idx_val, idx_test

    def _blockwise_random_indices(self, n: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size = max(1, min(int(block_size), n))
        train_count, val_count, _ = self._counts(n)

        blocks = []
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            blocks.append(torch.arange(start, end))

        generator = torch.Generator()
        generator.manual_seed(self.split_seed)
        permuted = torch.randperm(len(blocks), generator=generator)
        shuffled_blocks = [blocks[i] for i in permuted.tolist()]

        # Fill splits by whole blocks where possible, preserving order within each block.
        idx_train_parts = []
        idx_val_parts = []
        idx_test_parts = []
        c_train = 0
        c_val = 0

        for block in shuffled_blocks:
            if c_train < train_count:
                idx_train_parts.append(block)
                c_train += block.numel()
            elif c_val < val_count:
                idx_val_parts.append(block)
                c_val += block.numel()
            else:
                idx_test_parts.append(block)

        idx_train = torch.cat(idx_train_parts) if idx_train_parts else torch.empty(0, dtype=torch.long)
        idx_val = torch.cat(idx_val_parts) if idx_val_parts else torch.empty(0, dtype=torch.long)
        idx_test = torch.cat(idx_test_parts) if idx_test_parts else torch.empty(0, dtype=torch.long)

        # Trim overshoot to target counts while keeping at least one sample in each split.
        idx_train = idx_train[:train_count]
        idx_val = idx_val[:val_count]
        remaining = n - idx_train.numel() - idx_val.numel()
        idx_test = idx_test[:remaining]

        # Safety fallback if block allocation underfilled due trimming edge cases.
        if idx_train.numel() + idx_val.numel() + idx_test.numel() < n:
            used = torch.cat([idx_train, idx_val, idx_test]) if (idx_train.numel() + idx_val.numel() + idx_test.numel()) > 0 else torch.empty(0, dtype=torch.long)
            mask = torch.ones(n, dtype=torch.bool)
            if used.numel() > 0:
                mask[used] = False
            extras = torch.arange(n)[mask]
            if extras.numel() > 0:
                idx_test = torch.cat([idx_test, extras])

        return idx_train, idx_val, idx_test
