"""Rejection sampling to match a generated-essay sample to a reference
(human) word-count distribution, per prompt — mandatory per EXPERIMENT_5.md
("Length matching is mandatory... report the discard rate per cell").
"""
import numpy as np


def compute_quantile_bins(reference_values, n_bins: int) -> np.ndarray:
    return np.quantile(np.asarray(reference_values, dtype=float), np.linspace(0, 1, n_bins + 1))


def bin_targets(n_total: int, n_bins: int) -> list[int]:
    base, remainder = divmod(n_total, n_bins)
    return [base + 1 if i < remainder else base for i in range(n_bins)]


def assign_bin(value: float, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))


class RejectionSampler:
    """Accepts offered values into quantile bins until each bin's target is
    met, tracking how many offers were rejected along the way."""

    def __init__(self, edges: np.ndarray, targets: list[int]):
        self.edges = edges
        self.targets = targets
        self.counts = [0] * len(targets)
        self.n_offered = 0
        self.n_accepted = 0

    def offer(self, value: float) -> tuple[bool, int]:
        self.n_offered += 1
        b = assign_bin(value, self.edges)
        if self.counts[b] < self.targets[b]:
            self.counts[b] += 1
            self.n_accepted += 1
            return True, b
        return False, b

    def is_full(self) -> bool:
        return all(c >= t for c, t in zip(self.counts, self.targets))

    @property
    def discard_rate(self) -> float:
        if self.n_offered == 0:
            return 0.0
        return 1 - (self.n_accepted / self.n_offered)
