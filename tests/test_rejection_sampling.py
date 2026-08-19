"""Unit tests for src.rejection_sampling — length-matching generated essays
to a reference (human) word-count distribution, per-prompt, for Experiment 5.
Mandatory per EXPERIMENT_5.md: "Match the PERSUADE word-count distribution
per prompt... report the discard rate per cell.\""""
import numpy as np
import pytest

from src.rejection_sampling import RejectionSampler, assign_bin, bin_targets, compute_quantile_bins


def test_compute_quantile_bins_returns_n_plus_one_edges():
    values = list(range(1, 101))  # 1..100
    edges = compute_quantile_bins(values, n_bins=4)
    assert len(edges) == 5
    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(100.0)


def test_bin_targets_distributes_remainder_to_first_bins():
    assert bin_targets(10, 4) == [3, 3, 2, 2]
    assert bin_targets(8, 4) == [2, 2, 2, 2]


def test_assign_bin_places_value_in_correct_quantile():
    edges = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    assert assign_bin(10, edges) == 0
    assert assign_bin(30, edges) == 1
    assert assign_bin(60, edges) == 2
    assert assign_bin(90, edges) == 3


def test_assign_bin_clips_out_of_range_values_to_edge_bins():
    edges = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    assert assign_bin(-5, edges) == 0
    assert assign_bin(500, edges) == 3


def test_rejection_sampler_accepts_until_bin_target_met_then_rejects():
    edges = np.array([0.0, 50.0, 100.0])
    sampler = RejectionSampler(edges, targets=[1, 1])
    accepted_1, bin_1 = sampler.offer(10)   # bin 0, first offer -> accept
    accepted_2, bin_2 = sampler.offer(20)   # bin 0, already full -> reject
    accepted_3, bin_3 = sampler.offer(80)   # bin 1, first offer -> accept
    assert (accepted_1, bin_1) == (True, 0)
    assert (accepted_2, bin_2) == (False, 0)
    assert (accepted_3, bin_3) == (True, 1)


def test_rejection_sampler_reports_full_only_when_all_bins_met():
    edges = np.array([0.0, 50.0, 100.0])
    sampler = RejectionSampler(edges, targets=[1, 1])
    assert not sampler.is_full()
    sampler.offer(10)
    assert not sampler.is_full()
    sampler.offer(80)
    assert sampler.is_full()


def test_rejection_sampler_tracks_discard_rate():
    edges = np.array([0.0, 50.0, 100.0])
    sampler = RejectionSampler(edges, targets=[1, 1])
    sampler.offer(10)  # accept
    sampler.offer(10)  # reject (bin full)
    sampler.offer(80)  # accept
    assert sampler.n_offered == 3
    assert sampler.n_accepted == 2
    assert sampler.discard_rate == pytest.approx(1 / 3)
