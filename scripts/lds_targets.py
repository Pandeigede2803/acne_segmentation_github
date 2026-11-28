#!/usr/bin/env python3
"""Prokhorov & Kalinin (ISBI 2024) label-distribution targets.

Faithful port of the reference implementation (openface-io/acne-lds):
- 65 lesion-count bins (0..64)
- Gaussian label distribution with sigma=3.0
- piecewise-linear smoothing weight eps(count) between eps_min and 1.0,
  equal to 1.0 at Hayashi class borders and eps_min mid-range
- count distribution projected to 13 evenly spaced severity bins and to the
  4 Hayashi severity bins

Pure Python on purpose: the notebook and CPU-only checks can import it without
numpy, and the training script converts the lists to tensors.
"""
from __future__ import annotations

import math

NUM_COUNTS = 65
DEFAULT_SIGMA = 3.0
DEFAULT_EPS_MIN = 0.6
DEFAULT_EPS_MAX = 1.0


def count_index(count: int, num_counts: int = NUM_COUNTS) -> int:
    """Map a lesion count (>=1) to a 0-based bin; the reference code uses count-1."""
    if num_counts < 1:
        raise ValueError("num_counts harus >= 1.")
    return min(max(int(count) - 1, 0), num_counts - 1)


def gaussian_count_distribution(index: int, sigma: float = DEFAULT_SIGMA,
                                num_counts: int = NUM_COUNTS) -> list[float]:
    """Gaussian over count bins, normalized to sum 1 (eq. 1-2 in the paper)."""
    if sigma <= 0:
        raise ValueError("sigma harus > 0.")
    if not 0 <= index < num_counts:
        raise ValueError(f"index harus 0..{num_counts - 1}.")
    weights = [math.exp(-((bin_index - index) ** 2) / (2.0 * sigma * sigma))
               for bin_index in range(num_counts)]
    total = sum(weights)
    return [value / total for value in weights]


def _line(values, slope: float, intercept: float) -> list[float]:
    return [slope * value + intercept for value in values]


def piecewise_epsilon(eps_min: float = DEFAULT_EPS_MIN, eps_max: float = DEFAULT_EPS_MAX,
                      num_counts: int = NUM_COUNTS) -> list[float]:
    """Replicate SmoothLDL.get_smooth_param(eps_type='piecewise') for 0..64."""
    if not 0 <= eps_min < eps_max <= 1:
        raise ValueError("Butuh 0 <= eps_min < eps_max <= 1.")
    if num_counts != NUM_COUNTS:
        raise ValueError("Jadwal piecewise hanya didefinisikan untuk 65 bin (0..64).")

    eps_1 = _line(range(0, 5), (eps_max - eps_min) / 4, eps_min)          # 0..4
    eps_2 = _line(range(0, 8), -(eps_1[-1] - eps_min) / 7, eps_1[-1])     # 5..12
    eps_3 = _line(range(1, 8), (eps_2[0] - eps_min) / 7, eps_2[-1])       # 13..19
    eps_4 = _line(range(0, 15), -(eps_3[-1] - eps_min) / 14, eps_3[-1])   # 20..34
    eps_5 = _line(range(0, 15), (eps_4[0] - eps_min) / 14, eps_4[-1])     # 35..49
    eps_6 = _line(range(0, 15), -(eps_5[-1] - eps_min) / 14, eps_5[-1])   # 50..64
    schedule = eps_1 + eps_2 + eps_3 + eps_4 + eps_5 + eps_6
    if len(schedule) != num_counts:
        raise AssertionError(f"Jadwal epsilon {len(schedule)} != {num_counts}.")
    return schedule


def smooth_count_distribution(index: int, sigma: float = DEFAULT_SIGMA,
                              eps_min: float = DEFAULT_EPS_MIN,
                              eps_max: float = DEFAULT_EPS_MAX,
                              num_counts: int = NUM_COUNTS) -> list[float]:
    """(1-eps)*one_hot + eps*Gaussian, eq. (4)."""
    gaussian = gaussian_count_distribution(index, sigma, num_counts)
    eps = piecewise_epsilon(eps_min, eps_max, num_counts)[index]
    return [(1.0 - eps) * (1.0 if bin_index == index else 0.0) + eps * value
            for bin_index, value in enumerate(gaussian)]


def severity_distribution_13(count_distribution: list[float]) -> list[float]:
    """Project 65 count bins to 13 bins of five counts (Prokhorov's cls branch)."""
    if len(count_distribution) != NUM_COUNTS:
        raise ValueError("Distribusi count harus 65 bin.")
    return [sum(count_distribution[start:start + 5]) for start in range(0, 65, 5)]


def severity_distribution_4(count_distribution: list[float]) -> list[float]:
    """Project 65 count bins to Hayashi: 0-4, 5-19, 20-49, 50+ (0-based bins)."""
    if len(count_distribution) != NUM_COUNTS:
        raise ValueError("Distribusi count harus 65 bin.")
    return [sum(count_distribution[:5]), sum(count_distribution[5:20]),
            sum(count_distribution[20:50]), sum(count_distribution[50:])]


def count_to_severity_index(count: int) -> int:
    """Hayashi bins: 1-5 -> 0, 6-20 -> 1, 21-50 -> 2, 51+ -> 3."""
    count = max(1, int(count))
    if count <= 5:
        return 0
    if count <= 20:
        return 1
    if count <= 50:
        return 2
    return 3


def targets_for_count(count: int, sigma: float = DEFAULT_SIGMA,
                      eps_min: float = DEFAULT_EPS_MIN) -> dict[str, list[float]]:
    index = count_index(count)
    smoothed = smooth_count_distribution(index, sigma, eps_min)
    plain = gaussian_count_distribution(index, sigma)
    return {
        "count_index": index,
        "count_distribution": smoothed,
        "count_distribution_plain": plain,
        "severity_13": severity_distribution_13(plain),
        "severity_4": severity_distribution_4(plain),
    }
