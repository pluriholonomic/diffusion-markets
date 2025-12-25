from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np


def conditional_mean_character_on_subcube(
    *,
    S: Sequence[int],
    J: Sequence[int],
    a: Sequence[int],
) -> int:
    """
    For a Walsh character chi_S(z)=∏_{i in S} z_i on {-1,+1}^d:
      E[chi_S(Z) | Z_J = a] is 0 unless S ⊆ J; if S ⊆ J it equals chi_S(a|_S).

    Here J is the set of fixed coordinates and a is the assignment on J in {-1,+1}^|J|.
    """
    S_set = set(int(i) for i in S)
    J_list = [int(i) for i in J]
    J_set = set(J_list)
    if not S_set.issubset(J_set):
        return 0

    a_map = {j: int(a[t]) for t, j in enumerate(J_list)}
    prod = 1
    for i in S_set:
        prod *= a_map[i]
    return int(prod)


def iter_subcube_J(d: int, k: int) -> Iterator[tuple[int, ...]]:
    yield from itertools.combinations(range(d), k)


def iter_assignments(k: int) -> Iterator[tuple[int, ...]]:
    # assignments in {-1,+1}^k
    for bits in itertools.product((-1, 1), repeat=k):
        yield tuple(int(b) for b in bits)


def keys_for_J(z: np.ndarray, J: Sequence[int]) -> np.ndarray:
    """
    Map each row of z[:,J] (values in {-1,+1}) to an integer key in [0, 2^k).
    Convention: +1 -> 1, -1 -> 0 for that bit position.
    """
    J = list(J)
    sub = z[:, J].astype(np.int8)
    bits01 = (sub == 1).astype(np.uint32)
    # key = Σ bits01[:,t] << t
    shifts = (1 << np.arange(len(J), dtype=np.uint32)).reshape(1, -1)
    return (bits01 * shifts).sum(axis=1).astype(np.uint32)


@dataclass(frozen=True)
class GroupDiagRow:
    J: tuple[int, ...]
    max_abs_cond_mean: float
    argmax_a_key: int


def worst_group_abs_cond_mean_over_assignments(
    *, z: np.ndarray, residual: np.ndarray, J: Sequence[int]
) -> tuple[float, int]:
    """
    For a fixed J, compute:
      max_{a in {-1,+1}^k} | E[ residual(Z) | Z_J = a ] |
    using a sample estimate.
    Returns (max_abs_mean, argmax_key) where argmax_key is the integer key of the maximizing a.
    """
    k = len(J)
    keys = keys_for_J(z, J)
    counts = np.bincount(keys, minlength=1 << k).astype(np.int64)
    sums = np.bincount(keys, weights=residual, minlength=1 << k).astype(np.float64)
    means = np.zeros_like(sums, dtype=np.float64)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]
    abs_means = np.abs(means)
    idx = int(abs_means.argmax())
    return float(abs_means[idx]), idx


def parity_S_equals_J_diagnostic(
    *, z: np.ndarray, residual: np.ndarray, S: Sequence[int], max_J: Optional[int] = None
) -> list[GroupDiagRow]:
    """
    Enumerate J of size |S| and compute the max absolute conditional mean over assignments.
    Intended to empirically validate the parity fact:
      only J == S yields non-zero conditional expectation (in the infinite-sample limit),
    which is the key step in the proof in main.tex.
    """
    d = z.shape[1]
    k = len(S)
    out: list[GroupDiagRow] = []
    for t, J in enumerate(iter_subcube_J(d, k)):
        if max_J is not None and t >= max_J:
            break
        m, a_key = worst_group_abs_cond_mean_over_assignments(z=z, residual=residual, J=J)
        out.append(GroupDiagRow(J=tuple(int(i) for i in J), max_abs_cond_mean=m, argmax_a_key=a_key))
    out.sort(key=lambda r: r.max_abs_cond_mean, reverse=True)
    return out


