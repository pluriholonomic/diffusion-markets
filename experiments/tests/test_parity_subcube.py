import itertools

import numpy as np

from forecastbench.benchmarks.subcubes import conditional_mean_character_on_subcube, keys_for_J


def _full_cube(d: int) -> np.ndarray:
    grid = np.array(list(np.ndindex(*(2,) * d)), dtype=np.int8)
    return (2 * grid - 1).astype(np.int8)


def test_conditional_mean_character_on_subcube_matches_exact_enumeration():
    d = 8
    S = (0, 2, 5)
    k = len(S)
    z = _full_cube(d)
    chi = np.prod(z[:, S], axis=1).astype(np.int8)

    for J in itertools.combinations(range(d), k):
        keys = keys_for_J(z, J)
        for a_key in range(1 << k):
            # decode key to assignment a in {-1,+1}^k (bit=1 -> +1, bit=0 -> -1)
            a = tuple(1 if ((a_key >> t) & 1) else -1 for t in range(k))
            mask = keys == a_key
            assert mask.any()
            empirical = int(np.round(float(np.mean(chi[mask]))))  # exact should be -1,0,+1
            theo = conditional_mean_character_on_subcube(S=S, J=J, a=a)
            assert empirical == theo



