"""Generic discrete-time Markov-chain primitives used across the rating-
transition chapters (e.g. expected time to migrate into a default or
target-rating set).

Not ported from the MATLAB HSF toolbox -- no `.m` file in `hfs-archive`
implements this. It's a standard first-step-analysis result (solving the
linear system for expected hitting times), matching the Python formula
already used in this book's own notebooks (HSF-Notebooks chapter 2a).
"""

from __future__ import annotations

import numpy as np


def expected_hitting_time(p: np.ndarray, target_states: np.ndarray) -> np.ndarray:
    """Expected number of steps to reach any state in `target_states`
    (0-indexed) from each state of the time-homogeneous discrete-time
    Markov chain with transition matrix `p`. Entries for target states
    are 0 (already there).

    Solved via first-step analysis: for each non-target state `i`,
    ``h[i] = 1 + sum_j p[i,j] * h[j]`` restricted to the non-target
    states, i.e. ``(I - P_oo) @ h_o = 1`` where `P_oo` is `p` restricted
    to non-target rows/columns.
    """
    p = np.asarray(p, dtype=float)
    n = p.shape[0]
    target = np.asarray(target_states)
    other = np.setdiff1d(np.arange(n), target)

    h = np.zeros(n)
    if len(other) > 0:
        a = np.eye(len(other)) - p[np.ix_(other, other)]
        b = np.ones(len(other))
        h[other] = np.linalg.solve(a, b)
    return h
