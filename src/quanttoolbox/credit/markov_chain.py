"""Expected hitting times for discrete-time Markov chains.

Adapted from HSF-Notebooks chapter 02a. The implementation solves the
standard first-step linear system for the expected time to reach a target
rating set; it is not a MATLAB toolbox port.
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
