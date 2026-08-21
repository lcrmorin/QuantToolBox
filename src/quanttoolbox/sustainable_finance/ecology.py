"""Species-area/endemics-area relationships, species-abundance
distributions, and Hurlbert's rarefaction estimator -- classic biodiversity
measures from ecology, used in this toolbox's biodiversity-risk chapters.

Ported from HSF toolbox `hsf/{species_area_relationship,
endemics_area_relationship,species_abundance_distribution,hurlbert}.m`.

Translation notes:

- `species_area_relationship`/`endemics_area_relationship` accept `a` as a
  scalar or array of target sub-areas, and can work either directly from
  per-species individual counts `n_i`, or from a pre-binned
  species-abundance histogram (`s_j` = number of species with exactly `j`
  individuals) -- typically the output of `species_abundance_distribution`.
  Pass `s_j=None` for the first mode.
- `species_abundance_distribution.m`'s three-way branch on its second
  argument (`brk`) is preserved as three explicit modes rather than
  MATLAB's type-based dispatch: `breaks=None` (unique-count histogram),
  `breaks="octave"` (Preston's log2 "octave" abundance classes), or
  `breaks=<array>` (custom breakpoints).
- `hurlbert.m`'s `method=2` (log-gamma) branch is numerically more stable
  than `method=1`'s direct `scipy.special.comb` ratio for large sample
  sizes (avoids overflow in the individual binomial coefficients); both
  compute the same quantity and are verified to agree in this module's
  tests.
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb, gammaln


def species_area_relationship(
    n_i: np.ndarray, s_j: np.ndarray | None, area_total: float, area: np.ndarray | float
) -> np.ndarray:
    """Expected number of species found in a sub-area `area` out of a total
    surveyed area `area_total`, given either per-species individual counts
    `n_i` (`s_j=None`), or a species-abundance histogram `s_j` (number of
    species with exactly `j` = 1, 2, ... individuals).

    Original: hsf/species_area_relationship.m
    """
    area = np.asarray(area, dtype=float)
    ratio = 1.0 - area / area_total

    if s_j is None:
        n_i = np.asarray(n_i, dtype=float)
        total_species = n_i.shape[0]
        term = np.sum(ratio[..., None] ** n_i, axis=-1)
        return total_species - term

    s_j = np.asarray(s_j, dtype=float)
    total_species = np.sum(s_j)
    j = np.arange(1, s_j.shape[0] + 1)
    term = np.sum(s_j * ratio[..., None] ** j, axis=-1)
    return total_species - term


def endemics_area_relationship(
    n_i: np.ndarray, s_j: np.ndarray | None, area_total: float, area: np.ndarray | float
) -> np.ndarray:
    """Expected number of species confined entirely within a sub-area
    `area` out of a total surveyed area `area_total` ("endemics"), given
    either per-species individual counts `n_i` (`s_j=None`), or a
    species-abundance histogram `s_j`.

    Original: hsf/endemics_area_relationship.m
    """
    area = np.asarray(area, dtype=float)
    ratio = area / area_total

    if s_j is None:
        n_i = np.asarray(n_i, dtype=float)
        return np.sum(ratio[..., None] ** n_i, axis=-1)

    s_j = np.asarray(s_j, dtype=float)
    j = np.arange(1, s_j.shape[0] + 1)
    return np.sum(s_j * ratio[..., None] ** j, axis=-1)


def species_abundance_distribution(
    n_i: np.ndarray, breaks: np.ndarray | str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Bin per-species individual counts `n_i` into a species-abundance
    histogram: `s[k]` species fall into abundance class `j[k]`.

    - `breaks=None` (default): one class per distinct count value in `n_i`.
    - `breaks="octave"`: Preston's log2 "octave" classes (1; 2-3; 4-7;
      8-15; ...).
    - `breaks=<array>`: custom upper breakpoints; `j[k]` is the midpoint of
      class k's range.

    Returns `(j, s, breaks)` -- `breaks` is `None` for the default mode.

    Original: hsf/species_abundance_distribution.m
    """
    n_i = np.asarray(n_i, dtype=float)

    if breaks is None:
        j = np.unique(n_i)
        s = np.array([np.sum(n_i == jj) for jj in j], dtype=float)
        return j, s, None

    if isinstance(breaks, str):
        if breaks.lower() != "octave":
            raise ValueError("breaks must be None, 'octave', or an array of breakpoints")
        n_max = np.max(n_i)
        k_max = int(np.ceil(np.log2(max(n_max, 1)))) or 1
        k = np.arange(1, k_max + 1)
        octave_breaks = 2.0**k - 1.0
        s = np.zeros(k_max)
        for it in range(k_max):
            if it == 0:
                cond = n_i <= octave_breaks[it]
            else:
                cond = (n_i > octave_breaks[it - 1]) & (n_i <= octave_breaks[it])
            s[it] = np.sum(cond)
        return k.astype(float), s, octave_breaks

    breaks = np.asarray(breaks, dtype=float)
    n_classes = breaks.shape[0]
    j = np.zeros(n_classes)
    s = np.zeros(n_classes)
    for it in range(n_classes):
        if it == 0:
            cond = n_i <= breaks[it]
            j[it] = 0.5 * (1.0 + breaks[it])
        else:
            cond = (n_i > breaks[it - 1]) & (n_i <= breaks[it])
            j[it] = 0.5 * (breaks[it - 1] + breaks[it])
        s[it] = np.sum(cond)
    return j, s, breaks


def hurlbert(n_i: np.ndarray, m: int, method: int = 1) -> float:
    """Hurlbert's rarefaction estimator: expected number of species present
    in a random sample of `m` individuals drawn (without replacement) from
    a community with per-species counts `n_i`.

    `method=2` uses a log-gamma formulation (via `scipy.special.gammaln`)
    for numerical stability at large sample sizes; `method=1` (default)
    computes the binomial-coefficient ratio directly.

    Original: hsf/hurlbert.m
    """
    n_i = np.asarray(n_i, dtype=float)
    n = np.sum(n_i)

    sac = 0.0
    for n_s in n_i:
        if (n - n_s) >= m:
            if method == 2:
                log_q = (
                    gammaln(n - n_s + 1)
                    + gammaln(n - m + 1)
                    - (gammaln(n + 1) + gammaln(n - n_s - m + 1))
                )
                q = np.exp(log_q)
            else:
                q = comb(n - n_s, m) / comb(n, m)
            sac += 1.0 - q
        else:
            sac += 1.0

    return float(sac)
