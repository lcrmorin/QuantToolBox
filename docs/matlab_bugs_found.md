# Bugs found in the original MATLAB source

While porting QuantToolbox to Python, every numeric routine was tested
against an independent source of truth where one was available: known
analytical solutions, Monte Carlo simulation, algebraic identities that
must hold exactly, or agreement with established libraries (scipy,
statsmodels, scikit-learn). That process surfaced several genuine defects
in the **original MATLAB code** — not porting mistakes, but bugs that
were already present in the source being translated. This document
records each one: what it is, how it was found, what its practical impact
would have been, and how the Python port handles it.

A separate, much longer list of *porting* bugs (mistakes introduced while
translating and then caught by testing) is not reproduced here — see the
"Notes for translators" section of `docs/migration_map.md` for those.
This file is specifically about defects that predate the port.

---

## 1. Whittle estimation: analytical Jacobian is missing a `1/(2π)` factor

**Files:** `ects/whittle_local_level.m` (`local_level_sdf_jacobian`),
`ects/whittle_local_linear_trend.m` (`local_linear_trend_sdf_jacobian`)

**The bug:** `local_level_sdf.m` defines the spectral density as

```
sdf = (2*(1-cos(lambda))*sigma1^2 + sigma2^2) / (2*pi)
```

i.e. the raw quadratic form scaled by `1/(2π)`. But its companion
Jacobian function differentiates the *unscaled* quadratic form only:

```
J1 = 4*(1-cos(lambda))*sigma1
J2 = 2*sigma2
```

The true derivative of `sdf` with respect to `sigma1`/`sigma2` is `J1/(2π)`
and `J2/(2π)` — the Jacobian function silently omits the `1/(2π)` factor
that the function it's supposed to be differentiating actually has. The
same inconsistency is repeated in the local-linear-trend variant.

**How it was found:** the Python port's Whittle MLE, using the analytical
gradient exactly as translated from MATLAB, converged to noticeably
worse-fitting parameter estimates than the same optimization using a
plain numerical (finite-difference) gradient — a strong signal that the
analytical gradient was wrong. Directly comparing the two gradients at
the same point confirmed it: they differed by a factor of almost exactly
`2π` (6.283...) at every point checked, which is exactly what an omitted
`1/(2π)` scaling factor would produce.

**Practical impact:** any code path in the original MATLAB toolbox that
supplied the analytical Jacobian to `fminunc`/`fmincon` for these two
models (the `nb_functions == 2` branch in
`whittle_constrained_estimation.m`) would converge to the correct
optimum eventually regardless, since a Newton-type optimizer will still
find the same stationary point as long as the gradient's zero-crossing is
correct — but convergence would be slower and less numerically stable
than intended, since the reported gradient magnitude is off by a factor
of ~6.28, which throws off step-size heuristics and convergence-tolerance
checks. It would also make the "method of scoring" branch (which uses
the gradient directly in a Newton step, not just its sign) converge to a
step of the wrong size on each iteration.

**Fix in the Python port:** `quanttoolbox.econometrics.whittle`'s
`_local_level_sdf_jacobian` and `_local_linear_trend_sdf_jacobian`
include the missing `1/(2*pi)` factor, with the derivation documented
inline. Verified with a regression test
(`test_whittle_local_level_analytical_gradient_matches_numerical`) that
directly asserts the analytical and numerical gradients agree to within
`1e-4`, so this can't silently regress.

---

## 2. `simulate_multi_gbm.m` ignores its own correlation parameter

**File:** `maths/simulate_multi_gbm.m`

**The bug:** the function signature is
`simulate_multi_gbm(x0, mu, sigma, rho, t, nS)`, explicitly taking a
correlation parameter `rho` — but the function body is byte-identical to
`simulate_gbm.m` (the *single-asset*, uncorrelated simulator). `rho` is
accepted as an argument and then never referenced anywhere in the
function body.

**How it was found:** while reading the source ahead of porting, simply
diffing `simulate_multi_gbm.m` against `simulate_gbm.m` showed they were
character-for-character identical apart from the function name — an
unusual thing to happen by coincidence, and a giveaway that this was an
abandoned or unfinished implementation rather than an intentional
simplification.

**Practical impact:** any caller of `simulate_multi_gbm` in the original
codebase expecting correlated multi-asset paths would silently get
**independent** single-asset-style paths instead (broadcast across
whatever the actual multi-dimensional inputs were, ignoring cross-asset
correlation entirely). This is the kind of bug that's easy to miss
downstream, since the output still "looks like" a plausible simulation —
it just has the wrong joint distribution.

**Fix in the Python port:**
`quanttoolbox.maths.simulation.simulate_multi_gbm` is a genuine
N-asset correlated simulator, implemented via Cholesky decomposition of
the correlation matrix (the mathematically standard approach, and what
the original's signature/docstring evidently intended). This is a
deliberate *departure* from faithfully reproducing the original's
behavior, documented explicitly in the module docstring, since faithfully
reproducing a bug that discards the function's one documented purpose
would be pointless. Verified with a test that simulates a 3-asset system
and checks the empirical correlation matrix of the simulated returns
matches the target correlation matrix to within Monte Carlo sampling
error.

---

## 3. Newton's method has no positivity floor, and can diverge

**Files:** `crb/compute_rb_sd_admm_newton.m`,
`rpb/compute_rb_sd_newton.m` (the same Newton core is used, with minor
variations, in both the unconstrained and ADMM-embedded risk-budgeting
solvers)

**The bug:** the risk-budgeting Lagrangian's gradient and Hessian both
contain a `b ./ x` term (and `b ./ (x.*x)` in the Hessian), which is only
mathematically valid for `x > 0` — weights are supposed to stay strictly
positive near the solution. The only safeguard the original code has is
`RB_Newton_Correction`, which — when enabled — does
`x = min(x, RB_Newton_xMax)`, an **upper** cap only. Nothing in the
original prevents a Newton step from driving `x` negative, at which point
`b ./ x` and `b ./ (x.*x)` become numerically unstable (large, sign-flipped,
or in degenerate cases divide-by-zero), and the iteration can diverge
rather than converge.

**How it was found:** while testing the Python port's risk-budgeting
target-matching frontier (which evaluates the box-constrained solver at
extreme risk-aversion values as part of finding the achievable target
range), a call with a high risk-aversion parameter hung for minutes
instead of returning. Tracing it down: Newton's method was diverging to
large negative weight values, and since nothing detected the divergence,
both the inner ADMM loop and the outer bisection search kept running to
their full iteration budgets on a broken, non-converging state — up to
tens of millions of wasted Newton steps in the worst case.

**Practical impact:** in the original MATLAB toolbox, this would manifest
as the risk-budgeting solver silently failing to converge (or converging
to a nonsensical negative-weight "solution") for sufficiently extreme
risk-aversion parameters or unusual covariance structures, rather than
detecting the problem and reporting it. Since the failure mode is
divergence rather than an outright crash, it's the kind of thing that
could produce a bad optimization result without an obvious error message
— particularly risky in a portfolio-construction context.

**Fix in the Python port:** every Newton iteration in
`quanttoolbox.portfolio.risk_budgeting._solve_newton` clips `x` to a
small positive floor (`x = max(x, 1e-8)`) before evaluating the gradient
and Hessian, which is a minimal, standard safeguard for exactly this kind
of update rule. Verified directly: the specific input that previously
hung now completes in about a second and converges to a sensible,
strictly positive solution; a full run of the existing test suite (244
tests at the time) showed zero regressions from adding the floor.

---

## 4. Dead code path: `RB_lagrangian` global is never assigned before use

**Files:** `mixture/mixture_compute_rb_var.m`,
`mixture/mixture_compute_rb_es.m` (the `RB_algorithm != 1` / "algorithm
2" branch in each)

**The bug:** both functions offer two solver algorithms, selected by the
`RB_algorithm` global: algorithm 1 uses `fmincon` to directly minimize
risk-contribution deviations, while algorithm 2 uses `fminunc` on a
log-barrier-penalized objective that references `global RB_lagrangian`
inside its objective/gradient function
(`local_mixture_compute_rb_var2`/`local_mixture_compute_rb_es2`).
Searching the entire codebase, `RB_lagrangian` is declared as a global in
several places but **never assigned a value anywhere** before algorithm 2
would use it. Calling algorithm 2 would use whatever stale or default
value MATLAB gives an unset global (typically empty), which would error
or silently produce nonsense inside the log-barrier term
`lagrangian * sum(b .* log(x))`.

**How it was found:** while reading through the mixture risk-budgeting
code to plan the Python port, tracing every assignment site of
`RB_lagrangian` across the codebase turned up none — a `grep`-style
search for `RB_lagrangian =` (as opposed to `global RB_lagrangian` or
uses of the variable) returns no results anywhere in the source tree.

**Practical impact:** the `RB_algorithm == 1` branch (the default) works
fine and is presumably what the original toolbox's users actually
exercised; the `RB_algorithm != 1` branch appears to be genuinely
unreachable/broken code that was never completed or never tested.

**Fix in the Python port:**
`quanttoolbox.mixtures.gaussian_mixture.mixture_compute_rb_var`/
`mixture_compute_rb_es` implement only the working algorithm (the
`fmincon`-equivalent minimization via `scipy.optimize.minimize` with
SLSQP), matching the original's functional default. The broken algorithm-2
branch is not ported, and this is called out explicitly in the module
docstring so it's clear the omission is intentional rather than an
oversight.

---

## 5. Corner-case false-early-termination in the Dykstra projection loop

**File:** `optim/proximal_linear_constraints.m` (and the equivalent
pattern in `proximal_inequality.m`, `proximal_turnover.m`)

**The bug:** the loop's convergence check is exact floating-point
equality: `if x1 == x4 break end`. Dykstra's alternating-projection
algorithm can, for constraint sets that meet at a sharp corner (e.g. a
budget constraint intersecting a box constraint at a vertex), pass
through a **temporary plateau** — several consecutive iterations where
the iterate doesn't move at all in floating-point terms — before
continuing to converge toward the true intersection point. An exact
equality check can't distinguish "converged" from "stuck on a plateau
that will resume moving a few iterations later," and will exit at the
first plateau it hits.

**How it was found:** while testing the general linear-constrained
risk-budgeting solver (which uses this same Dykstra pattern) against a
symmetric two-asset equality constraint, the solver returned an
infeasible result — the equality constraint was violated. Manually
tracing the iteration sequence by hand showed the true fixed point was
still 15–20 iterations away, but the loop had exited early because the
iterate happened to repeat bit-for-bit for two consecutive steps before
resuming its (very slow, corner-geometry-driven) convergence.

**Practical impact:** for symmetric/corner-case inputs, the original
MATLAB's proximal-projection-based constrained solvers could return an
answer that looks converged (no error raised) but doesn't actually
satisfy the constraints it was supposed to enforce — a silent
correctness bug rather than a crash, which is the more dangerous kind.

**Fix in the Python port:** this one is **not fixed** — the same exact
floating-point equality check (via `np.allclose`, which has the same
plateau-blindness) is used in
`quanttoolbox.optim.proximal.proximal_linear_constraints`, to stay
faithful to the original's iteration structure. Instead, the limitation
is documented explicitly in the function's docstring, with a concrete
recommendation (increase `max_iters`, perturb the starting point, or
verify constraint satisfaction directly) for anyone who hits it. This was
a deliberate choice: silently "fixing" convergence-detection logic can
change *which* fixed point a general iterative algorithm reports for
other, non-corner-case inputs too, and doing that safely would need much
more careful analysis than local pattern-matching. Flagging the known
failure mode clearly seemed like the more honest and lower-risk option
than a quick patch. A related, deliberate robustness improvement was
made one level up: `portfolio.risk_budgeting.solve_constrained`'s outer
lambda-bisection now auto-expands its search bracket if the default
window doesn't contain a sign change, since that specific failure mode
(as opposed to this one) was straightforward to detect and fix safely.

---

## Summary table

| # | Location | Nature | Fixed in port? |
|---|---|---|---|
| 1 | `whittle_local_level.m` / `whittle_local_linear_trend.m` Jacobians | Missing `1/(2π)` scaling factor | ✅ Yes |
| 2 | `simulate_multi_gbm.m` | Unused parameter / incomplete implementation | ✅ Yes (reimplemented properly) |
| 3 | `compute_rb_sd_admm_newton.m` / Newton core | No positivity floor, can diverge | ✅ Yes |
| 4 | `mixture_compute_rb_{var,es}.m` algorithm-2 branch | Dead code (unset global) | ✅ Yes (branch not ported, documented) |
| 5 | `proximal_linear_constraints.m` | Exact-equality convergence check can false-exit on a plateau | ⚠️ Not fixed — faithfully reproduced and documented |
