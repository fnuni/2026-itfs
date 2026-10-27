"""
Statistical procedures used by every table of the paper.

Non-parametric throughout, because the run-level indicators (maximum
credibility, workload, hypervolume) are neither normal nor homoscedastic
across families:

  bootstrap_ci        percentile bootstrap CI of the median
  bootstrap_mean_ci   percentile bootstrap CI of the mean
  mannwhitney_u       two-sided Mann-Whitney U (normal approximation with tie
                      correction) returning (U, p, Vargha-Delaney A12)
  holm                Holm-Bonferroni step-down adjustment of a family of p
  mcnemar_p           exact two-sided McNemar test on paired binary outcomes
  sign_test_p         exact two-sided sign test on paired differences

The exact tests are used wherever the comparison is paired (ablation variants,
paired method comparisons), which the Mann-Whitney U is not.
"""

from __future__ import annotations

import math
import random
import statistics
from math import comb


def bootstrap_ci(values, reps=2000, level=0.95, seed=11):
    """Percentile bootstrap CI of the MEDIAN."""
    rng = random.Random(seed)
    n = len(values)
    meds = sorted(statistics.median(rng.choices(values, k=n)) for _ in range(reps))
    return meds[int((1 - level) / 2 * reps)], meds[int((1 + level) / 2 * reps) - 1]


def bootstrap_mean_ci(values, reps=4000, level=0.95, seed=11):
    """Percentile bootstrap CI of the MEAN (used for the equivalence analysis)."""
    rng = random.Random(seed)
    n = len(values)
    ms = sorted(sum(rng.choices(values, k=n)) / n for _ in range(reps))
    return ms[int((1 - level) / 2 * reps)], ms[int((1 + level) / 2 * reps) - 1]


def mannwhitney_u(x, y):
    """Two-sided Mann-Whitney U, normal approximation with tie correction.

    Returns (U, p, A12); A12 is the Vargha-Delaney effect size for x over y.
    """
    nx, ny = len(x), len(y)
    combined = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    n = nx + ny
    ranks, i, tie_term = [0.0] * n, 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    rx = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    U = rx - nx * (nx + 1) / 2.0
    mu = nx * ny / 2.0
    var = nx * ny / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    sd = math.sqrt(max(var, 0.0))
    z = 0.0 if sd == 0 else (U - mu) / sd
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return U, p, U / (nx * ny)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, prev = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, (m - rank) * pvals[i]))
        out[i] = adj
        prev = adj
    return out


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def sign_test_p(diffs, tol=1e-12):
    """Exact two-sided sign test on paired differences (ties dropped)."""
    pos = sum(1 for d in diffs if d > tol)
    neg = sum(1 for d in diffs if d < -tol)
    return mcnemar_p(pos, neg)


def _ranks(xs):
    """Fractional (midpoint) ranks, 1-based."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman_rho(x, y):
    """Spearman rank correlation with midpoint ranks for ties."""
    rx, ry = _ranks(x), _ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def kendall_tau_b(x, y):
    """Kendall's tau-b (ties handled in both margins)."""
    n, conc, disc, tx, ty = len(x), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = x[i] - x[j], y[i] - y[j]
            s = a * b
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
            else:
                if a == 0 and b != 0:
                    tx += 1
                if b == 0 and a != 0:
                    ty += 1
    d = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return (conc - disc) / d if d > 0 else float("nan")
