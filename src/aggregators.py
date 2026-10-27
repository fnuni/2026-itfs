"""
Fairness aggregators on the vehicle-shift risk matrix R in [0,1]^{K x P}.

The FFF is the cell-anonymous OWA over all K*P cells; the row-wise
functional (average of vehicle OWA scores) is kept as RowOWAFairness.

All aggregators are risk functionals: lower is better.  They share the
RiskAggregator interface so that the solver can be held fixed while only the
aggregation operator varies (controlled component comparison).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math


def front_loaded_weights(P: int) -> list[float]:
    """w_p = 2 (P - p + 1) / (P (P + 1)), strictly decreasing, sums to 1."""
    return [2.0 * (P - p + 1) / (P * (P + 1)) for p in range(1, P + 1)]


class RiskAggregator(ABC):
    """Maps a K x P risk matrix to a scalar risk score in [0, 1]."""

    name: str = "abstract"

    @abstractmethod
    def __call__(self, R: list[list[float]]) -> float:
        ...

    @staticmethod
    def _flat(R):
        return [v for row in R for v in row]


class MeanRisk(RiskAggregator):
    name = "Mean"

    def __call__(self, R):
        f = self._flat(R)
        return sum(f) / len(f)


class MaxRisk(RiskAggregator):
    name = "Max"

    def __call__(self, R):
        return max(self._flat(R))


class OWAFairness(RiskAggregator):
    """
    FFF: cell-anonymous OWA over all K*P vehicle-shift cells,
    FFF(R) = sum_j w_j r_[j], r_[1] >= ... >= r_[KP] the non-increasing
    rearrangement of every cell risk, w the front-loaded (arithmetic) profile
    on K*P ranks.  Because the cells are interchangeable routes (homogeneous
    vehicles, no inter-shift coupling), the value does not depend on which
    vehicle a route is assigned to.  For K = 1 it coincides with the row-wise
    row-wise functional.
    """

    name = "FFF (OWA)"

    def __init__(self, weights=None):
        self._w = weights

    def _weights(self, n):
        return self._w if self._w is not None else front_loaded_weights(n)

    def __call__(self, R):
        f = sorted(self._flat(R), reverse=True)
        w = self._weights(len(f))
        return sum(wi * ri for wi, ri in zip(w, f))


class RowOWAFairness(RiskAggregator):
    """
    Row-wise functional (diagnostic only): uniform average over
    vehicles of a row OWA, sum_k (1/K) sum_p w_p r_{k(p)}.  With interchangeable
    cells it is minimised by assigning the riskiest routes to the same vehicle
    (block assignment; Proposition in the manuscript), see block_assignment().
    """

    name = "Row-wise FFF"

    def __init__(self, weights=None):
        self._w = weights

    def __call__(self, R):
        K, P = len(R), len(R[0])
        w = self._w if self._w is not None else front_loaded_weights(P)
        return sum(sum(wi * ri for wi, ri in zip(w, sorted(row, reverse=True)))
                   for row in R) / K


class GeneralizedOWAFairness(OWAFairness):
    """
    g-OWA family on the cells: sum_j w_j g(r_[j]) with g(x) = x^theta,
    theta >= 1 (non-decreasing and convex).  theta = 1 recovers the FFF.
    """

    def __init__(self, theta: float = 2.0, weights=None):
        super().__init__(weights)
        self.theta = float(theta)
        self.name = f"g-OWA (theta={theta:g})"

    def __call__(self, R):
        f = sorted(self._flat(R), reverse=True)
        w = self._weights(len(f))
        return sum(wi * (ri ** self.theta) for wi, ri in zip(w, f))


# ---------------------------------------------------------------------------
# Assignment of routes to vehicles (cells are interchangeable)
# ---------------------------------------------------------------------------

def _partitions(items, K, P):
    """Unordered partitions of items into K groups of size P (exhaustive)."""
    if K == 1:
        yield [list(items)]
        return
    first, rest = items[0], items[1:]
    from itertools import combinations
    for comb in combinations(range(len(rest)), P - 1):
        grp = [first] + [rest[i] for i in comb]
        left = [rest[i] for i in range(len(rest)) if i not in comb]
        for tail in _partitions(left, K - 1, P):
            yield [grp] + tail


def block_assignment(R):
    """Rows filled with consecutive blocks of the sorted cell risks."""
    K, P = len(R), len(R[0])
    f = sorted(RiskAggregator._flat(R), reverse=True)
    return [f[k * P:(k + 1) * P] for k in range(K)]


def balanced_assignment(R, max_partitions=200000):
    """
    Re-assign the K*P cell risks (routes) to vehicles so that the vector of
    vehicle risk totals, sorted non-increasingly, is lexicographically
    minimal (min-max first).  Exhaustive for the sizes used in the paper;
    for larger instances a serpentine (boustrophedon) deal is returned.
    Workload, every cell risk, the ceiling and the FFF are unchanged.
    """
    K, P = len(R), len(R[0])
    f = sorted(RiskAggregator._flat(R), reverse=True)
    if K == 1:
        return [f]
    from math import comb, factorial
    count = 1
    for k in range(K):
        count *= comb(P * (K - k) - 1, P - 1)
    if count > max_partitions:
        rows = [[] for _ in range(K)]
        for j, v in enumerate(f):
            r, c = divmod(j, K)
            rows[c if r % 2 == 0 else K - 1 - c].append(v)
        return rows
    best, best_key = None, None
    for part in _partitions(list(range(K * P)), K, P):
        key = tuple(sorted((round(sum(f[i] for i in g), 12) for g in part), reverse=True))
        if best_key is None or key < best_key:
            best_key, best = key, part
    return [[f[i] for i in g] for g in best]


def vehicle_totals(R):
    return sorted((sum(row) for row in R), reverse=True)


class CVaRRisk(RiskAggregator):
    """Conditional value at risk of the vectorised risk grid at level beta."""

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.name = f"CVaR({beta:g})"

    def __call__(self, R):
        return cvar(R, self.beta)


class GiniPenalisedRisk(RiskAggregator):
    """mean(r) + eta * Gini(r): a scalar inequality-penalised mean."""

    def __init__(self, eta: float = 1.0):
        self.eta = eta
        self.name = f"Mean+Gini (eta={eta:g})"

    def __call__(self, R):
        f = self._flat(R)
        n = len(f)
        mu = sum(f) / n
        if mu <= 1e-12:
            return 0.0
        s = sorted(f)
        gini = sum((2 * (i + 1) - n - 1) * s[i] for i in range(n)) / (n * n * mu)
        return min(1.0, mu + self.eta * mu * gini)


class JainPenalisedRisk(RiskAggregator):
    """mean(r) + eta * (1 - Jain index)."""

    def __init__(self, eta: float = 0.5):
        self.eta = eta
        self.name = f"Mean+Jain (eta={eta:g})"

    def __call__(self, R):
        f = self._flat(R)
        n = len(f)
        s1, s2 = sum(f), sum(v * v for v in f)
        mu = s1 / n
        jain = 1.0 if s2 <= 1e-12 else (s1 * s1) / (n * s2)
        return min(1.0, mu + self.eta * (1.0 - jain))


# ---------------------------------------------------------------------------
# Reporting metrics (common evaluation scale, independent of the search
# objective; used so that every aggregator is judged on the same yardstick).
# ---------------------------------------------------------------------------

def max_risk(R):
    return max(v for row in R for v in row)


def gini(R):
    f = sorted(v for row in R for v in row)
    n = len(f)
    mu = sum(f) / n
    if mu <= 1e-12:
        return 0.0
    return sum((2 * (i + 1) - n - 1) * f[i] for i in range(n)) / (n * n * mu)


def jain(R):
    f = [v for row in R for v in row]
    n = len(f)
    s1, s2 = sum(f), sum(v * v for v in f)
    return 1.0 if s2 <= 1e-12 else (s1 * s1) / (n * s2)


def cvar(R, beta=0.9):
    f = sorted((v for row in R for v in row), reverse=True)
    if not 0.0 <= beta < 1.0:
        raise ValueError("CVaR level must be in [0,1)")
    if not f:
        raise ValueError("CVaR requires a nonempty risk grid")
    mass = (1.0 - beta) * len(f)
    whole = int(math.floor(mass))
    fraction = mass - whole
    total = math.fsum(f[:whole])
    if fraction > 0.0 and whole < len(f):
        total += fraction * f[whole]
    return total / mass


FFF_REPORT = OWAFairness()          # common reporting functional


def aggregator_suite():
    """The aggregators compared in the isolation study (Table: aggregator study)."""
    return [
        OWAFairness(),
        GeneralizedOWAFairness(2.0),
        GeneralizedOWAFairness(3.0),
        MeanRisk(),
        MaxRisk(),
        CVaRRisk(0.9),
        GiniPenalisedRisk(1.0),
        JainPenalisedRisk(0.5),
    ]
