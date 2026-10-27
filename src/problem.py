"""
MS-VRPPU-FR instance model and solution encoding.
"""

from __future__ import annotations

import math
import random

from fuzzy_core import TriangularFN, TrapezoidalFN, GaussianFN, IntervalType2FN


class MembershipFactory:
    """Builds fuzzy durations of a given shape from a modal value and spreads."""

    SHAPES = ("tfn", "trapezoidal", "gaussian", "it2")

    def __init__(self, shape: str = "tfn", left: float = 0.25, right: float = 0.25,
                 plateau: float = 0.08, delta: float = 0.25):
        if shape not in self.SHAPES:
            raise ValueError(f"unknown shape {shape}")
        self.shape, self.left, self.right = shape, left, right
        self.plateau, self.delta = plateau, delta

    def zero(self):
        if self.shape == "tfn":
            return TriangularFN(0.0, 0.0, 0.0)
        if self.shape == "trapezoidal":
            return TrapezoidalFN(0.0, 0.0, 0.0, 0.0)
        if self.shape == "gaussian":
            return GaussianFN(0.0, 1e-9)
        return IntervalType2FN(0.0, 0.0, 0.0, self.delta)

    def make(self, modal: float):
        a = modal * (1.0 - self.left)
        c = modal * (1.0 + self.right)
        if self.shape == "tfn":
            return TriangularFN(a, modal, c)
        if self.shape == "trapezoidal":
            h = modal * self.plateau
            return TrapezoidalFN(a, max(a, modal - h), min(c, modal + h), c)
        if self.shape == "gaussian":
            # The Gaussian half-height points match the TFN support endpoints
            # for symmetric spreads. This is a sensitivity convention, not
            # equality of the TFN/Gaussian supports or half-height widths.
            hw = 0.5 * (modal * self.left + modal * self.right)
            return GaussianFN(modal, max(1e-6, hw / math.sqrt(2.0 * math.log(2.0))))
        return IntervalType2FN(a, modal, c, self.delta)


class Instance:
    """A multi-shift possibilistic routing instance."""

    def __init__(self, n, K, P, L, coords, service_modal, speed,
                 factory: MembershipFactory):
        self.n, self.K, self.P, self.L = n, K, P, L
        self.coords = coords                 # index 0 = depot
        self.service_modal = service_modal   # length n+1, entry 0 unused
        self.speed = speed
        self.factory = factory
        self.cell_evals = 0                  # fuzzy cell evaluations (budget unit)
        self._dist = self._distance_matrix()
        mk_t = getattr(factory, "make_travel", factory.make)
        mk_s = getattr(factory, "make_service", factory.make)
        self._travel = [[mk_t(self._dist[i][j] / speed) if i != j else factory.zero()
                         for j in range(n + 1)] for i in range(n + 1)]
        self._service = [factory.zero()] + [mk_s(service_modal[i])
                                            for i in range(1, n + 1)]

    def _distance_matrix(self):
        c = self.coords
        m = len(c)
        return [[math.dist(c[i], c[j]) for j in range(m)] for i in range(m)]

    def travel(self, i, j):
        return self._travel[i][j]

    def service(self, i):
        return self._service[i]

    def modal_travel(self, i, j):
        return self._dist[i][j] / self.speed

    # --- evaluation -------------------------------------------------------
    def cell_duration(self, seq):
        """Fuzzy duration of one (vehicle, shift) cell given its ordered tasks."""
        f = self.factory
        self.cell_evals += 1
        if not seq:
            return f.zero(), 0.0
        total = self.travel(0, seq[0])
        modal = self.modal_travel(0, seq[0])
        for idx, t in enumerate(seq):
            total = total + self.service(t)
            modal += self.service_modal[t]
            nxt = seq[idx + 1] if idx + 1 < len(seq) else 0
            total = total + self.travel(t, nxt)
            modal += self.modal_travel(t, nxt)
        return total, modal

    def evaluate(self, solution):
        """Returns (risk matrix R [K x P], total modal makespan)."""
        R, makespan = [], 0.0
        for k in range(self.K):
            row = []
            for h in range(self.P):
                dur, modal = self.cell_duration(solution.cells[k][h])
                row.append(dur.overtime_risk(self.L))
                makespan += modal
            R.append(row)
        return R, makespan


class Solution:
    """Vehicle x shift grid of ordered task sequences."""

    __slots__ = ("cells", "K", "P")

    def __init__(self, K, P):
        self.K, self.P = K, P
        self.cells = [[[] for _ in range(P)] for _ in range(K)]

    def clone(self):
        s = Solution(self.K, self.P)
        s.cells = [[list(c) for c in row] for row in self.cells]
        return s

    def all_tasks(self):
        return [t for row in self.cells for c in row for t in c]

    def remove(self, tasks):
        ts = set(tasks)
        for row in self.cells:
            for h in range(len(row)):
                row[h] = [t for t in row[h] if t not in ts]


class InstanceGenerator:
    """Reproducible generator for the synthetic municipal case families."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    CALIBRATION_ITERS = 60

    def generate(self, n, K, P, L, layout="uniform", factory=None,
                 service_range=(8.0, 18.0), speed=1.0, n_clusters=4,
                 tightness=0.85):
        """
        `tightness` rho calibrates the instance so that the expected modal
        workload per (vehicle, shift) cell equals rho * L.  Without this the
        overtime credibilities saturate at 0 or 1 and the fairness problem
        degenerates; rho is reported with every case.
        """
        rng = self.rng
        factory = factory or MembershipFactory("tfn")
        coords = [(50.0, 50.0)]
        if layout == "clustered":
            centres = [(rng.uniform(15, 85), rng.uniform(15, 85)) for _ in range(n_clusters)]
            for _ in range(n):
                cx, cy = centres[rng.randrange(n_clusters)]
                coords.append((min(100, max(0, rng.gauss(cx, 8.0))),
                               min(100, max(0, rng.gauss(cy, 8.0)))))
        else:
            for _ in range(n):
                coords.append((rng.uniform(0, 100), rng.uniform(0, 100)))
        coords = [(round(x, 4), round(y, 4)) for x, y in coords]
        service = [0.0] + [rng.uniform(*service_range) for _ in range(n)]

        # --- tightness calibration -------------------------------------
        # Every duration is linear in the common scale of service times and
        # travel times, so one factor calibrates the instance.  The workload
        # proxy is the modal makespan of a short, seeded, makespan-only
        # neighbourhood search (the reference construction heuristic followed
        # by CALIBRATION_ITERS iterations of the risk-aware LNS with alpha = 1):
        # a single nearest-neighbour tour under-counts the depot legs when
        # K*P is large and would make multi-vehicle families infeasible.
        from solver import RALNS                      # local import: no cycle
        from aggregators import MeanRisk
        probe = Instance(n, K, P, L, coords, service, speed, factory)
        srch = RALNS(probe, MeanRisk(), alpha=1.0,
                     iterations=self.CALIBRATION_ITERS, seed=0)
        nominal = srch.run()["makespan"]
        factor = (K * P * tightness * L) / max(nominal, 1e-9)
        service = [round(v * factor, 6) for v in service]
        speed = round(speed / factor, 6)
        return Instance(n, K, P, L, coords, service, speed, factory)


CASES = {
    # name: (n, K, P, L, layout, left spread, right spread)
    "C1": dict(n=30, K=1, P=3, L=120.0, layout="uniform", left=0.25, right=0.25),
    "C2": dict(n=50, K=1, P=4, L=150.0, layout="clustered", left=0.25, right=0.25),
    "C3": dict(n=80, K=1, P=5, L=180.0, layout="clustered", left=0.30, right=0.30),
    "C4": dict(n=50, K=1, P=4, L=130.0, layout="uniform", left=0.15, right=0.35),
    "F2": dict(n=60, K=2, P=3, L=150.0, layout="clustered", left=0.25, right=0.25),
    "F4": dict(n=96, K=4, P=3, L=150.0, layout="clustered", left=0.25, right=0.25),
}
