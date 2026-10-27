"""
Second reference solver: hybrid genetic search (HGS-CVRP, Vidal
2022, Python binding `hygese`) on the crisp routing equivalent.

HGS-CVRP supports a route-duration limit with service times but a single
distance matrix for cost and duration.  In the synthetic families every leg is
a scaled copy of one membership per leg type, so the planning duration of an
arc is f_t times its modal travel time (crisp_equivalent); with the planning
matrix as distance matrix, the HGS objective is f_t times the modal travel
time, which differs from the total modal workload by a constant (service) and
a positive factor.  The reference is therefore exact in objective and
constraints for these families only; it is not applied to the heterogeneous
public-data memberships.  Every returned plan is re-evaluated with the fuzzy
evaluator and reported as "no plan" if it violates the ceiling or the fleet
size.
"""

from __future__ import annotations

import time

import numpy as np

from crisp_equivalent import CredibilisticCeiling, PlanningDurationMatrix
from problem import Solution
from solver import _State
from aggregators import max_risk, FFF_REPORT


class HGSCeilingSolver:
    """HGS-CVRP with duration limit L and K*P vehicles (wall-clock limited)."""

    def __init__(self, instance, r_bar: float, time_limit: float = 1.0, seed: int = 0):
        self.inst = instance
        self.ceiling = CredibilisticCeiling(r_bar)
        self.time_limit = time_limit
        self.seed = seed
        self.Q = PlanningDurationMatrix(instance, self.ceiling)

    def run(self) -> dict:
        import hygese as hgs
        inst = self.inst
        n, m = inst.n, inst.K * inst.P
        dist = np.array(self.Q.travel, dtype=float)
        data = dict(distance_matrix=dist,
                    demands=np.array([0.0] + [1.0] * n),
                    vehicle_capacity=float(n),
                    num_vehicles=m,
                    depot=0,
                    service_times=np.array(self.Q.service, dtype=float),
                    duration_limit=float(inst.L) - 1e-7)
        params = hgs.AlgorithmParameters(timeLimit=float(self.time_limit), seed=self.seed)
        t0 = time.perf_counter()
        res = hgs.Solver(parameters=params, verbose=False).solve_cvrp(data, rounding=False)
        elapsed = time.perf_counter() - t0
        base = dict(method="HGS", r_bar=self.ceiling.r_bar, time_limit=self.time_limit,
                    wall_clock_s=elapsed)
        routes = [list(r) for r in (res.routes or []) if len(r) > 0]
        served = sorted(t for r in routes for t in r)
        if len(routes) > m or served != list(range(1, n + 1)):
            return dict(base, found=False, ceiling_met=False)
        routes += [[] for _ in range(m - len(routes))]
        plan = Solution(inst.K, inst.P)
        plan.cells = [[routes[k * inst.P + h] for h in range(inst.P)] for k in range(inst.K)]
        st = _State(inst, plan)
        R, workload, feasible = st.R, st.makespan, st.violation <= 1e-9
        mr = max_risk(R)
        met = bool(feasible and mr <= self.ceiling.r_bar + 1e-12)
        return dict(base, found=met, makespan=workload, max_risk=mr, fff=FFF_REPORT(R),
                    feasible=bool(feasible), ceiling_met=met,
                    solution=[[list(c) for c in row] for row in plan.cells])
