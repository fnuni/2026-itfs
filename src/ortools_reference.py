"""
Reference solver: a general-purpose routing engine applied through the crisp
routing equivalent of the credibilistic ceiling.

Each of the K*P vehicle--shift cells becomes one route slot of Google OR-Tools
(constraint-programming routing library).  The arc cost is the modal travel
time; a capacity dimension accumulates planning durations (crisp_equivalent),
service included, with the shift limit L as route-length bound.  Planning
durations are rounded up and the limit down after integer scaling, so every
plan accepted by OR-Tools satisfies the ceiling; plans are nevertheless
re-evaluated with the fuzzy evaluator of problem.Instance.  Node disjunctions
with a large penalty let the construction start from partial routes; a plan
that leaves a task unserved is reported as "no plan".
"""

from __future__ import annotations

import math
import time

from crisp_equivalent import CredibilisticCeiling, PlanningDurationMatrix
from problem import Solution
from solver import _State
from aggregators import max_risk, FFF_REPORT


class ORToolsCeilingSolver:
    """Guided local search on the crisp equivalent (wall-clock limited)."""

    SCALE = 1000.0
    DROP_PENALTY = 10 ** 9

    def __init__(self, instance, r_bar: float, time_limit: float = 10.0,
                 first_solution: str = "PARALLEL_CHEAPEST_INSERTION"):
        self.inst = instance
        self.ceiling = CredibilisticCeiling(r_bar)
        self.time_limit = time_limit
        self.first_solution = first_solution
        self.Q = PlanningDurationMatrix(instance, self.ceiling)

    def run(self) -> dict:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        inst, S = self.inst, self.SCALE
        n, m = inst.n, inst.K * inst.P
        man = pywrapcp.RoutingIndexManager(n + 1, m, 0)
        model = pywrapcp.RoutingModel(man)

        def cost(i, j):
            a, b = man.IndexToNode(i), man.IndexToNode(j)
            return int(round(S * inst.modal_travel(a, b)))

        def length(i, j):
            a, b = man.IndexToNode(i), man.IndexToNode(j)
            return int(math.ceil(S * (self.Q.travel[a][b] + self.Q.service[a])))

        model.SetArcCostEvaluatorOfAllVehicles(model.RegisterTransitCallback(cost))
        model.AddDimension(model.RegisterTransitCallback(length), 0,
                           int(math.floor(S * inst.L)), True, "planning_duration")
        for node in range(1, n + 1):
            model.AddDisjunction([man.NodeToIndex(node)], self.DROP_PENALTY)
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = getattr(
            routing_enums_pb2.FirstSolutionStrategy, self.first_solution)
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        params.time_limit.FromMilliseconds(int(1000 * self.time_limit))
        t0 = time.perf_counter()
        sol = model.SolveWithParameters(params)
        elapsed = time.perf_counter() - t0
        base = dict(method="OR-Tools", r_bar=self.ceiling.r_bar,
                    time_limit=self.time_limit, wall_clock_s=elapsed)
        if sol is None:
            return dict(base, found=False, ceiling_met=False)
        dropped = sum(1 for node in range(1, n + 1)
                      if sol.Value(model.NextVar(man.NodeToIndex(node)))
                      == man.NodeToIndex(node))
        if dropped:
            return dict(base, found=False, ceiling_met=False, dropped=dropped)
        routes = []
        for v in range(m):
            idx, seq = sol.Value(model.NextVar(model.Start(v))), []
            while not model.IsEnd(idx):
                seq.append(man.IndexToNode(idx))
                idx = sol.Value(model.NextVar(idx))
            routes.append(seq)
        plan = Solution(inst.K, inst.P)
        plan.cells = [[routes[k * inst.P + h] for h in range(inst.P)]
                      for k in range(inst.K)]
        st = _State(inst, plan)
        R, workload, feasible = st.R, st.makespan, st.violation <= 1e-9
        mr = max_risk(R)
        return dict(base, found=True, makespan=workload, max_risk=mr,
                    fff=FFF_REPORT(R), feasible=bool(feasible),
                    ceiling_met=bool(feasible and mr <= self.ceiling.r_bar + 1e-12),
                    solution=[[list(c) for c in row] for row in plan.cells])
