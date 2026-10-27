"""Exhaustive direct OWA reference, independent of heuristic search.

Columns are feasible routes with fixed workload and credibility. For scaled
nonnegative memberships the shortest modal tour dominates every other tour of
the same subset; for heterogeneous triangular legs all directed tours are
enumerated and only workload/risk dominated feasible columns are removed.
Unlabelled partitions and all retained route choices are enumerated exactly.
Certificates are numerical, with the explicit floating feasibility tolerance.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import time
import numpy as np
from exact_small import SubsetTSP
from problem import Solution
from fuzzy_core import TriangularFN

TOL = 1e-9

@dataclass(frozen=True)
class RouteColumn:
    mask: int
    sequence: tuple[int, ...]
    workload: float
    risk: float
    optimistic: float

def cell_owa(risks, slots):
    ordered = sorted(risks, reverse=True)
    return sum(2.0 * (slots - j) / (slots * (slots + 1)) * x
               for j, x in enumerate(ordered))

def partitions(mask, slots):
    """Each partition of a nonempty mask into at most slots sets once."""
    if not mask:
        yield ()
        return
    if slots == 1:
        yield (mask,)
        return
    first = mask & -mask
    sub = mask
    while sub:
        if sub & first:
            for tail in partitions(mask ^ sub, slots - 1):
                yield (sub,) + tail
        sub = (sub - 1) & mask

def homothetic_columns(inst):
    """The caller must establish the common-shape, nonnegative-leg premise."""
    if inst.n > 12:
        raise ValueError('Direct enumeration is limited to 12 tasks')
    # The released homothetic numerical reference uses triangular legs.
    # Reject a generic call that would silently apply the shortest-tour
    # reduction to heterogeneous shapes or negative duration supports.
    for legs in [[(inst.travel(i,j),inst.modal_travel(i,j))
                  for i in range(inst.n+1) for j in range(inst.n+1) if i!=j],
                 [(inst.service(i),inst.service_modal[i]) for i in range(1,inst.n+1)]]:
        ratios=[]
        for duration, reference in legs:
            if not isinstance(duration,TriangularFN) or duration.a < 0:
                raise ValueError('The released scaled-copy reference requires nonnegative TFNs')
            if reference > 0:
                ratios.append((duration.a/reference,duration.b/reference,duration.c/reference))
        if ratios and any(max(abs(a-b) for a,b in zip(r,ratios[0]))>1e-10 for r in ratios):
            raise ValueError('Heterogeneous legs require the all-tour reference')
    D = np.array([[inst.modal_travel(i, j) for j in range(inst.n + 1)]
                  for i in range(inst.n + 1)])
    tsp = SubsetTSP(D)
    columns = {}
    for mask in range(1, 1 << inst.n):
        seq = tsp.tour(mask)
        duration, workload = inst.cell_duration(seq)
        optimistic = duration.support[0]
        if optimistic <= inst.L + TOL:
            columns[mask] = [RouteColumn(mask, tuple(seq), float(workload),
                                        duration.overtime_risk(inst.L), optimistic)]
    return columns

def heterogeneous_columns(inst):
    if inst.n > 8:
        raise ValueError('Full directed route enumeration is limited to 8 tasks')
    columns = {}
    routes_enumerated = 0
    for k in range(1, inst.n + 1):
        for seq in itertools.permutations(range(1, inst.n + 1), k):
            routes_enumerated += 1
            duration, workload = inst.cell_duration(seq)
            if duration.support[0] > inst.L + TOL:
                continue
            mask = sum(1 << (i - 1) for i in seq)
            risk = duration.overtime_risk(inst.L)
            cand = RouteColumn(mask, seq, float(workload), risk, duration.support[0])
            old = columns.setdefault(mask, [])
            if any(x.workload <= cand.workload and x.risk <= cand.risk for x in old):
                continue
            old[:] = [x for x in old if not
                      (cand.workload <= x.workload and cand.risk <= x.risk)]
            old.append(cand)
    return columns, routes_enumerated

def better_ceiling(workload, fff, previous):
    return previous is None or workload < previous['workload'] - TOL or (abs(workload-previous['workload']) <= TOL and fff < previous['fff'])

def plan_cells(routes, K, P):
    flat = [list(x.sequence) for x in routes] + [[]] * (K * P - len(routes))
    return [[list(c) for c in flat[k * P:(k + 1) * P]] for k in range(K)]

def direct_reference(columns, n, K, P, scale, alphas=(.1, .3, .7)):
    start = time.perf_counter()
    slots = K * P
    best = {a: None for a in alphas}
    by_max = {}
    evaluated = 0
    for groups in partitions((1 << n) - 1, slots):
        if any(g not in columns for g in groups):
            continue
        for routes in itertools.product(*(columns[g] for g in groups)):
            evaluated += 1
            workload = sum(x.workload for x in routes)
            risks = [x.risk for x in routes]
            fff, maximum = cell_owa(risks, slots), max(risks)
            values = {a: a * workload / scale + (1 - a) * fff for a in alphas}
            improved = [a for a in alphas if best[a] is None or values[a] < best[a]['J']]
            ceiling_key = maximum
            prev = by_max.get(ceiling_key)
            improved_ceiling = better_ceiling(workload, fff, prev)
            if improved or improved_ceiling:
                rec = dict(workload=workload, fff=fff, max_risk=maximum,
                           solution=plan_cells(routes, K, P))
                for a in improved:
                    best[a] = dict(rec, J=values[a])
                if improved_ceiling:
                    by_max[ceiling_key] = rec
    if not evaluated:
        return dict(status='Infeasible', evaluated_plans=0)
    # Every breakpoint of the true max-risk objective, solved optimally, with
    # FFF used only to select among tied minimum-workload plans. This is a
    # stronger comparator than a finite heuristic ceiling sweep.
    frontier = []
    incumbent = None
    for ceiling, rec in sorted(by_max.items()):
        if better_ceiling(rec['workload'], rec['fff'], incumbent):
            incumbent = rec
            frontier.append(dict(rec, ceiling=ceiling))
    sweep = {}
    for a in alphas:
        rec = min(frontier, key=lambda x: a * x['workload'] / scale + (1 - a) * x['fff'])
        sweep[a] = dict(rec, J=a * rec['workload'] / scale + (1 - a) * rec['fff'])
    return dict(status='Exhaustive optimum', direct={str(a): x for a, x in best.items()},
                oracle_sweep={str(a): x for a, x in sweep.items()},
                evaluated_plans=evaluated, route_columns=sum(map(len, columns.values())),
                ceiling_breakpoints=len(by_max), frontier_size=len(frontier), ceiling_frontier=frontier,
                seconds=time.perf_counter()-start, feasibility_tolerance=TOL)

def audit_plan(inst, out, alpha, scale):
    sol = Solution(inst.K, inst.P)
    sol.cells = out['solution']
    assert sorted(sol.all_tasks()) == list(range(1, inst.n + 1))
    risks, workload = inst.evaluate(sol)
    flat = [x for row in risks for x in row]
    fff = cell_owa(flat, inst.K * inst.P)
    for key, value in [('workload', workload), ('fff', fff), ('max_risk', max(flat)),
                       ('J', alpha * workload / scale + (1-alpha)*fff)]:
        assert abs(out[key]-value) <= 1e-7 * max(1., abs(value)), (key,out[key],value)
    assert all(inst.cell_duration(c)[0].support[0] <= inst.L+TOL
               for row in sol.cells for c in row)

