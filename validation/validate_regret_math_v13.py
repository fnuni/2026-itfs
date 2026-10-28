"""Independent finite-rational audit of the v13 shared-completion theorem.

Uses only the Python standard library for the exact arithmetic and circulation
tests. It does not import the v13 research prototype. The reference feasibility
calculation enumerates rational LP vertices, independently of network flow.
Internal checks are not an external review or a proof of literature novelty.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import combinations, product
from math import lcm
from pathlib import Path
import json
import random
import time
import importlib.util
from types import SimpleNamespace

_flow_path = Path(__file__).resolve().parents[1] / 'src/exact_flow_oracle.py'
_spec = importlib.util.spec_from_file_location('audited_flow_implementation', _flow_path)
_flow_implementation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_flow_implementation)


class Dinic:
    def __init__(self, n):
        self.g = [[] for _ in range(n)]

    def add(self, u, v, capacity):
        assert isinstance(capacity, int) and capacity >= 0
        a = [v, capacity, len(self.g[v])]
        b = [u, 0, len(self.g[u])]
        self.g[u].append(a)
        self.g[v].append(b)
        return (u, len(self.g[u]) - 1, capacity)

    def flow(self, s, t):
        answer = 0
        while True:
            level = [-1] * len(self.g)
            level[s] = 0
            queue = [s]
            for u in queue:
                for v, cap, _ in self.g[u]:
                    if cap and level[v] < 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[t] < 0:
                return answer
            pointers = [0] * len(self.g)

            def send(u, amount):
                if u == t:
                    return amount
                while pointers[u] < len(self.g[u]):
                    edge = self.g[u][pointers[u]]
                    v, cap, reverse = edge
                    if cap and level[v] == level[u] + 1:
                        pushed = send(v, min(amount, cap))
                        if pushed:
                            edge[1] -= pushed
                            self.g[v][reverse][1] += pushed
                            return pushed
                    pointers[u] += 1
                return 0

            while True:
                pushed = send(s, sum(e[1] for e in self.g[s]))
                if not pushed:
                    break
                answer += pushed


def circulation(n, edges):
    """Return an exact integer feasible flow, or None."""
    graph = Dinic(n + 2)
    balance = [0] * n
    refs = []
    for u, v, lower, upper in edges:
        if lower > upper:
            return None
        refs.append((graph.add(u, v, upper - lower), lower))
        balance[u] -= lower
        balance[v] += lower
    total = 0
    for u, demand in enumerate(balance):
        if demand > 0:
            graph.add(n, u, demand)
            total += demand
        elif demand < 0:
            graph.add(u, n + 1, -demand)
    if graph.flow(n, n + 1) != total:
        return None
    result = []
    for (u, i, original), lower in refs:
        result.append(lower + original - graph.g[u][i][1])
    return result


def network_mask(lo, hi, horizon, left, right, flags, cells=2, use_k=True):
    """Exact strict feasibility for once-per-plan primitive-leg incidence.

    left[e],right[e] are cell indices, or -1 for a private leg. No leg is
    absent from both plans. Return a common rational endpoint witness.
    """
    d = lcm(*(x.denominator for x in [*lo, *hi, horizon]))
    strict_count = sum(flags)
    k = max(1, strict_count) if use_k else 1
    scale = k * d
    a0, b0 = 0, cells + 1
    source, sink = 2 * (cells + 1), 2 * (cells + 1) + 1
    n = sink + 1
    capacity = sum(int(scale * x) for x in hi)
    limit = int(scale * horizon)
    edges = []
    for e, (i, j) in enumerate(zip(left, right)):
        u = a0 + (i if i >= 0 else cells)
        v = b0 + (j if j >= 0 else cells)
        edges.append((u, v, int(scale * lo[e]), int(scale * hi[e])))
    for i in range(cells):
        low, high = (limit + 1, capacity) if flags[i] else (0, limit)
        edges.append((source, a0 + i, low, high))
    edges.append((source, a0 + cells, 0, capacity))
    for j in range(cells):
        low, high = (limit + 1, capacity) if flags[cells + j] else (0, limit)
        edges.append((b0 + j, sink, low, high))
    edges.append((b0 + cells, sink, 0, capacity))
    edges.append((sink, source, 0, capacity))
    flow = circulation(n, edges)
    if flow is None:
        return None
    point = [F(v, scale) for v in flow[:len(lo)]]
    for z, lower, upper in zip(point, lo, hi):
        assert lower <= z <= upper
    sums = [sum(point[e] for e, i in enumerate(left) if i == c) for c in range(cells)]
    sums += [sum(point[e] for e, i in enumerate(right) if i == c) for c in range(cells)]
    assert [int(s > horizon) for s in sums] == list(flags)
    return point


def solve_square(matrix, rhs):
    n = len(rhs)
    a = [[F(x) for x in row] + [F(b)] for row, b in zip(matrix, rhs)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col]), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        v = a[col][col]
        a[col] = [x / v for x in a[col]]
        for row in range(n):
            if row != col and a[row][col]:
                v = a[row][col]
                a[row] = [x - v * y for x, y in zip(a[row], a[col])]
    return [a[r][-1] for r in range(n)]


def vertex_mask(lo, hi, horizon, left, right, flags, cells=2):
    """Independent exact reference: enumerate vertices of max common margin LP."""
    dim = len(lo) + 1
    constraints = []
    for e in range(len(lo)):
        row = [F(0)] * dim
        row[e] = 1
        constraints.append((row, hi[e]))
        constraints.append(([-x for x in row], -lo[e]))
    row = [F(0)] * dim
    row[-1] = 1
    constraints += [(row, F(1)), ([-x for x in row], F(0))]
    for c, flag in enumerate(flags):
        side = left if c < cells else right
        cell = c if c < cells else c - cells
        row = [F(int(i == cell)) for i in side] + [F(0)]
        if flag:
            row = [-x for x in row]
            row[-1] = 1
            constraints.append((row, -horizon))
        else:
            constraints.append((row, horizon))
    best = None
    for chosen in combinations(constraints, dim):
        point = solve_square([a for a, b in chosen], [b for a, b in chosen])
        if point is None:
            continue
        if all(sum(a * x for a, x in zip(row, point)) <= rhs for row, rhs in constraints):
            if best is None or point[-1] > best[-1]:
                best = point
    return best if best is not None and (not any(flags) or best[-1] > 0) else None


def implementation_mask(lo, hi, horizon, left, right, flags, cells=2):
    d = lcm(*(x.denominator for x in [*lo, *hi, horizon]))
    edges = [(i if i >= 0 else cells, j if j >= 0 else cells,
              int(d * lo[e]), int(d * hi[e])) for e, (i, j) in enumerate(zip(left, right))]
    left_hi, right_hi = [0] * (cells + 1), [0] * (cells + 1)
    for i, j, lower, upper in edges:
        left_hi[i] += upper
        right_hi[j] += upper
    ok, point = _flow_implementation.mask_feasible(left_hi, right_hi, edges, flags,
                                                  int(d * horizon), witness=True)
    if not ok:
        return None
    point = [x / d for x in point]
    sums = [sum(point[e] for e, i in enumerate(left) if i == c) for c in range(cells)]
    sums += [sum(point[e] for e, i in enumerate(right) if i == c) for c in range(cells)]
    assert all(a <= z <= b for a, z, b in zip(lo, point, hi))
    assert [int(s > horizon) for s in sums] == list(flags)
    return point


def owa(values, weights):
    return sum(x * w for x, w in zip(sorted(values, reverse=True), weights))


def exact_gap(lo, hi, horizon, left, right, weights, beta=F(1)):
    cells = len(weights)
    cumulative = [F(0)]
    for w in weights:
        cumulative.append(cumulative[-1] + w)
    patterns = []
    for flags in product((0, 1), repeat=2 * cells):
        value = cumulative[sum(flags[:cells])] - beta * cumulative[sum(flags[cells:])]
        patterns.append((value, flags))
    patterns.sort(reverse=True)
    for value, flags in patterns:
        point = network_mask(lo, hi, horizon, left, right, flags, cells)
        if point is not None:
            return value, point
    raise AssertionError('No feasible pattern')


def direct_membership_risks(endpoints, levels, plateaus, sliver, side, horizon, cells):
    """Evaluate full Pos/Nec sup-min arithmetic of constructed memberships.

    Constructs membership values at duration breakpoints, then adds CLOSED
    positive alpha cuts. It does not use the strong-endpoint ceiling test.
    """
    legs = len(endpoints[0])
    memberships = []
    for e in range(legs):
        b = endpoints[-1][e]
        spans = []
        points = {b}
        for h in range(len(levels) - 1):
            spans.append((endpoints[h][e], 2 * (levels[h] + sliver)))
            spans.append((plateaus[h][e], 2 * levels[h + 1]))
            points.update((endpoints[h][e], plateaus[h][e]))
        memberships.append({x: max([F(0)] + [level for q, level in spans if b <= x <= q])
                            for x in points})
    answer = []
    for c in range(cells):
        es = [e for e, cell in enumerate(side) if cell == c]
        if not es:
            answer.append(F(0))
            continue
        alpha_levels = sorted({v for e in es for v in memberships[e].values() if v > 0})
        above = F(0)
        below = F(0)
        for a in alpha_levels:
            cuts = [[x for x, v in memberships[e].items() if v >= a] for e in es]
            if sum(max(cut) for cut in cuts) > horizon:
                above = max(above, a)
            if sum(min(cut) for cut in cuts) <= horizon:
                below = max(below, a)
        answer.append((above + 1 - below) / 2)
    return answer


def run():
    started = time.perf_counter()
    rng = random.Random(130028)
    masks_checked = 0
    for legs in (1, 2, 3):
        for _ in range(8):
            left, right = [], []
            for e in range(legs):
                pair = rng.choice([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                                   if (a, b) != (-1, -1)])
                left.append(pair[0]); right.append(pair[1])
            lo = [F(rng.randrange(3), 4) for _ in range(legs)]
            hi = [x + F(rng.randrange(4), 4) for x in lo]
            horizon = F(rng.randrange(5), 4)
            for flags in product((0, 1), repeat=4):
                network = network_mask(lo, hi, horizon, left, right, flags)
                reference = vertex_mask(lo, hi, horizon, left, right, flags)
                implementation = implementation_mask(lo, hi, horizon, left, right, flags)
                assert (network is not None) == (reference is not None), (lo, hi, horizon, left, right, flags)
                assert (implementation is not None) == (reference is not None), (lo, hi, horizon, left, right, flags)
                masks_checked += 1

    # Strict feasibility may require fractional flows even with integer data.
    # x>0,y>0 and x+y<=1 is strictly feasible, but all-strict +1 is not.
    lo, hi, left, right = [F(0), F(0)], [F(1), F(1)], [0, 1], [0, 0]
    # Different row horizons are needed for this textbook fixture; the common
    # horizon version uses private fixed offsets to move the lower bounds.
    lo = [F(0), F(0), F(1), F(1)]
    hi = [F(1), F(1), F(1), F(1)]
    left, right = [0, 1, 0, 1], [0, 0, -1, -1]
    flags = (1, 1, 0, 0)
    assert network_mask(lo, hi, F(1), left, right, flags) is not None
    assert implementation_mask(lo, hi, F(1), left, right, flags) is not None
    assert network_mask(lo, hi, F(1), left, right, flags, use_k=False) is None

    # A tiny strictly positive margin can still change a whole count level.
    tiny = F(1, 2**70)
    assert vertex_mask([F(0)], [tiny], F(0), [0], [-1], (1, 0, 0, 0))[-1] == tiny
    assert network_mask([F(0)], [tiny], F(0), [0], [-1], (1, 0, 0, 0)) is not None
    assert implementation_mask([F(0)], [tiny], F(0), [0], [-1], (1, 0, 0, 0)) is not None

    # The public routing oracle must reject repeated primitive legs.
    import numpy as np
    mock = SimpleNamespace(endpoints=np.array([[0.] * 6, [2., 1.8, 1.6, 1.4, 1.2, 0.]]),
                           L=1., incidence=np.array([[0, 0], [1, 1]]))
    try:
        _flow_implementation.ExactFlowOracle(mock, [1.], horizon=mock.L)
    except ValueError:
        pass
    else:
        raise AssertionError('Repeated within-route primitive leg was not rejected')
    mock.incidence = np.array([[0], [1]])
    oracle = _flow_implementation.ExactFlowOracle(mock, [1.], horizon=mock.L)
    assert oracle.pair_fraction([0], [0]) == 0
    assert oracle.pair_fraction([1], [1]) == 0
    oracle = _flow_implementation.ExactFlowOracle(mock, [F(1, 3), F(2, 3)], horizon=mock.L)
    try:
        oracle._network([1, 1], [0, 0], 0)
    except ValueError:
        pass
    else:
        raise AssertionError('Across-cell repeated primitive leg was not rejected')

    construction_checks = 0
    levels = [F(0), F(1, 4), F(1, 2)]
    for _ in range(24):
        legs = 3
        pairs = [rng.choice([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                             if (a, b) != (-1, -1)]) for e in range(legs)]
        left, right = [p[0] for p in pairs], [p[1] for p in pairs]
        middle = [F(rng.randrange(5), 4) for e in range(legs)]
        upper = [x + F(rng.randrange(5), 4) for x in middle]
        endpoints = [upper, middle, [F(0)] * legs]
        horizon = F(3, 4)
        weights = [F(1, 3), F(2, 3)]  # deliberately NON-decreasing
        for beta in (F(0), F(1), F(3, 2)):
            gap_results = [exact_gap(endpoints[h+1], endpoints[h], horizon,
                                     left, right, weights, beta) for h in range(2)]
            theoretical = sum((levels[h+1] - levels[h]) * gap_results[h][0] for h in range(2))
            plateaus = [r[1] for r in gap_results]
            sliver = F(1, 10000)
            a = direct_membership_risks(endpoints, levels, plateaus, sliver, left, horizon, 2)
            b = direct_membership_risks(endpoints, levels, plateaus, sliver, right, horizon, 2)
            actual = owa(a, weights) - beta * owa(b, weights)
            assert 0 <= theoretical - actual <= (1 + beta) * 2 * sliver
            construction_checks += 1
    result = {
        'status': 'passed',
        'scope': 'Independent internal mathematical/numerical audit; not external verification or novelty certification.',
        'random_seed': 130028,
        'rational_mask_comparisons': masks_checked,
        'audited_implementation_mask_comparisons': masks_checked,
        'independent_sup_min_membership_constructions': construction_checks,
        'k_scaling_counterexample_verified': True,
        'repeated_leg_rejection_checked': True,
        'identical_and_empty_plan_checks_passed': True,
        'tiny_positive_margin_fixture': str(tiny),
        'numerics': 'Fractions and arbitrary-precision integer flow; no floating-point tolerance in these checks.',
        'elapsed_seconds': time.perf_counter() - started,
    }
    output = Path(__file__).with_name('independent_exact_validation.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
