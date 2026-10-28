"""Exact common-leg regret oracle for once-per-plan routing incidence.

Input endpoint floats and horizon are interpreted as exact dyadic rationals.
Violation patterns use bounded bipartite circulation. Multiplying capacities
by k (the number of strict node rows) and adding one to their lower bounds
decides simultaneous STRICT feasibility exactly by total unimodularity.
The gapwise reduction uses the random-set/layer-cake identity; no floating LP
feasibility decision is used. Each pair is aggregated to at most (C+1)^2
parallel-edge sums. General repeated-leg incidences are deliberately rejected.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import itertools
import numpy as np


class Dinic:
    def __init__(self, n):
        self.g = [[] for _ in range(n)]

    def add(self, u, v, capacity):
        edge = [v, int(capacity), len(self.g[v])]
        self.g[u].append(edge)
        self.g[v].append([u, 0, len(self.g[u])-1])
        return len(self.g[u])-1

    def flow(self, source, sink, target):
        total = 0
        g = self.g
        while total < target:
            level = [-1] * len(g)
            level[source] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v, cap, _ in g[u]:
                    if cap and level[v] < 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[sink] < 0:
                break
            pos = [0] * len(g)

            def send(u, amount):
                if u == sink:
                    return amount
                while pos[u] < len(g[u]):
                    edge = g[u][pos[u]]
                    v, cap, reverse = edge
                    if cap and level[v] == level[u] + 1:
                        value = send(v, min(cap, amount))
                        if value:
                            edge[1] -= value
                            g[v][reverse][1] += value
                            return value
                    pos[u] += 1
                return 0

            while total < target:
                value = send(source, target-total)
                if not value:
                    break
                total += value
        return total


def bounded_circulation(n, arcs, witness=False):
    """Return feasibility, optionally integer flows in original arc order."""
    graph = Dinic(n+2)
    demand = [0]*n
    handles = []
    for u, v, lo, hi in arcs:
        if lo < 0 or hi < lo:
            return (False, None) if witness else False
        handle = graph.add(u, v, hi-lo)
        handles.append((u, handle, lo, hi-lo))
        demand[u] -= lo
        demand[v] += lo
    total = 0
    for u, amount in enumerate(demand):
        if amount > 0:
            graph.add(n, u, amount)
            total += amount
        elif amount < 0:
            graph.add(u, n+1, -amount)
    ok = graph.flow(n, n+1, total) == total
    if not witness:
        return ok
    values = [lo+capacity-graph.g[u][idx][1] for u, idx, lo, capacity in handles] if ok else None
    return ok, values


def mask_feasible(left_bounds, right_bounds, edges, flags, horizon, witness=False):
    """Strict degree mask on a bipartite box; last node each side is dummy.

    edges are (left,right,lo,hi). flags excludes the dummy nodes.
    All numeric capacities are integers in an arbitrary common duration unit.
    """
    C = len(left_bounds)-1
    if len(right_bounds) != C+1 or len(flags) != 2*C:
        raise ValueError('Inconsistent cell counts')
    strict = sum(flags)
    multiplier = max(1, strict)
    total = sum(hi for _, _, _, hi in edges)*multiplier
    source, sink = 2*(C+1), 2*(C+1)+1
    arcs = []
    for i, hi in enumerate(left_bounds):
        if i == C:
            lo, upper = 0, hi*multiplier
        elif flags[i]:
            lo, upper = horizon*multiplier+1, hi*multiplier
        else:
            lo, upper = 0, min(horizon, hi)*multiplier
        arcs.append((source, i, lo, upper))
    edge_offset = len(arcs)
    for i, j, lo, hi in edges:
        arcs.append((i, C+1+j, lo*multiplier, hi*multiplier))
    for j, hi in enumerate(right_bounds):
        if j == C:
            lo, upper = 0, hi*multiplier
        elif flags[C+j]:
            lo, upper = horizon*multiplier+1, hi*multiplier
        else:
            lo, upper = 0, min(horizon, hi)*multiplier
        arcs.append((C+1+j, sink, lo, upper))
    arcs.append((sink, source, 0, total))
    result = bounded_circulation(sink+1, arcs, witness)
    if not witness:
        return result
    ok, values = result
    return ok, ([Fraction(v, multiplier) for v in values[edge_offset:edge_offset+len(edges)]] if ok else None)


class ExactFlowOracle:
    def __init__(self, case, weights, grid='four', horizon=None):
        self.case = case
        self.C = len(weights)
        self.weights = tuple(Fraction(str(float(w))).limit_denominator(10**6) for w in weights)
        if sum(self.weights) != 1 or any(w < 0 for w in self.weights):
            raise ValueError('Weights must be normalized nonnegative rational numbers')
        self.W = [Fraction(0)]
        for w in self.weights:
            self.W.append(self.W[-1]+w)
        indices = {'four': list(range(6)), 'two': [0,2,3,5]}[grid]
        original_levels = [0., .05, .15, .30, .40, .50]
        self.levels = [Fraction(str(original_levels[h])) for h in indices]
        self.widths = [b-a for a,b in zip(self.levels, self.levels[1:])]
        self.H = len(self.widths)
        q = case.endpoints[:, indices]
        ratios = [[float(v).as_integer_ratio() for v in row] for row in q]
        # The released v12 domains and endpoint classes use Q <= L+TOL.
        # Treat that existing numeric RHS as the explicit common horizon.
        # This includes exact modal-boundary routes consistently, rather than
        # silently deleting them when dyadic sums differ from floating sums.
        self.reference_horizon = float(case.L)
        self.effective_horizon = float(case.L+1e-9) if horizon is None else float(horizon)
        hnum, hden = self.effective_horizon.as_integer_ratio()
        self.D = max([hden]+[den for row in ratios for _,den in row])
        self.L = hnum*(self.D//hden)
        self.q = [[num*(self.D//den) for num,den in row] for row in ratios]
        self.legs = []
        for row in case.incidence:
            ids = tuple(int(e) for e in row if e)
            if len(ids) != len(set(ids)):
                raise ValueError('Repeated primitive leg within a route is outside the network theorem')
            self.legs.append(ids)
        self.route_q = [[sum(self.q[e][h] for e in ids) for h in range(self.H+1)] for ids in self.legs]
        self.stats = {'pairs':0, 'flows':0, 'gap_cache_hits':0,
                      'forced_patterns':0, 'masks_rejected':0,
                      'analytic_gaps':0}
        self.gap_cache = {}

    def bounds(self):
        lower, upper = [], []
        for route, values in enumerate(self.route_q):
            if values[-1] > self.L:
                lower.append(np.nan); upper.append(np.nan)
                continue
            passed = next(h for h,v in enumerate(values) if v <= self.L)
            upper.append(float(self.levels[passed]) if route else 0.)
            lower.append(float(self.levels[max(0,passed-1)]) if route else 0.)
        return np.array(lower), np.array(upper)

    def _network(self, x, y, h):
        C = self.C
        left, right = {}, {}
        for labels, target in ((x,left),(y,right)):
            for cell, route in enumerate(labels):
                for e in self.legs[int(route)]:
                    if e in target:
                        raise ValueError('A primitive leg occurs in two cells of the same plan')
                    target[e] = cell
        grouped = {}
        for e in left.keys() | right.keys():
            edge = (left.get(e,C), right.get(e,C))
            lo, hi = grouped.get(edge,(0,0))
            grouped[edge] = (lo+self.q[e][h+1], hi+self.q[e][h])
        edges = [(i,j,lo,hi) for (i,j),(lo,hi) in grouped.items()]
        left_hi, right_hi = [0]*(C+1), [0]*(C+1)
        for i,j,lo,hi in edges:
            left_hi[i] += hi
            right_hi[j] += hi
        return left_hi, right_hi, edges

    def gap_fraction(self, x, y, h, witness=False):
        key = (tuple(x),tuple(y),h)
        if not witness and key in self.gap_cache:
            self.stats['gap_cache_hits'] += 1
            return self.gap_cache[key]
        routes = tuple(x)+tuple(y)
        flags = [0]*len(routes)
        active = []
        for i,route in enumerate(routes):
            lo, hi = self.route_q[int(route)][h+1], self.route_q[int(route)][h]
            if hi <= self.L:
                flags[i] = 0
            elif lo > self.L:
                flags[i] = 1
            else:
                active.append(i)
        # If uncertainty can change counts on only one side, every adverse
        # classification is simultaneous (take all upper or all lower ends).
        # Fixed classes remain fixed throughout the full box.
        active_left = {i for i in active if i < self.C}
        active_right = {i-self.C for i in active if i >= self.C}
        if not active_left or not active_right:
            f = flags.copy()
            for i in active_left:
                f[i] = 1
            value = self.W[sum(f[:self.C])]-self.W[sum(f[self.C:])]
            if not witness:
                self.stats['analytic_gaps'] += 1
                self.gap_cache[key] = value
                return value
        patterns = []
        for bits in itertools.product((0,1),repeat=len(active)):
            f = flags.copy()
            for i,b in zip(active,bits):
                f[i] = b
            value = self.W[sum(f[:self.C])]-self.W[sum(f[self.C:])]
            patterns.append((value,f))
        patterns.sort(key=lambda p:p[0],reverse=True)
        left_hi, right_hi, edges = self._network(x,y,h)
        if not witness and not any(i in active_left and j in active_right and hi > lo
                                   for i,j,lo,hi in edges):
            # No uncertain primitive quantity couples two active cells;
            # upper-left/lower-right endpoint choices are then compatible.
            value = patterns[0][0]
            self.stats['analytic_gaps'] += 1
            self.gap_cache[key] = value
            return value
        for value, f in patterns:
            # Identical routes must have identical classifications.
            if any(routes[i] == routes[j] and f[i] != f[j]
                   for i in range(len(routes)) for j in range(i)):
                self.stats['masks_rejected'] += 1
                continue
            self.stats['flows'] += 1
            ok = mask_feasible(left_hi,right_hi,edges,f,self.L)
            if ok:
                self.gap_cache[key] = value
                if witness:
                    _, edge_values = mask_feasible(left_hi,right_hi,edges,f,self.L,True)
                    return value, edges, edge_values, f
                return value
            self.stats['masks_rejected'] += 1
        raise AssertionError('No realizable violation mask')

    def pair_fraction(self, x, y):
        self.stats['pairs'] += 1
        if tuple(x) == tuple(y):
            return Fraction(0)
        return sum((width*self.gap_fraction(x,y,h) for h,width in enumerate(self.widths)),Fraction(0))

    def pair(self, x, y):
        return float(self.pair_fraction(x,y))

