"""Minimax planning from finitely many elicited endpoints.

Theory (main text, Section "Finite endpoint information"):

* EndpointGrid: elicited ceilings rho_1 < ... < rho_H < 1/2 (sentinel 1/2 is the
  representative duration). A cell with endpoint sums Q(rho_h) has the
  credibility interval (lower, upper] of Theorem 13.
* StaircaseCompletion / LowerStaircaseCompletion: the least- and most-favourable
  admissible completions of one leg. The upper one attains the upper bound of
  every cell of every plan at a fixed horizon (with a sufficiently long finite
  tail); the lower one approaches the lower bound on modal-feasible cells.
* CountRepresentation: OWA of grid-valued risks as a weighted sum of a concave
  function of the ceiling-violation counts.
* CountMILP: exact set-partitioning model of the minimax objective J^+ and of
  the grid ceiling sweep (HiGHS).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import time

import numpy as np

TOL = 1e-9


# ---------------------------------------------------------------------------
# Grid and cell intervals
# ---------------------------------------------------------------------------
class EndpointGrid:
    """Elicited ceilings with the representative sentinel 1/2."""

    def __init__(self, levels):
        levels = [float(r) for r in levels]
        if (len(levels) < 2 or not all(math.isfinite(r) for r in levels)
                or levels != sorted(set(levels)) or levels[-1] != 0.5):
            raise ValueError('levels must increase strictly and end with the sentinel 1/2')
        if levels[0] < 0:
            raise ValueError('ceilings must be nonnegative')
        self.levels = tuple(levels)
        # possible upper values of a nonempty cell, in increasing order
        self.values = tuple(sorted(set(levels) | {1.0}))

    def upper(self, Q, L):
        """Smallest grid ceiling met by the endpoint sums Q (1 if none)."""
        for r, q in zip(self.levels, Q):
            if q <= L + TOL:
                return r
        return 1.0

    def lower(self, Q, L):
        out = 0.0
        for r, q in zip(self.levels, Q):
            if q > L + TOL:
                out = r
        return out

    def violation_counts(self, upper_risks):
        """N_h = #cells with upper risk >= v_h for the positive values v_h."""
        positive = [v for v in self.values if v > 0]
        return [sum(1 for x in upper_risks if x >= v - 1e-12) for v in positive]

    def gap(self):
        pts = sorted(set((0.0, 1.0) + self.levels))
        return max(b - a for a, b in zip(pts, pts[1:]))


# ---------------------------------------------------------------------------
# Completions (polymorphic memberships of one leg)
# ---------------------------------------------------------------------------
class LegCompletion(ABC):
    """Complete nonnegative membership consistent with the endpoints
    q(rho_h) of one leg (nonincreasing in h, q(1/2) = representative value).
    Closed step intervals take their maximum at shared endpoints, making the
    membership upper semicontinuous. It is zero outside these finite intervals.

    If rho_1 > 0, an upper completion needs a finite tail long enough for the
    fixed planning horizon. The default is only a convenient finite support;
    it does not certify every possible horizon. Taking tail > max(q_1, L)
    suffices simultaneously for all nonempty cells with nonnegative durations.
    """

    def __init__(self, grid: EndpointGrid, endpoints, tail=None):
        self.grid = grid
        self.q = tuple(float(x) for x in endpoints)
        if (len(self.q) != len(grid.levels) or not all(math.isfinite(x) for x in self.q)
                or any(x < 0 for x in self.q)):
            raise ValueError('one finite nonnegative endpoint is required per grid level')
        if any(a < b - TOL for a, b in zip(self.q, self.q[1:])):
            raise ValueError('endpoints must be nonincreasing in the ceiling')
        # finite end of the support when rho_1 > 0 (compact cuts)
        self.tail = 10.0 * abs(self.q[0]) + 10.0 if tail is None else float(tail)
        if not math.isfinite(self.tail) or self.tail < self.q[0]:
            raise ValueError('tail must be finite and no smaller than the largest endpoint')

    @abstractmethod
    def step_value(self, h: int) -> float:
        """Membership on (q(rho_h), q(rho_{h-1})], h = 1..H; h = 0 is the tail."""

    def breakpoints(self):
        lv, q = self.grid.levels, self.q
        out = [(q[-1], q[-1], 1.0)]
        for h in range(len(lv) - 1, 0, -1):
            out.append((q[h], q[h - 1], self.step_value(h)))
        if lv[0] > 0:
            out.append((q[0], self.tail, self.step_value(0)))
        return out

    def membership(self, x):
        best = 0.0
        for lo, hi, val in self.breakpoints():
            if lo <= x <= hi:
                best = max(best, val)
        return best

    def cut(self, alpha, strict=False):
        """Endpoints of a positive alpha-cut, or its strict-cut closure.

        A strict cut at zero is allowed for the support endpoint. A closed
        zero-cut is unbounded for any membership and is deliberately excluded.
        """
        if not math.isfinite(alpha) or not (0 <= alpha < 1 if strict else 0 < alpha <= 1):
            raise ValueError('invalid membership level')
        spans = [(lo, hi) for lo, hi, value in self.breakpoints()
                 if (value > alpha if strict else value >= alpha)]
        if not spans:
            raise ValueError('empty cut')
        return min(lo for lo, _ in spans), max(hi for _, hi in spans)

    def planning_endpoint(self, rho):
        """Evaluate the strong endpoint directly from this full membership."""
        if not math.isfinite(rho) or not 0 <= rho < 1:
            raise ValueError('ceiling must lie in [0,1)')
        return self.cut(2 * rho, strict=True)[1] if rho < .5 else self.cut(2 - 2 * rho)[0]


class StaircaseCompletion(LegCompletion):
    """Least favourable completion: mu = 2 rho_h on (q(rho_h), q(rho_{h-1})]."""

    def step_value(self, h):
        return 2 * self.grid.levels[h]


class LowerStaircaseCompletion(LegCompletion):
    """Lower completion on modal-feasible cells, with singleton core {b}.

    The singleton is essential: a core [0,b] changes the known endpoint
    q(1/2)=ell(1) to zero. No tail beyond q_1 is needed for this completion.
    """

    def __init__(self, grid, endpoints, eps, tail=None):
        super().__init__(grid, endpoints, tail)
        gaps = [b - a for a, b in zip(grid.levels, grid.levels[1:])]
        if not 0 < eps < min(gaps) / 2:
            raise ValueError('eps must be below half the smallest grid gap')
        self.eps = float(eps)

    def step_value(self, h):
        return 2 * self.eps if h == 0 else 2 * self.grid.levels[h - 1] + 2 * self.eps

    def breakpoints(self):
        return [(self.q[-1], self.q[-1], 1.0)] + [
            (self.q[h], self.q[h - 1], self.step_value(h))
            for h in range(len(self.q) - 1, 0, -1)]


class SupMinSum:
    """Full Pos/Nec credibility from level-set addition of step memberships.

    Finite positive membership levels determine both sides of the sum's cuts.
    Neither the strong-endpoint equivalent nor the interval bound is used.
    """

    def __init__(self, legs):
        self.legs = list(legs)

    def possibility_pair(self, L):
        """Return Pos(sum > L), Pos(sum <= L), respecting equality at L."""
        if not math.isfinite(L):
            raise ValueError('horizon must be finite')
        if not self.legs:
            return (1.0, 0.0) if L < 0 else (0.0, 1.0)
        values = sorted({v for leg in self.legs for _, _, v in leg.breakpoints()})
        above = below = 0.0
        for v in values:
            cuts = [leg.cut(v) for leg in self.legs]
            left = sum(a for a, _ in cuts)
            right = sum(b for _, b in cuts)
            if right > L + TOL:
                above = max(above, v)
            if left <= L + TOL:
                below = max(below, v)
        return above, below

    def credibility(self, L):
        above, below = self.possibility_pair(L)
        return .5 * (above + 1.0 - below)

    def right_credibility(self, L):
        """Backward-compatible entry point; now also valid left of the core."""
        return self.credibility(L)


# ---------------------------------------------------------------------------
# Count representation of the OWA
# ---------------------------------------------------------------------------
class OWAProfile:
    def __init__(self, slots, weights=None):
        if not isinstance(slots, (int, np.integer)) or isinstance(slots, bool) or slots < 1:
            raise ValueError('slots must be a positive integer')
        self.slots = slots
        if weights is None:
            weights = [2.0 * (slots - j) / (slots * (slots + 1)) for j in range(slots)]
        self.w = np.asarray(weights, dtype=float)
        if (self.w.ndim != 1 or len(self.w) != slots or not np.all(np.isfinite(self.w))
                or np.any(self.w < 0) or abs(self.w.sum() - 1) > 1e-12
                or np.any(np.diff(self.w) > 1e-15)):
            raise ValueError('one finite nonnegative normalized nonincreasing weight is required per slot')
        self.W = np.concatenate([[0.0], np.cumsum(self.w)])   # W(n), n = 0..slots

    def owa(self, risks):
        risks = list(risks)
        if len(risks) > self.slots or any(not math.isfinite(x) or not 0 <= x <= 1 for x in risks):
            raise ValueError('at most slots finite risks in [0,1] are required')
        x = sorted(risks + [0.0] * (self.slots - len(risks)), reverse=True)
        return float(np.dot(self.w, x))


class CountRepresentation:
    """FFF^+ = sum_h (v_h - v_{h-1}) W(N_h)."""

    def __init__(self, grid: EndpointGrid, profile: OWAProfile):
        self.grid, self.profile = grid, profile
        vals = [0.0] + [v for v in grid.values if v > 0]
        self.steps = [b - a for a, b in zip(vals, vals[1:])]

    def value(self, counts):
        return float(sum(d * self.profile.W[n] for d, n in zip(self.steps, counts)))


# ---------------------------------------------------------------------------
# Columns and exact models
# ---------------------------------------------------------------------------
@dataclass
class EndpointColumns:
    """Feasible routes: subset mask, modal workload, endpoint sums per level."""
    masks: np.ndarray
    workload: np.ndarray
    Q: np.ndarray            # (columns, levels)
    upper: np.ndarray
    lower: np.ndarray
    n: int


class ColumnBuilder:
    """Homothetic legs: Q_h = T k_t(h) + V k_s(h) on the shortest modal tour."""

    def __init__(self, T, V, grid: EndpointGrid, kt, ks, L, optimistic=0.7, representative=False):
        self.T, self.V, self.grid, self.L = np.asarray(T), np.asarray(V), grid, float(L)
        self.kt, self.ks, self.optimistic = np.asarray(kt), np.asarray(ks), optimistic
        self.representative = representative

    def build(self, n):
        """Optimistic feasibility; with representative=True also Q(1/2) <= L (upper risk <= 1/2)."""
        masks = np.arange(1, 1 << n)
        T, V = self.T[masks], self.V[masks]
        factor = 1.0 if self.representative else self.optimistic
        keep = factor * (T + V) <= self.L + TOL
        masks, T, V = masks[keep], T[keep], V[keep]
        Q = T[:, None] * self.kt[None, :] + V[:, None] * self.ks[None, :]
        up = np.array([self.grid.upper(row, self.L) for row in Q])
        lo = np.array([self.grid.lower(row, self.L) for row in Q])
        return EndpointColumns(masks, T + V, Q, up, lo, n)


class CountMILP:
    """min alpha*lambda/lambda0 + (1-alpha) sum_h d_h W(N_h) over set partitions.

    N_h = sum_r [upper_r >= v_h] x_r is linear in the route variables, and
    W(N_h) is expressed exactly with one binary per possible count.
    """

    def __init__(self, cols: EndpointColumns, slots, grid, profile, time_limit=3600.0):
        self.c, self.slots, self.grid, self.profile = cols, slots, grid, profile
        self.rep = CountRepresentation(grid, profile)
        self.positive = [v for v in grid.values if v > 0]
        self.time_limit = time_limit
        self.members = [[i for i in range(cols.n) if (int(m) >> i) & 1] for m in cols.masks]

    def solve(self, alpha, scale, ceiling=None, workload_cap=None, risk_weight=None):
        out = self._solve(alpha, scale, ceiling, workload_cap, risk_weight, True)
        if out['status'] not in ('Optimal', 'Infeasible', 'Time limit reached'):
            out = self._solve(alpha, scale, ceiling, workload_cap, risk_weight, False)
        return out

    def _solve(self, alpha, scale, ceiling, workload_cap, risk_weight, presolve):
        import highspy
        c, n, S = self.c, self.c.n, self.slots
        use = np.ones(len(c.masks), bool) if ceiling is None else c.upper <= ceiling + 1e-12
        idx = np.nonzero(use)[0]
        if idx.size == 0 or np.bitwise_or.reduce(c.masks[idx]) != (1 << n) - 1:
            return dict(status='Infeasible')
        G = len(self.positive)
        nx, nz = idx.size, G * (S + 1)
        h = highspy.Highs()
        h.setOptionValue('output_flag', False)
        h.setOptionValue('threads', 1)
        h.setOptionValue('time_limit', float(self.time_limit))
        h.setOptionValue('mip_rel_gap', 0.0)
        h.setOptionValue('mip_abs_gap', 1e-10)
        if not presolve:
            h.setOptionValue('presolve', 'off')
        inf = highspy.kHighsInf
        rw = (1 - alpha) if risk_weight is None else risk_weight
        cost = np.concatenate([alpha * c.workload[idx] / scale,
                               np.concatenate([[rw * d * self.profile.W[k] for k in range(S + 1)]
                                               for d in self.rep.steps])])
        lower, upper = np.zeros(nx + nz), np.ones(nx + nz)
        h.addVars(nx + nz, lower, upper)
        h.changeColsCost(nx + nz, np.arange(nx + nz, dtype=np.int32), cost)
        h.changeColsIntegrality(nx + nz, np.arange(nx + nz, dtype=np.int32),
                                np.array([highspy.HighsVarType.kInteger] * (nx + nz)))
        # partition rows
        rows = [[] for _ in range(n)]
        for j, r in enumerate(idx):
            for i in self.members[r]:
                rows[i].append(j)
        for i in range(n):
            h.addRow(1.0, 1.0, len(rows[i]), np.array(rows[i], dtype=np.int32), np.ones(len(rows[i])))
        h.addRow(0.0, float(S), nx, np.arange(nx, dtype=np.int32), np.ones(nx))
        for g, v in enumerate(self.positive):
            sel = np.nonzero(c.upper[idx] >= v - 1e-12)[0]
            zi = nx + g * (S + 1) + np.arange(S + 1)
            # sum_r a_r x_r - sum_k k z_k = 0 ; sum_k z_k = 1
            h.addRow(0.0, 0.0, sel.size + S + 1, np.concatenate([sel, zi]).astype(np.int32),
                     np.concatenate([np.ones(sel.size), -np.arange(S + 1, dtype=float)]))
            h.addRow(1.0, 1.0, S + 1, zi.astype(np.int32), np.ones(S + 1))
        if workload_cap is not None:
            h.addRow(-inf, float(workload_cap), nx, np.arange(nx, dtype=np.int32), c.workload[idx])
        t0 = time.perf_counter()
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        secs = time.perf_counter() - t0
        if status != 'Optimal':
            return dict(status=status, seconds=secs)
        x = np.array(h.getSolution().col_value)
        chosen = idx[np.nonzero(x[:nx] > 0.5)[0]]
        return self.record(chosen, alpha, scale, status, secs)

    def record(self, chosen, alpha, scale, status='Optimal', secs=0.0):
        c = self.c
        up = [float(c.upper[r]) for r in chosen]
        lo = [float(c.lower[r]) for r in chosen]
        counts = self.grid.violation_counts(up)
        fff_sorted = self.profile.owa(up)
        fff_counts = self.rep.value(counts)
        assert abs(fff_sorted - fff_counts) <= 1e-9, (fff_sorted, fff_counts)
        workload = float(sum(c.workload[r] for r in chosen))
        return dict(status=status, seconds=secs, masks=[int(c.masks[r]) for r in chosen],
                    workload=workload, fff_upper=fff_sorted, fff_lower=self.profile.owa(lo),
                    max_upper=max(up), counts=counts,
                    J=alpha * workload / scale + (1 - alpha) * fff_sorted)


class GridSweep:
    """Every grid ceiling solved exactly: minimum workload, then minimum FFF^+."""

    def __init__(self, milp: CountMILP, tiebreak_limit=60.0):
        self.milp, self.tiebreak_limit = milp, tiebreak_limit

    def plans(self, scale):
        out = {}
        for ceiling in self.milp.grid.values:
            first = self.milp.solve(1.0, scale, ceiling=ceiling)
            if first['status'] not in ('Optimal', 'Infeasible'):
                raise RuntimeError(first['status'])
            if first['status'] != 'Optimal':
                out[ceiling] = first
                continue
            cap = first['workload'] * (1 + 1e-9) + 1e-9
            limit, self.milp.time_limit = self.milp.time_limit, self.tiebreak_limit
            second = self.milp.solve(0.0, scale, ceiling=ceiling, workload_cap=cap, risk_weight=1.0)
            self.milp.time_limit = limit
            secondary_status = second['status']
            secondary_seconds = second.get('seconds', 0.0)
            if second['status'] == 'Optimal':
                assert second['workload'] <= cap + 1e-7
                second['tiebreak'] = 'optimal'
            else:
                second = dict(first, tiebreak='uncertified; first minimum-workload plan kept')
            second['secondary_status'] = secondary_status
            second['secondary_seconds'] = secondary_seconds
            second['seconds'] = first['seconds'] + secondary_seconds
            out[ceiling] = second
        return out

    @staticmethod
    def best(plans, alpha, scale):
        feas = [p for p in plans.values() if p['status'] == 'Optimal']
        vals = [alpha * p['workload'] / scale + (1 - alpha) * p['fff_upper'] for p in feas]
        k = int(np.argmin(vals))
        return feas[k], vals[k]


class Bottleneck:
    """Certified min over partitions into <= slots routes of the max modal route duration."""

    def __init__(self, masks, duration, n, slots, time_limit=3600.0):
        self.masks, self.duration = np.asarray(masks), np.asarray(duration)
        self.n, self.slots, self.time_limit = n, slots, time_limit
        self.members = [[i for i in range(n) if (int(m) >> i) & 1] for m in self.masks]

    def feasible(self, theta):
        status = self._feasible(theta, True)
        if status not in ('Optimal', 'Infeasible'):
            status = self._feasible(theta, False)
        if status not in ('Optimal', 'Infeasible'):
            raise RuntimeError(status)
        return status == 'Optimal'

    def _feasible(self, theta, presolve):
        import highspy
        idx = np.nonzero(self.duration <= theta + 1e-12)[0]
        if idx.size == 0 or np.bitwise_or.reduce(self.masks[idx]) != (1 << self.n) - 1:
            return 'Infeasible'
        h = highspy.Highs()
        h.setOptionValue('output_flag', False)
        h.setOptionValue('threads', 1)
        h.setOptionValue('time_limit', float(self.time_limit))
        if not presolve:
            h.setOptionValue('presolve', 'off')
        h.addVars(idx.size, np.zeros(idx.size), np.ones(idx.size))
        h.changeColsIntegrality(idx.size, np.arange(idx.size, dtype=np.int32),
                                np.array([highspy.HighsVarType.kInteger] * idx.size))
        rows = [[] for _ in range(self.n)]
        for j, r in enumerate(idx):
            for i in self.members[r]:
                rows[i].append(j)
        for i in range(self.n):
            h.addRow(1.0, 1.0, len(rows[i]), np.array(rows[i], dtype=np.int32), np.ones(len(rows[i])))
        h.addRow(0.0, float(self.slots), idx.size, np.arange(idx.size, dtype=np.int32), np.ones(idx.size))
        h.run()
        return h.modelStatusToString(h.getModelStatus())

    def value(self, ratio=1.05):
        """Geometric search from the largest single-task route, then bisection
        over the distinct route durations of the last bracket."""
        single = max(self.duration[self.masks == (1 << i)][0] for i in range(self.n))
        lo_val, hi_val = None, single
        while not self.feasible(hi_val):
            lo_val, hi_val = hi_val, hi_val * ratio
        cand = self.duration[(self.duration <= hi_val + 1e-12)]
        if lo_val is not None:
            cand = cand[cand > lo_val + 1e-12]
        cand = np.unique(np.concatenate([cand, [hi_val]]))
        lo, hi = 0, len(cand) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.feasible(cand[mid]):
                hi = mid
            else:
                lo = mid + 1
        return float(cand[lo])
