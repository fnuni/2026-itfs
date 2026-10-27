"""
Risk-aware large neighbourhood search, parameterised by the search aggregator.

The solver is held FIXED across aggregators so that any difference in the
reported (common-scale) indicators is attributable to the aggregation operator
alone -- the controlled component comparison requested in the review.
"""

from __future__ import annotations

import math
import random

from aggregators import FFF_REPORT, max_risk, gini, jain, cvar
from problem import Solution


class EpsilonArchive:
    """epsilon-nondominated archive on (makespan, reporting FFF), both minimised."""

    def __init__(self, eps=(1.0, 0.01)):
        self.eps = eps
        self.items = {}

    def _box(self, point):
        return tuple(int(v / e) for v, e in zip(point, self.eps))

    def add(self, point):
        box = self._box(point)
        for b in list(self.items):
            if b != box and all(x <= y for x, y in zip(b, box)):
                return False
        for b in list(self.items):
            if b != box and all(y <= x for x, y in zip(b, box)):
                del self.items[b]
        cur = self.items.get(box)
        if cur is None or sum(point) < sum(cur):
            self.items[box] = point
        return True

    def points(self):
        return list(self.items.values())


class _State:
    """Incrementally maintained per-cell evaluation of a solution."""

    __slots__ = ("inst", "sol", "risk", "modal", "opt")

    def __init__(self, inst, sol):
        self.inst, self.sol = inst, sol
        self.risk = [[0.0] * inst.P for _ in range(inst.K)]
        self.modal = [[0.0] * inst.P for _ in range(inst.K)]
        self.opt = [[0.0] * inst.P for _ in range(inst.K)]
        for k in range(inst.K):
            for h in range(inst.P):
                self.refresh(k, h)

    def refresh(self, k, h):
        dur, modal = self.inst.cell_duration(self.sol.cells[k][h])
        self.risk[k][h] = dur.overtime_risk(self.inst.L)
        self.modal[k][h] = modal
        self.opt[k][h] = dur.support[0] if self.sol.cells[k][h] else 0.0

    def cell_value(self, k, h, seq):
        dur, modal = self.inst.cell_duration(seq)
        opt = dur.support[0] if seq else 0.0
        return dur.overtime_risk(self.inst.L), modal, opt

    @property
    def violation(self):
        """Sum of relative optimistic-duration overruns; 0 for a feasible plan."""
        L = self.inst.L
        return sum(max(0.0, o - L) for row in self.opt for o in row) / L

    @property
    def R(self):
        return [row[:] for row in self.risk]

    @property
    def makespan(self):
        return sum(v for row in self.modal for v in row)

    def clone(self):
        s = _State.__new__(_State)
        s.inst = self.inst
        s.sol = self.sol.clone()
        s.risk = [row[:] for row in self.risk]
        s.modal = [row[:] for row in self.modal]
        s.opt = [row[:] for row in self.opt]
        return s


def two_opt_cell(inst, seq):
    """First-improvement 2-opt on one cell under modal travel times (in place)."""
    m = len(seq)
    if m < 4:
        return seq
    d = inst.modal_travel
    improved = True
    while improved:
        improved = False
        for i in range(0, m - 1):
            a = seq[i - 1] if i else 0
            b = seq[i]
            for j in range(i + 1, m):
                c = seq[j]
                e = seq[j + 1] if j + 1 < m else 0
                if d(a, c) + d(b, e) < d(a, b) + d(c, e) - 1e-9:
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    improved = True
                    b = seq[i]
    return seq


def polish(inst, sol, cells=None):
    """Apply two_opt_cell to the given cells (all cells when None)."""
    cells = cells if cells is not None else [(k, h) for k in range(sol.K)
                                              for h in range(sol.P)]
    for k, h in cells:
        two_opt_cell(inst, sol.cells[k][h])
    return sol


class RALNS:
    """Risk-aware LNS with boundary-aware, risk-proportional and inter-vehicle
    destroy operators, adaptive operator weights and simulated annealing."""

    def __init__(self, instance, aggregator, alpha=0.3, iterations=300,
                 seed=0, destroy_frac=(0.10, 0.25), t0=0.04, cooling=0.985,
                 n_candidate_positions=8):
        self.inst = instance
        self.agg = aggregator
        self.alpha = alpha
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.destroy_frac = destroy_frac
        self.t0, self.cooling = t0, cooling
        self.ncand = n_candidate_positions
        self.op_weights = [1.0, 1.0, 1.0]
        self.op_scores = [0.0, 0.0, 0.0]
        self.op_counts = [1e-9, 1e-9, 1e-9]
        self._lambda_scale = 1.0
        self._eval0 = instance.cell_evals

    @property
    def evaluations(self):
        """Fuzzy cell evaluations spent by this solver (the shared budget unit)."""
        return self.inst.cell_evals - self._eval0

    # --- objective --------------------------------------------------------
    INFEASIBILITY_PENALTY = 10.0

    def _scalar(self, R, makespan, violation=0.0):
        return (self.alpha * (makespan / self._lambda_scale)
                + (1 - self.alpha) * self.agg(R)
                + self.INFEASIBILITY_PENALTY * violation)

    # --- construction -----------------------------------------------------
    def initial(self):
        inst = self.inst
        sol = Solution(inst.K, inst.P)
        unvisited = set(range(1, inst.n + 1))
        per_cell = math.ceil(inst.n / (inst.K * inst.P))
        for k in range(inst.K):
            for h in range(inst.P):
                cur = 0
                for _ in range(per_cell):
                    if not unvisited:
                        break
                    nxt = min(unvisited, key=lambda j: inst.modal_travel(cur, j))
                    sol.cells[k][h].append(nxt)
                    unvisited.discard(nxt)
                    cur = nxt
        for t in list(unvisited):
            sol.cells[self.rng.randrange(inst.K)][self.rng.randrange(inst.P)].append(t)
        return sol

    # --- destroy operators ------------------------------------------------
    def _destroy_boundary(self, sol, q):
        pool = []
        for k in range(sol.K):
            for h in range(sol.P):
                c = sol.cells[k][h]
                pool.extend(c[:2] + c[-2:])
        self.rng.shuffle(pool)
        return list(dict.fromkeys(pool))[:q]

    def _destroy_risk(self, sol, R, q):
        order = sorted(((R[k][h], k, h) for k in range(sol.K) for h in range(sol.P)),
                       reverse=True)
        picked = []
        for _, k, h in order:
            c = list(sol.cells[k][h])
            self.rng.shuffle(c)
            picked.extend(c)
            if len(picked) >= q:
                break
        return picked[:q]

    def _destroy_intervehicle(self, sol, R, q):
        if sol.K < 2:
            return self._destroy_risk(sol, R, q)
        h = self.rng.randrange(sol.P)
        ks = sorted(range(sol.K), key=lambda k: R[k][h], reverse=True)[:2]
        picked = []
        for k in ks:
            c = list(sol.cells[k][h])
            self.rng.shuffle(c)
            picked.extend(c[: max(1, q // 2)])
        return picked[:q]

    def _select_operator(self):
        tot = sum(self.op_weights)
        u = self.rng.random() * tot
        acc = 0.0
        for i, w in enumerate(self.op_weights):
            acc += w
            if u <= acc:
                return i
        return len(self.op_weights) - 1

    # --- repair (incremental, single-cell delta evaluation) ---------------
    def _repair(self, st, removed):
        inst = self.inst
        self.rng.shuffle(removed)
        cells = [(k, h) for k in range(inst.K) for h in range(inst.P)]
        touched = set()
        for t in removed:
            best, best_val, best_triple = None, float("inf"), None
            for k, h in cells:
                c = st.sol.cells[k][h]
                m = len(c) + 1
                positions = range(m) if m <= self.ncand else self.rng.sample(range(m), self.ncand)
                base = (st.risk[k][h], st.modal[k][h], st.opt[k][h])
                for pos in positions:
                    c.insert(pos, t)
                    r_new, m_new, o_new = st.cell_value(k, h, c)
                    c.pop(pos)
                    st.risk[k][h], st.modal[k][h], st.opt[k][h] = r_new, m_new, o_new
                    val = self._scalar(st.risk, st.makespan, st.violation)
                    st.risk[k][h], st.modal[k][h], st.opt[k][h] = base
                    if val < best_val:
                        best_val, best = val, (k, h, pos)
                        best_triple = (r_new, m_new, o_new)
            k, h, pos = best
            st.sol.cells[k][h].insert(pos, t)
            st.risk[k][h], st.modal[k][h], st.opt[k][h] = best_triple
            touched.add((k, h))
        for k, h in touched:              # intra-cell 2-opt polish, then refresh
            if getattr(self, "polish_on", True):
                two_opt_cell(inst, st.sol.cells[k][h])
                st.refresh(k, h)
        return st

    # --- main loop --------------------------------------------------------
    def run(self, archive=None):
        inst = self.inst
        st = _State(inst, self.initial())
        self._lambda_scale = max(st.makespan, 1.0)
        cur_val = self._scalar(st.risk, st.makespan, st.violation)
        best_st, best_val = st.clone(), cur_val
        archive = archive if archive is not None else EpsilonArchive()
        if st.violation <= 1e-9:
            archive.add((st.makespan, FFF_REPORT(st.risk)))
        temp = self.t0
        for it in range(self.iterations):
            q = max(2, int(self.rng.uniform(*self.destroy_frac) * inst.n))
            op = self._select_operator()
            cand = st.clone()
            if op == 0:
                removed = self._destroy_boundary(cand.sol, q)
            elif op == 1:
                removed = self._destroy_risk(cand.sol, cand.risk, q)
            else:
                removed = self._destroy_intervehicle(cand.sol, cand.risk, q)
            if not removed:
                continue
            touched = [(k, h) for k in range(inst.K) for h in range(inst.P)
                       if any(t in removed for t in cand.sol.cells[k][h])]
            cand.sol.remove(removed)
            for k, h in touched:
                cand.refresh(k, h)
            cand = self._repair(cand, list(removed))
            cval = self._scalar(cand.risk, cand.makespan, cand.violation)
            if cand.violation <= 1e-9:
                archive.add((cand.makespan, FFF_REPORT(cand.risk)))
            self.op_counts[op] += 1
            if cval < cur_val or self.rng.random() < math.exp(
                    -(cval - cur_val) / max(temp, 1e-9)):
                st, cur_val = cand, cval
                self.op_scores[op] += 1.0 if cval < best_val else 0.3
            if cval < best_val:
                best_st, best_val = cand.clone(), cval
            temp *= self.cooling
            if (it + 1) % 25 == 0:
                self.op_weights = [max(0.05, 0.7 * w + 0.3 * (s / c))
                                   for w, s, c in zip(self.op_weights, self.op_scores,
                                                      self.op_counts)]
                self.op_scores = [0.0, 0.0, 0.0]
                self.op_counts = [1e-9, 1e-9, 1e-9]
        R = best_st.R
        return dict(R=R, makespan=best_st.makespan, fff=FFF_REPORT(R),
                    violation=best_st.violation, feasible=best_st.violation <= 1e-9,
                    max_risk=max_risk(R), gini=gini(R), jain=jain(R),
                    cvar=cvar(R), archive=archive, evaluations=self.evaluations)


def hypervolume_2d(points, ref):
    """HV of a 2-D minimisation front w.r.t. reference point ref."""
    pts = sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1])
    if not pts:
        return 0.0
    front, best_y = [], float("inf")
    for x, y in pts:
        if y < best_y:
            front.append((x, y))
            best_y = y
    hv = 0.0
    for i, (x, y) in enumerate(front):
        x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
        hv += (x_next - x) * (ref[1] - y)
    return hv
