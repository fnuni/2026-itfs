"""
Certified optima of the credibilistic ceiling problem on small instances.

By the crisp routing equivalent (crisp_equivalent.py) a cell visiting the task
set S along a tour of modal travel time T and modal service time V satisfies
the ceiling r_bar iff

        f_t(r_bar) * T + f_s(r_bar) * V <= L,

where f_t, f_s are the planning durations of a unit-modal travel and service
leg.  These multipliers exist whenever every leg membership is a scaled copy
of one membership per leg type, which holds for all instance families of the
study (triangular, trapezoidal, Gaussian and panel envelopes).  Because
f_t > 0, the tour of minimum modal travel time minimises both the cost and the
planning duration of S.  Hence:

  1. SubsetTSP computes the minimum-travel closed tour T*(S) of every subset by
     Held--Karp dynamic programming (vectorised over subsets of equal size);
  2. ExactCeilingSolver keeps the subsets that satisfy the inequality above and
     solves the set-partitioning model

        min  sum_S T*(S) x_S
        s.t. sum_{S ∋ i} x_S = 1   for every task i,
             sum_S x_S <= K * P,   x_S in {0, 1},

     with HiGHS.  Infeasibility certifies that the ceiling is unattainable;
     otherwise the optimum is the minimum total modal workload (plus the
     constant total service time).
  3. min_max_risk() returns the exact smallest attainable worst-cell
     credibility (bottleneck set partitioning by bisection on thresholds).

Returned plans are re-evaluated with the fuzzy evaluator of problem.Instance.
"""

from __future__ import annotations

import time

import numpy as np

from crisp_equivalent import CredibilisticCeiling
from fuzzy_core import TriangularFN
from problem import Solution


class SubsetTSP:
    """Held--Karp minimum closed tours from the depot for all task subsets."""

    def __init__(self, D: np.ndarray):
        # D: (n+1) x (n+1) modal travel times, node 0 = depot
        self.D = np.asarray(D, dtype=float)
        self.n = n = self.D.shape[0] - 1
        if n > 18:
            raise ValueError("SubsetTSP is intended for n <= 18 tasks")
        N = 1 << n
        INF = np.inf
        dp = np.full((N, n), INF)
        par = np.full((N, n), -1, dtype=np.int8)
        for j in range(n):
            dp[1 << j, j] = self.D[0, j + 1]
        Dt = self.D[1:, 1:]                                  # task-task block
        masks = np.arange(N, dtype=np.int64)
        popc = np.array([bin(m).count("1") for m in range(N)], dtype=np.int8)
        for k in range(2, n + 1):
            layer = masks[popc == k]
            for j in range(n):
                bit = 1 << j
                sel = layer[(layer & bit) != 0]
                if sel.size == 0:
                    continue
                prev = sel ^ bit
                cand = dp[prev] + Dt[:, j][None, :]          # (|sel|, n) over i
                arg = np.argmin(cand, axis=1)
                dp[sel, j] = cand[np.arange(sel.size), arg]
                par[sel, j] = arg
        back = dp + self.D[1:, 0][None, :]
        self.best = np.min(back, axis=1)
        self.best[0] = 0.0
        self.last = np.argmin(back, axis=1)
        self.par = par
        self.popcount = popc

    def tour(self, mask: int) -> list[int]:
        """Task ids (1..n) of the optimal tour over `mask`, depot excluded."""
        if mask == 0:
            return []
        seq = []
        j = int(self.last[mask])
        m = int(mask)
        while True:
            seq.append(j + 1)
            if m == (1 << j):
                break
            i = int(self.par[m, j])
            m ^= (1 << j)
            j = i
        return seq[::-1]


class ExactCeilingSolver:
    """Certified optimum of min workload s.t. max_kh r_kh <= r_bar (small n)."""

    def __init__(self, instance, time_limit: float = 600.0):
        self.inst = instance
        n = instance.n
        D = np.array([[instance.modal_travel(i, j) for j in range(n + 1)]
                      for i in range(n + 1)])
        t0 = time.perf_counter()
        self.tsp = SubsetTSP(D)
        self.tsp_seconds = time.perf_counter() - t0
        N = 1 << n
        serv = np.zeros(N)
        sv = np.array(instance.service_modal[1:], dtype=float)
        for m in range(1, N):
            low = m & -m
            serv[m] = serv[m ^ low] + sv[low.bit_length() - 1]
        self.serv = serv
        self.slots = instance.K * instance.P
        self.time_limit = time_limit
        f = instance.factory
        self._mk_t = getattr(f, "make_travel", f.make)
        self._mk_s = getattr(f, "make_service", f.make)
        self._members = None

    # --- per-subset quantities --------------------------------------------
    def multipliers(self, ceiling: CredibilisticCeiling) -> tuple[float, float]:
        return (ceiling.planning_duration(self._mk_t(1.0)),
                ceiling.planning_duration(self._mk_s(1.0)))

    def subset_risk(self) -> np.ndarray:
        """Overtime credibility of every subset served along its best tour."""
        L = self.inst.L
        lt, ls = self._mk_t(1.0), self._mk_s(1.0)
        if isinstance(lt, TriangularFN) and isinstance(ls, TriangularFN):
            # vectorised closed form (Proposition: TFN credibility map)
            T, V = self.tsp.best, self.serv
            a = T * lt.a + V * ls.a
            b = T * lt.b + V * ls.b
            c = T * lt.c + V * ls.c
            with np.errstate(divide="ignore", invalid="ignore"):
                right = np.where(c > b, (c - L) / (2.0 * (c - b)), 0.0)
                left = np.where(b > a, 1.0 - (L - a) / (2.0 * (b - a)), 1.0)
            out = np.where(L >= c, 0.0,
                           np.where(L >= b, right, np.where(L > a, left, 1.0)))
            out[0] = 0.0
            return np.clip(out, 0.0, 1.0)
        out = np.zeros(1 << self.inst.n)
        for m in range(1, 1 << self.inst.n):
            dur = self._mk_t(float(self.tsp.best[m])) + self._mk_s(float(self.serv[m]))
            out[m] = dur.overtime_risk(L)
        return out

    # --- set partitioning with HiGHS --------------------------------------
    def _members_of(self):
        if self._members is None:
            n = self.inst.n
            self._members = [[i for i in range(n) if (m >> i) & 1]
                             for m in range(1 << n)]
        return self._members

    def _partition(self, columns: np.ndarray, cost, feasibility_only=False):
        """Set-partitioning MIP; if HiGHS reports neither optimality nor
        infeasibility (e.g. a presolve inconsistency flagged as "Solve error"),
        the model is solved again with presolve switched off."""
        status, obj, chosen = self._partition_once(columns, cost, feasibility_only, True)
        if status not in ("Optimal", "Infeasible"):
            status, obj, chosen = self._partition_once(columns, cost, feasibility_only, False)
        return status, obj, chosen

    def _partition_once(self, columns: np.ndarray, cost, feasibility_only, presolve):
        import highspy
        n = self.inst.n
        members = self._members_of()
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", float(self.time_limit))
        h.setOptionValue("mip_rel_gap", 0.0)
        h.setOptionValue("mip_abs_gap", 1e-9)
        if not presolve:
            h.setOptionValue("presolve", "off")
        lp = highspy.HighsLp()
        ncol = int(columns.size)
        lp.num_col_ = ncol
        lp.num_row_ = n + 1
        obj = np.zeros(ncol) if feasibility_only else cost
        lp.col_cost_ = np.asarray(obj, dtype=float)
        lp.col_lower_ = np.zeros(ncol)
        lp.col_upper_ = np.ones(ncol)
        lp.row_lower_ = np.array([1.0] * n + [0.0])
        lp.row_upper_ = np.array([1.0] * n + [float(self.slots)])
        starts, index, value = [0], [], []
        for m in columns:
            rows = members[int(m)] + [n]
            index.extend(rows)
            value.extend([1.0] * len(rows))
            starts.append(len(index))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.array(starts, dtype=np.int32)
        lp.a_matrix_.index_ = np.array(index, dtype=np.int32)
        lp.a_matrix_.value_ = np.array(value, dtype=float)
        lp.integrality_ = [highspy.HighsVarType.kInteger] * ncol
        h.passModel(lp)
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        if status != "Optimal":
            return status, None, None
        x = np.array(h.getSolution().col_value)
        chosen = [int(columns[i]) for i in np.nonzero(x > 0.5)[0]]
        return status, float(h.getInfo().objective_function_value), chosen

    def _plan(self, chosen: list[int]) -> Solution:
        inst = self.inst
        sol = Solution(inst.K, inst.P)
        cells = [(k, h) for k in range(inst.K) for h in range(inst.P)]
        for (k, h), m in zip(cells, chosen):
            sol.cells[k][h] = self.tsp.tour(m)
        return sol

    # --- public API -----------------------------------------------------
    def solve(self, r_bar: float) -> dict:
        """Certified minimum workload under the ceiling, or certified infeasibility."""
        ceiling = CredibilisticCeiling(r_bar)
        ft, fs = self.multipliers(ceiling)
        qdur = ft * self.tsp.best + fs * self.serv
        cols = np.nonzero(qdur <= self.inst.L + 1e-9)[0]
        cols = cols[cols > 0]
        t0 = time.perf_counter()
        covered = np.bitwise_or.reduce(cols) if cols.size else 0
        if covered != (1 << self.inst.n) - 1:
            return dict(r_bar=r_bar, status="Infeasible", attainable=False,
                        seconds=time.perf_counter() - t0, columns=int(cols.size))
        status, obj, chosen = self._partition(cols, self.tsp.best[cols])
        seconds = time.perf_counter() - t0
        if status == "Infeasible":
            return dict(r_bar=r_bar, status=status, attainable=False,
                        seconds=seconds, columns=int(cols.size))
        if chosen is None:
            return dict(r_bar=r_bar, status=status, attainable=None,
                        seconds=seconds, columns=int(cols.size))
        sol = self._plan(chosen)
        R, workload = self.inst.evaluate(sol)
        mr = max(v for row in R for v in row)
        assert mr <= r_bar + 1e-9, (mr, r_bar)
        assert abs(workload - (obj + self.serv[-1])) < 1e-6 * max(1.0, workload)
        return dict(r_bar=r_bar, status=status, attainable=True, workload=workload,
                    max_risk=mr, solution=[[list(c) for c in row] for row in sol.cells],
                    seconds=seconds, columns=int(cols.size))

    def min_max_risk(self, start=(0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
                                  0.90, 0.99)) -> dict:
        """Exact smallest attainable worst-cell overtime credibility.

        An attainable ceiling from `start` supplies a plan whose worst-cell
        credibility bounds r_min from above; bisection over the subset risks
        below that bound, with set-partitioning feasibility models, closes
        the interval exactly.
        """
        hi_val = None
        for r in start:
            o = self.solve(r)
            if o.get("attainable"):
                hi_val = o["max_risk"]
                break
        if hi_val is None:
            return dict(r_min=None)
        risk = self.subset_risk()
        idx = np.arange(1, 1 << self.inst.n)
        thresholds = np.unique(risk[idx][risk[idx] <= hi_val + 1e-12])
        full = (1 << self.inst.n) - 1

        def attainable(theta):
            cols = idx[risk[idx] <= theta + 1e-12]
            if cols.size == 0 or np.bitwise_or.reduce(cols) != full:
                return False
            status, _, _ = self._partition(cols, None, feasibility_only=True)
            return status == "Optimal"

        lo, hi = 0, len(thresholds) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if attainable(thresholds[mid]):
                hi = mid
            else:
                lo = mid + 1
        return dict(r_min=float(thresholds[lo]))
