"""
CC-LNS: chance-constrained risk-aware large neighbourhood search.

Two operating modes share the same destroy/repair machinery:

  bi-objective   minimise J_alpha = alpha*workload_hat + (1-alpha)*FFF.

  ceiling        minimise the total modal workload subject to
                 max_kh r_kh <= r_bar and possibilistic feasibility.  Every
                 candidate insertion is evaluated on the exact cellwise
                 ceiling (penalised excess); the incumbent is kept
                 lexicographically by (feasibility, excess, workload).  The
                 FFF sublevel set {FFF <= tau(r_bar)} is a sufficient
                 certificate of the ceiling (worst-cell certificate) and is
                 used as a descent surrogate while the ceiling is violated.

Repair evaluates exactly, for every cell, the `positions_per_cell` insertion
positions of least modal cost (one by default: when every leg membership is a
scaled copy of one membership per leg type, the position of least modal cost
also minimises every alpha-cut endpoint of the receiving cell, hence its
credibility and the merit within that cell); a share `random_removal` of the
destroy steps removes uniformly random tasks.  These settings were fixed on
development instances (generator seeds 110-150, experiments/run_development.py)
that are disjoint from the evaluation seeds.

The schedules of the penalty weight and of the operator mix are fixed.

workload_scale fixes the normalisation of the workload term; the experiment
drivers pass the same value to every method (common normalisation).
"""

from __future__ import annotations

import math

from aggregators import FFF_REPORT, front_loaded_weights, max_risk, gini, jain, cvar
from solver import EpsilonArchive, RALNS, _State, two_opt_cell


def certificate_target(inst, r_bar, w1=None):
    """tau(r_bar) of the worst-cell certificate: FFF <= tau implies max r <= r_bar.

    With the cell-anonymous FFF, FFF >= w_1 max_kh r_kh, w_1 = 2/(KP+1).
    """
    w1 = w1 if w1 is not None else front_loaded_weights(inst.K * inst.P)[0]
    return w1 * r_bar


class CCLNS(RALNS):
    """
    Chance-constrained risk-aware LNS (CC-LNS) with fixed penalty and
    operator schedules.
    """

    PATIENCE = 40
    PENALTY_MIN, PENALTY_MAX = 1.0, 256.0

    def __init__(self, instance, aggregator, use_controller=False,
                 r_bar=None, penalty_init=4.0, budget=None,
                 schedule=(1.7, 0.85, 20),
                 lexicographic=True, use_surrogate=True, polish=True,
                 generic_destroy=False, cost_only_repair=False,
                 guided_positions=True, positions_per_cell=1, random_removal=0.2,
                 beta=0.0, beta_max=0.6, workload_scale=None, **kw):
        if use_controller:
            raise ValueError("the rule-based controller (CC-LNS+FC) is not available")
        super().__init__(instance, aggregator, **kw)
        self.workload_scale = workload_scale
        self.cost_only_repair = cost_only_repair
        self.guided_positions = guided_positions
        self.positions_per_cell = positions_per_cell
        self.random_removal = random_removal
        self.beta = beta            # margin pressure in the repair criterion
        self.beta_max = beta_max
        self.lexicographic = lexicographic
        self.use_surrogate = use_surrogate
        self.polish_on = polish
        self.generic_destroy = generic_destroy
        self.r_bar = r_bar
        self.tau = (certificate_target(instance, r_bar)
                    if r_bar is not None else None)
        self.penalty = penalty_init
        self.schedule = schedule      # (growth, shrink, period) of the fixed rule
        self.budget = budget          # stop on evaluations, not iterations
        self.trace = []

    # --- objective ------------------------------------------------------
    def _excess(self, R):
        over = [max(0.0, v - self.r_bar) for row in R for v in row]
        return max(over) + 0.25 * (sum(over) / len(over))

    BARRIER_EXP = 6

    def _barrier(self, R):
        """Margin consumed towards the ceiling: ~0 well below r_bar, 1 at r_bar.

        The hinge (r_kh - r_bar)^+ in the merit is blind to how close a
        *feasible* cell sits to the ceiling, so a repair that merely avoids
        violating cells fills them to just under r_bar and violates on the next
        insertion.  This term charges for the margin a candidate insertion
        consumes, and beta is the lever between the two regimes the ceiling
        study puts at its extremes: beta = 0 inserts by workload alone (the
        LNS-AO behaviour: cheaper plans, the ceiling met less often), large beta
        keeps every cell away from the ceiling (the ceiling met more often, at a
        dearer plan).  It shapes only the direction of search: the incumbent is
        still kept lexicographically by (feasibility, excess, workload), so no
        value of beta can change what a run reports.
        """
        q = self.BARRIER_EXP
        n = len(R) * len(R[0])
        return sum((v / self.r_bar) ** q for row in R for v in row) / n

    def _scalar(self, R, makespan, violation=0.0):
        base = self.INFEASIBILITY_PENALTY * violation
        if self.r_bar is None:
            return (self.alpha * (makespan / self._lambda_scale)
                    + (1 - self.alpha) * self.agg(R) + base)
        # ceiling mode: cheapest plan meeting the overtime ceiling.  The exact
        # constraint is max_kh r_kh <= r_bar; the FFF surrogate enters through
        # the worst-cell certificate ({FFF <= tau} lies inside the feasible set).
        excess = self._excess(R)
        surrogate = (max(0.0, self.agg(R) - self.tau)
                     if (excess > 0.0 and self.use_surrogate) else 0.0)
        margin = self.beta * self._barrier(R) if self.beta > 0.0 else 0.0
        return (makespan / self._lambda_scale
                + self.penalty * (excess + 0.25 * surrogate) + margin + base)

    def _rank(self, st):
        """Lexicographic merit used to keep the incumbent in ceiling mode."""
        if self.r_bar is None or not self.lexicographic:
            return (self._scalar(st.risk, st.makespan, st.violation),)
        return (st.violation > 1e-9, round(self._excess(st.risk), 6), st.makespan)

    # --- destroy operators specific to the ceiling mode -------------------
    def _destroy_ceiling(self, sol, R, q):
        """Remove tasks from the cells that exceed the ceiling, worst first."""
        cells = sorted(((R[k][h] - self.r_bar, k, h) for k in range(sol.K)
                        for h in range(sol.P) if R[k][h] > self.r_bar),
                       reverse=True)
        if not cells:
            return self._destroy_risk(sol, R, q)
        picked = []
        for _, k, h in cells:
            c = list(sol.cells[k][h])
            self.rng.shuffle(c)
            picked.extend(c[: max(1, q // len(cells) + 1)])
            if len(picked) >= q:
                break
        return picked[:q]

    def _destroy_workload(self, st, q):
        """Remove boundary tasks of the busiest cells to compress the workload."""
        cells = sorted(((st.modal[k][h], k, h) for k in range(st.sol.K)
                        for h in range(st.sol.P)), reverse=True)
        picked = []
        for _, k, h in cells[: max(1, (st.sol.K * st.sol.P) // 2)]:
            c = st.sol.cells[k][h]
            picked.extend(c[:2] + c[-2:])
        self.rng.shuffle(picked)
        return list(dict.fromkeys(picked))[:q]

    # --- repair -------------------------------------------------------------
    def _guided_repair(self, st, removed):
        """Exact-merit insertion restricted to modal-cost candidate positions.

        For every cell the `positions_per_cell` positions of least modal
        insertion cost (crisp data, no fuzzy evaluation) are evaluated exactly:
        one fuzzy cell evaluation each, scored by the full merit after the
        receiving cell is updated.  The best candidate is inserted.
        """
        inst = self.inst
        d = inst.modal_travel
        self.rng.shuffle(removed)
        cells = [(k, h) for k in range(inst.K) for h in range(inst.P)]
        touched = set()
        for t in removed:
            best, best_val, best_triple = None, float("inf"), None
            for k, h in cells:
                c = st.sol.cells[k][h]
                costs = []
                for pos in range(len(c) + 1):
                    prev = c[pos - 1] if pos else 0
                    nxt = c[pos] if pos < len(c) else 0
                    costs.append((d(prev, t) + d(t, nxt) - d(prev, nxt), pos))
                costs.sort()
                base = (st.risk[k][h], st.modal[k][h], st.opt[k][h])
                for _, pos in costs[: self.positions_per_cell]:
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
        for k, h in touched:
            if self.polish_on:
                two_opt_cell(inst, st.sol.cells[k][h])
                st.refresh(k, h)
        return st

    def _repair(self, st, removed):
        if not self.cost_only_repair:
            if self.guided_positions:
                return self._guided_repair(st, removed)
            return super()._repair(st, removed)
        inst = self.inst
        self.rng.shuffle(removed)
        touched = set()
        for t in removed:
            bk, bh, bp, bc = 0, 0, 0, float("inf")
            for k in range(inst.K):
                for h in range(inst.P):
                    c = st.sol.cells[k][h]
                    for pos in range(len(c) + 1):
                        prev = c[pos - 1] if pos else 0
                        nxt = c[pos] if pos < len(c) else 0
                        delta = (inst.modal_travel(prev, t) + inst.modal_travel(t, nxt)
                                 - inst.modal_travel(prev, nxt))
                        if delta < bc:
                            bk, bh, bp, bc = k, h, pos, delta
            st.sol.cells[bk][bh].insert(bp, t)
            touched.add((bk, bh))
        for k, h in touched:
            if self.polish_on:
                two_opt_cell(inst, st.sol.cells[k][h])
            st.refresh(k, h)
        return st

    # --- destroy selection ------------------------------------------------
    def _controlled_destroy(self, cand, out, q):
        inst = self.inst
        xv = out["cross_vehicle"]
        u = self.rng.random()
        if self.random_removal > 0.0 and self.rng.random() < self.random_removal:
            pool = cand.sol.all_tasks()
            self.rng.shuffle(pool)
            return 5, pool[:q]
        if inst.K > 1 and u < xv:
            op = 2
        elif self.r_bar is None or self.generic_destroy:
            op = 1 if self.rng.random() < out["risk_focus"] else 0
        else:
            op = 3 if self.rng.random() < out["violation_focus"] else 4
        if op == 0:
            removed = self._destroy_boundary(cand.sol, q)
        elif op == 1:
            removed = self._destroy_risk(cand.sol, cand.risk, q)
        elif op == 2:
            removed = self._destroy_intervehicle(cand.sol, cand.risk, q)
        elif op == 3:
            removed = self._destroy_ceiling(cand.sol, cand.risk, q)
        else:
            removed = self._destroy_workload(cand, q)
        return op, removed

    # --- fixed operator mix and destroy size for one iteration ----------
    def _decide(self, st, since, temp, it, record_trace):
        out = dict(risk_focus=0.5, cross_vehicle=0.2, violation_focus=0.5)
        q = max(2, int(self.rng.uniform(*self.destroy_frac) * self.inst.n))
        return out, q, temp

    # --- main loop -------------------------------------------------------
    def run(self, archive=None, record_trace=False):
        inst = self.inst
        st = _State(inst, self.initial())
        self._lambda_scale = (self.workload_scale if self.workload_scale
                              else max(st.makespan, 1.0))
        cur_val = self._scalar(st.risk, st.makespan, st.violation)
        best_st, best_rank = st.clone(), self._rank(st)
        archive = archive if archive is not None else EpsilonArchive()
        if st.violation <= 1e-9:
            archive.add((st.makespan, FFF_REPORT(st.risk)))
        since = 0
        temp = self.t0
        op_use = [0, 0, 0, 0, 0, 0]
        n_iter = self.iterations if self.budget is None else 10 ** 9
        for it in range(n_iter):
            if self.budget is not None and self.evaluations >= self.budget:
                break
            pen_before = self.penalty
            beta_before = self.beta
            out, q, temp_now = self._decide(st, since, temp, it, record_trace)
            if self.penalty != pen_before or self.beta != beta_before:
                cur_val = self._scalar(st.risk, st.makespan, st.violation)
                if not self.lexicographic:
                    best_rank = self._rank(best_st)
            cand = st.clone()
            op, removed = self._controlled_destroy(cand, out, q)
            if not removed:
                continue
            op_use[op] += 1
            touched = [(k, h) for k in range(inst.K) for h in range(inst.P)
                       if any(t in removed for t in cand.sol.cells[k][h])]
            cand.sol.remove(removed)
            for k, h in touched:
                cand.refresh(k, h)
            cand = self._repair(cand, list(removed))
            cval = self._scalar(cand.risk, cand.makespan, cand.violation)
            if cand.violation <= 1e-9:
                archive.add((cand.makespan, FFF_REPORT(cand.risk)))
            if cval < cur_val or self.rng.random() < math.exp(
                    -(cval - cur_val) / max(temp_now, 1e-9)):
                st, cur_val = cand, cval
            rk = self._rank(cand)
            if rk < best_rank:
                best_st, best_rank = cand.clone(), rk
                since = 0
            else:
                since += 1
            temp *= self.cooling
            growth, shrink, period = self.schedule
            if self.r_bar is not None and (it + 1) % period == 0:
                # fixed penalty schedule
                if max_risk(best_st.R) > self.r_bar:
                    self.penalty = min(self.PENALTY_MAX, self.penalty * growth)
                else:
                    self.penalty = max(self.PENALTY_MIN, self.penalty * shrink)
                cur_val = self._scalar(st.risk, st.makespan, st.violation)
                if not self.lexicographic:
                    best_rank = self._rank(best_st)
        R = best_st.R
        f = FFF_REPORT(R)
        res = dict(R=R, makespan=best_st.makespan, fff=f,
                   solution=[[list(c) for c in row] for row in best_st.sol.cells],
                   violation=best_st.violation,
                   feasible=best_st.violation <= 1e-9,
                   max_risk=max_risk(R), gini=gini(R), jain=jain(R),
                   cvar=cvar(R), archive=archive, evaluations=self.evaluations,
                   operator_use=op_use)
        if self.r_bar is not None:
            res.update(tau=self.tau, r_bar=self.r_bar, beta_final=self.beta,
                       ceiling_met=bool(best_st.violation <= 1e-9
                                        and max_risk(R) <= self.r_bar + 1e-12),
                       certified=bool(best_st.violation <= 1e-9 and f <= self.tau),
                       certificate_slack=f / self.tau if self.tau > 0 else None)
        if record_trace:
            res["trace"] = self.trace
        return res


FICLNS = CCLNS          # alias


def certified_from_archive(inst, archive, r_bar):
    """Cheapest archive point whose FFF attains the certificate target."""
    tau = certificate_target(inst, r_bar)
    ok = [p for p in archive.points() if p[1] <= tau]
    if not ok:
        return dict(certified=False, makespan=None, fff=None, tau=tau)
    best = min(ok, key=lambda p: p[0])
    return dict(certified=True, makespan=best[0], fff=best[1], tau=tau)
