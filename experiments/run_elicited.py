"""Elicited-panel study: what the first-session expert answers imply for ceiling plans.

    python3 experiments/run_elicited.py --part describe              # descriptive summary
    python3 experiments/run_elicited.py --part small --workers 2     # certified
    python3 experiments/run_elicited.py --part extension --workers 2 # paired readings, leave-one-out
    python3 experiments/run_elicited.py --part quantile --workers 2  # price vs protected share

The twenty R1 forms of the expert campaign (data/reliability20/responses_numeric.csv,
session R1) define, per scenario, admissible planning durations at the
credibility ceiling 0.30: the directly elicited q060 and, where a complete
coherent triangle was given, its interpolation q_tri = b + 0.4(c - b).  Travel
legs of an instance are assessed with S1 (urban) or S2 (extra-urban), service
legs with S3 (access available) or S4 (waiting for access); every leg is a
scaled copy of the assessment relative to the panel-median mode.

Planning bases (multipliers of unit travel and service legs at r = 0.30):
  consensus_direct      median direct q060
  consensus_triangular  q_tri of the parameter-wise median triangle
  envelope_direct       largest direct q060 (all coherent judgments)
  envelope_pass         largest direct q060 among respondents passing both checks
  envelope_triangular   largest q_tri over complete coherent triangles

The shift limit is rescaled to L' = kappa L so that the consensus planner of
an instance faces the benchmark tightness (kappa: consensus over nominal
planning workload of the deterministic initial plan).  For every basis the
certified minimum-workload plan at L' and the certified minimum shift length
L* are computed; plans are checked against the direct envelopes.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common
from common import instances, common_scale
from crisp_equivalent import CredibilisticCeiling
from elicitation import (CONTEXTS, READINGS, SCENARIOS, LEG_TYPE, THRESHOLDS, RESPONSES,
                         ElicitationPanel, PlanningBasis)
from exact_small import ExactCeilingSolver
from expert_panel import PanelFactory
from problem import Instance, Solution

R_BAR = 0.30
SESSION = "R1"
BASES = ("consensus_direct", "consensus_triangular", "envelope_direct",
         "envelope_pass", "envelope_triangular")


def load_panel() -> ElicitationPanel:
    return ElicitationPanel.from_files(RESPONSES, session=SESSION)


def all_bases(panel: ElicitationPanel, ctx) -> dict[str, PlanningBasis]:
    b = panel.bases(ctx)
    passed = panel.subset(lambda r: r.passes_checks, "comprehension pass")
    e = passed.bases(ctx, reference=panel)["envelope_direct"]
    b["envelope_pass"] = PlanningBasis("envelope_pass", e.travel, e.service,
                                       "largest direct q, both checks passed")
    return b


# =============================================================================
# exact solver with explicit planning multipliers
# =============================================================================
class MultiplierCeilingSolver(ExactCeilingSolver):
    """Certified optima of the crisp equivalent for given planning multipliers.

    The cell visiting subset S along its minimum-travel tour satisfies the
    ceiling iff f_t T*(S) + f_s V(S) <= L (crisp routing equivalent); the
    multipliers come from a planning basis instead of the instance factory.
    """

    def planning(self, basis: PlanningBasis) -> np.ndarray:
        return basis.travel * self.tsp.best + basis.service * self.serv

    def solve_basis(self, basis: PlanningBasis, limit: float) -> dict:
        q = self.planning(basis)
        cols = np.nonzero(q <= limit + 1e-9)[0]
        cols = cols[cols > 0]
        full = (1 << self.inst.n) - 1
        if cols.size == 0 or np.bitwise_or.reduce(cols) != full:
            return dict(attainable=False)
        status, obj, chosen = self._partition(cols, self.tsp.best[cols])
        if status == "Infeasible":
            return dict(attainable=False)
        if chosen is None:
            return dict(attainable=None, status=status)
        return dict(attainable=True, workload=float(obj + self.serv[-1]), masks=chosen)

    def feasible(self, q: np.ndarray, limit: float) -> bool:
        cols = np.nonzero(q <= limit + 1e-12)[0]
        cols = cols[cols > 0]
        if cols.size == 0 or np.bitwise_or.reduce(cols) != (1 << self.inst.n) - 1:
            return False
        status, _, _ = self._partition(cols, None, feasibility_only=True)
        return status == "Optimal"

    def min_shift(self, basis: PlanningBasis, start: float) -> float:
        """Certified smallest L at which the ceiling is attainable for the basis.

        An exponential search from `start` brackets L*; bisection over the
        subset planning durations inside the bracket closes it exactly.
        """
        q = self.planning(basis)
        n = self.inst.n
        lower = float(max(q[1 << i] for i in range(n)))
        hi_L = max(lower, start)
        lo_L = lower
        while not self.feasible(q, hi_L):
            lo_L, hi_L = hi_L, hi_L * 1.1
        vals = np.unique(q[(q >= lo_L - 1e-12) & (q <= hi_L + 1e-12)])
        lo, hi = 0, len(vals) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.feasible(q, vals[mid]):
                hi = mid
            else:
                lo = mid + 1
        assert self.feasible(q, vals[lo])
        return float(vals[lo])

    def cell_planning(self, masks, basis: PlanningBasis) -> list[float]:
        q = self.planning(basis)
        return [float(q[m]) for m in masks]

    def plan(self, masks) -> Solution:
        return self._plan(masks)


class IndependentCheck:
    """Recomputes cell planning durations from coordinates and tours."""

    def __init__(self, inst):
        self.inst = inst

    def cells(self, sol: Solution, basis: PlanningBasis) -> list[float]:
        out = []
        for row in sol.cells:
            for seq in row:
                if not seq:
                    out.append(0.0)
                    continue
                path = [0] + list(seq) + [0]
                T = sum(self.inst.modal_travel(path[i], path[i + 1]) for i in range(len(path) - 1))
                V = sum(self.inst.service_modal[t] for t in seq)
                out.append(basis.travel * T + basis.service * V)
        return out

    def coverage(self, sol: Solution) -> bool:
        return sorted(t for row in sol.cells for c in row for t in c) == list(range(1, self.inst.n + 1))


def nominal_multipliers(inst) -> tuple[float, float]:
    """Planning multipliers of the instance's own unit legs at r = 0.30."""
    f = inst.factory
    c = CredibilisticCeiling(R_BAR)
    return (c.planning_duration(getattr(f, "make_travel", f.make)(1.0)),
            c.planning_duration(getattr(f, "make_service", f.make)(1.0)))


def calibration(inst, cons: PlanningBasis) -> dict:
    """kappa: consensus over nominal planning workload of the initial plan."""
    ft, fs = nominal_multipliers(inst)
    W = common_scale(inst)
    V = float(sum(inst.service_modal[1:]))
    T = W - V
    kappa = (cons.travel * T + cons.service * V) / (ft * T + fs * V)
    return dict(kappa=kappa, nominal_travel=ft, nominal_service=fs,
                service_share=V / W, limit=inst.L * kappa)


# =============================================================================
# jobs
# =============================================================================
def small_job(job):
    family, seed = job
    panel = load_panel()
    inst = instances().build(family, seed)
    solver = MultiplierCeilingSolver(inst)
    check = IndependentCheck(inst)
    recs = []
    for ctx in CONTEXTS:
        bases = all_bases(panel, ctx)
        cal = calibration(inst, bases["consensus_direct"])
        Lp = cal["limit"]
        rec = dict(family=family, seed=seed, context=ctx.name, L=inst.L, **cal)
        plans = {}
        for name in BASES:
            b = bases[name]
            o = solver.solve_basis(b, Lp)
            r = dict(attainable=o["attainable"])
            if o["attainable"]:
                sol = solver.plan(o["masks"])
                assert check.coverage(sol)
                own = check.cells(sol, b)
                assert max(own) <= Lp + 1e-6, (name, max(own), Lp)
                r["workload"] = o["workload"]
                r["solution"] = sol.cells
                for env in ("envelope_direct", "envelope_pass", "consensus_direct"):
                    cells = check.cells(sol, bases[env])
                    r[f"overrun_{env}"] = max(cells) / Lp - 1.0
                plans[name] = sol
            o_fixed = solver.solve_basis(b, inst.L) if name in ("consensus_direct", "envelope_pass") else None
            if o_fixed is not None:
                r["attainable_at_benchmark_L"] = o_fixed["attainable"]
            rec[name] = r
        # certified minimum shift lengths (independent of the rescaled limit)
        for name in BASES:
            b = bases[name]
            Ls = solver.min_shift(b, Lp)
            rec[name]["min_shift_ratio"] = Ls / Lp
            o = solver.solve_basis(b, Ls)
            assert o["attainable"], (family, seed, name)
            sol = solver.plan(o["masks"])
            assert check.coverage(sol) and max(check.cells(sol, b)) <= Ls + 1e-6
            rec[name]["at_min_shift"] = dict(
                workload=o["workload"], solution=sol.cells,
                **{f"overrun_{env}": max(check.cells(sol, bases[env])) / Ls - 1.0
                   for env in ("envelope_direct", "envelope_pass", "consensus_direct")})
        # workload price of protection once the shift admits every basis
        Lc = max(rec[n]["min_shift_ratio"] for n in BASES) * Lp
        rec["common_limit_ratio"] = Lc / Lp
        rec["workload_at_common_limit"] = {n: solver.solve_basis(bases[n], Lc)["workload"]
                                           for n in BASES}
        recs.append(rec)
    return recs


def fuzzy_check_job(job):
    """Worst-case credibility of robust-triangular plans with the PanelFN machinery.

    The certified minimum-workload plan of the triangular envelope at its own
    minimum shift length is re-evaluated as a fuzzy plan under every leg-wise
    choice of the elicited triangles (robust-envelope corollary); its worst-case credibility
    must not exceed 0.30; the consensus plan is checked against the same panel.
    """
    family, seed = job
    panel = load_panel()
    inst = instances().build(family, seed)
    solver = MultiplierCeilingSolver(inst)
    out = []
    for ctx in CONTEXTS:
        bases = all_bases(panel, ctx)
        cal = calibration(inst, bases["consensus_direct"])
        tri = panel.triangular_panel(ctx)
        rec = dict(family=family, seed=seed, context=ctx.name)
        for name in ("envelope_triangular", "consensus_triangular"):
            Ls = solver.min_shift(bases[name], cal["limit"])
            o = solver.solve_basis(bases[name], Ls)
            robust = Instance(inst.n, inst.K, inst.P, Ls, inst.coords,
                              inst.service_modal, inst.speed, PanelFactory(tri))
            R, _ = robust.evaluate(solver.plan(o["masks"]))
            rec[name] = max(v for row in R for v in row)
        assert rec["envelope_triangular"] <= R_BAR + 1e-9, rec
        out.append(rec)
    return out


class CachedShift:
    """Certified minimum shift lengths cached by planning multipliers."""

    def __init__(self, solver: MultiplierCeilingSolver, start: float):
        self.solver, self.start, self.cache = solver, start, {}

    def __call__(self, basis: PlanningBasis) -> float:
        key = (round(basis.travel, 12), round(basis.service, 12))
        if key not in self.cache:
            self.cache[key] = self.solver.min_shift(basis, self.start)
        return self.cache[key]


def extension_job(job):
    """Same-respondent comparison, coverage effect and leave-one-out sensitivity.

    * paired: direct and triangular readings restricted to respondents who gave
      both a complete coherent triangle and a direct answer, so that the
      interpolation is compared on identical assessors;
    * coverage: the direct envelope of all coherent answers against the direct
      envelope of that paired subset (assessors without a finite support);
    * leave-one-out: consensus and envelope of the direct answers with each
      respondent removed in turn (references stay those of the full panel).
    """
    family, seed = job
    panel = load_panel()
    inst = instances().build(family, seed)
    solver = MultiplierCeilingSolver(inst)
    check = IndependentCheck(inst)
    out = []
    for ctx in CONTEXTS:
        full = all_bases(panel, ctx)
        paired = panel.bases_paired(ctx)
        Lp = calibration(inst, full["consensus_direct"])["limit"]
        shift = CachedShift(solver, Lp)
        rec = dict(family=family, seed=seed, context=ctx.name, limit=Lp)
        cons = shift(full["consensus_direct"])
        rec["min_shift"] = {k: shift(b) / Lp for k, b in {**full, **paired}.items()}
        # plan protecting the paired triangles, checked on both direct envelopes
        name = "envelope_paired_triangular"
        Ls = shift(paired[name])
        o = solver.solve_basis(paired[name], Ls)
        sol = solver.plan(o["masks"])
        rec["paired_triangular_solution"] = sol.cells
        rec["paired_triangular_plan"] = {
            f"overrun_{env}": max(check.cells(sol, b)) / Ls - 1.0
            for env, b in (("envelope_paired_direct", paired["envelope_paired_direct"]),
                           ("envelope_direct", full["envelope_direct"]))}
        loo = {}
        for r in panel.respondents:
            sub = panel.subset(lambda x, rid=r.rid: x.rid != rid, f"without {r.rid}")
            b = sub.bases(ctx, reference=panel)
            loo[r.rid] = dict(consensus=shift(b["consensus_direct"]) / Lp,
                              envelope=shift(b["envelope_direct"]) / Lp)
        rec["leave_one_out"] = loo
        rec["price_full"] = rec["min_shift"]["envelope_direct"] * Lp / cons
        out.append(rec)
    return out


# =============================================================================
# descriptive summary of the returned forms
# =============================================================================
def _describe(vals):
    vals = sorted(float(v) for v in vals)
    if not vals:
        return dict(n=0)
    def pct(p):
        pos = (len(vals) - 1) * p
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    return dict(n=len(vals), median=statistics.median(vals), q1=pct(.25), q3=pct(.75),
                minimum=vals[0], maximum=vals[-1])


def describe(panel: ElicitationPanel) -> dict:
    d, t = READINGS["direct"], READINGS["triangular"]
    out = dict(n=len(panel), session=SESSION, scenarios={}, contexts={})
    passed = panel.subset(lambda r: r.passes_checks, "pass")
    for s in SCENARIOS:
        js = panel.judgments(s)
        coh = [j for j in js if j.coherent]
        ref = panel.reference_mode(s)
        tri = [j for j in js if j.complete_triangle]
        deltas = [j.delta for j in js if j.delta is not None]
        rel = [j.delta / j.q_triangular for j in js if j.delta is not None]
        out["scenarios"][s] = dict(
            leg=LEG_TYPE[s], reference_mode=ref,
            n_direct=len(panel.planning_values(s, d)),
            n_triangle=len(tri),
            n_flagged=sum(not j.coherent for j in js),
            flags={f: sum(f in j.flags for j in js) for f in
                   sorted({f for j in js for f in j.flags})},
            missing={k: sum(getattr(j, k) is None for j in js) for k in ("a", "b", "c", "q")},
            q_without_c=sum(j.q is not None and j.c is None and j.coherent for j in js),
            triangle_without_q=sum(j.complete_triangle and j.q is None for j in js),
            b=_describe([j.b for j in coh if j.b is not None]),
            q=_describe(panel.planning_values(s, d)),
            qprime=_describe([j.qprime for j in coh if j.qprime is not None]),
            delta_prime=_describe([j.qprime-(j.b+.7*(j.c-j.b)) for j in tri if j.qprime is not None]),
            abs_delta_prime=_describe([abs(j.qprime-(j.b+.7*(j.c-j.b))) for j in tri if j.qprime is not None]),
            q_pass=_describe(passed.planning_values(s, d)),
            q_tri=_describe(panel.planning_values(s, t)),
            delta=_describe(deltas), abs_delta=_describe([abs(x) for x in deltas]),
            rel_delta=_describe(rel),
            delta_negative=sum(x < -1e-9 for x in deltas),
            delta_zero=sum(abs(x) <= 1e-9 for x in deltas),
            delta_positive=sum(x > 1e-9 for x in deltas),
            left_spread=_describe([1 - j.a / j.b for j in tri]),
            right_spread=_describe([j.c / j.b - 1 for j in tri]),
            mode_ratio=_describe([j.b / ref for j in coh if j.b is not None]),
            own_multiplier=_describe([j.q / j.b for j in coh if j.q is not None]),
            multiplier_median=panel.multiplier(s, d, "median"),
            multiplier_max=panel.multiplier(s, d, "max"),
            multiplier_pass_max=passed.multiplier(s, d, "max", ref),
            multiplier_tri_median=panel.multiplier(s, t, "median"),
            multiplier_tri_max=panel.multiplier(s, t, "max"),
            binding_direct=panel.binding(s, d), binding_triangular=panel.binding(s, t),
            binding_pass=passed.binding(s, d),
            confidence=_describe([j.confidence for j in js if j.confidence is not None]),
            unbounded_c=sum(j.c_unbounded for j in js),
            thresholds={k: _describe([getattr(j, "qprime" if k == "q030" else ("q" if k == "q060" else k))
                                      for j in coh if getattr(j, "qprime" if k == "q030" else ("q" if k == "q060" else k)) is not None])
                        for k, _ in THRESHOLDS},
            residual={k: _describe([j.branch_residual(k) for j in coh if j.branch_residual(k) is not None])
                      for k in ("q080", "q030", "q010")},
            residual_values={k: [j.branch_residual(k) for j in coh if j.branch_residual(k) is not None]
                             for k in ("q080", "q030", "q010")})
    for ctx in CONTEXTS:
        out["contexts"][ctx.name] = {k: v.as_dict() for k, v in all_bases(panel, ctx).items()}
    R = panel.respondents
    num = lambda k: [float(r.answer(k)) for r in R if r.answer(k)]
    out["checks"] = dict(both_passed=sum(r.passes_checks for r in R),
                         ceiling_item=sum(r.passes_ceiling_item for r in R))
    out["practicality"] = _describe(num("practicality"))
    out["practicality_counts"] = {v: num("practicality").count(v) for v in sorted(set(num("practicality")))}
    out["minutes"] = _describe(num("completion_minutes"))
    out["flagged_respondents"] = sorted({r.rid for r in R for s in SCENARIOS
                                         if not r.judgments[s].coherent})
    out["failed_checks"] = sorted(r.rid for r in R if not r.passes_checks)
    return out


def quantile_job(job):
    """Protection price as a function of how many respondents are protected.

    For k = 1..n the basis plans every leg at the k-th largest direct answer of
    its scenario (k = 1: the envelope of all answers; protecting the
    n - k + 1 least pessimistic respondents of each leg type).  The certified
    minimum shift length is compared with that of the median consensus.
    """
    family, seed = job
    panel = load_panel()
    inst = instances().build(family, seed)
    solver = MultiplierCeilingSolver(inst)
    d = READINGS["direct"]
    out = []
    for ctx in CONTEXTS:
        full = all_bases(panel, ctx)
        Lp = calibration(inst, full["consensus_direct"])["limit"]
        shift = CachedShift(solver, Lp)
        cons = shift(full["consensus_direct"])
        rt, rs = panel.reference_mode(ctx.travel), panel.reference_mode(ctx.service)
        qt = sorted(panel.planning_values(ctx.travel, d), reverse=True)
        qs = sorted(panel.planning_values(ctx.service, d), reverse=True)
        price = {}
        for k in range(1, min(len(qt), len(qs)) + 1):
            b = PlanningBasis(f"order_{k}", qt[k - 1] / rt, qs[k - 1] / rs,
                              f"{k}-th largest direct answer")
            price[k] = shift(b) / cons - 1.0
        out.append(dict(family=family, seed=seed, context=ctx.name,
                        n_travel=len(qt), n_service=len(qs), price=price))
    return out


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("describe", "small", "extension", "quantile"), default="describe")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.perf_counter()
    panel = load_panel()
    meta = common.meta(block=f"elicited_{a.part}", r_bar=R_BAR,
                       responses=str(RESPONSES.relative_to(common.REPO)),
                       n_respondents=len(panel), session=SESSION,
                       data_sha256=hashlib.sha256(RESPONSES.read_bytes()).hexdigest())
    if a.part == "describe":
        payload = dict(meta=meta, summary=describe(panel))
        name = "elicited_summary.json"
    elif a.part == "small":
        jobs = [(f, s) for f in common.SMALL_FAMILIES for s in common.SEEDS]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            recs = [x for chunk in ex.map(small_job, jobs, chunksize=1) for x in chunk]
            fz = [x for chunk in ex.map(fuzzy_check_job, jobs, chunksize=1) for x in chunk]
        payload = dict(meta=dict(meta, wall_clock_s=time.perf_counter() - t0),
                       raw=recs, fuzzy_check=fz)
        name = "elicited_small.json"
    elif a.part == "extension":
        jobs = [(f, s) for f in common.SMALL_FAMILIES for s in common.SEEDS]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            recs = [x for chunk in ex.map(extension_job, jobs, chunksize=1) for x in chunk]
        payload = dict(meta=dict(meta, wall_clock_s=time.perf_counter() - t0), raw=recs)
        name = "elicited_extension.json"
    elif a.part == "quantile":
        jobs = [(f, s) for f in common.SMALL_FAMILIES for s in common.SEEDS]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            recs = [x for chunk in ex.map(quantile_job, jobs, chunksize=1) for x in chunk]
        payload = dict(meta=dict(meta, wall_clock_s=time.perf_counter() - t0), raw=recs)
        name = "elicited_quantile.json"
    path = common.dump(name, payload)
    print(f"[elicited {a.part}] -> {path} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
