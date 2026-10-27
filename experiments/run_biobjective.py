"""Bi-objective (workload, FFF) studies and membership conventions.

    python3 experiments/run_biobjective.py scalarized aggregators fleet shapes --workers 2

  scalarized   six methods on C1-C4, five seeds, alpha sweep, common
               normalisation; feasible archives only (hypervolume valid).
  fleet        CC-LNS, LNS-AO, NSGA-II on F2 and F4.
  aggregators  CC-LNS with eight search aggregators (solver, budget and
               reporting held fixed).
  shapes       ceiling mode under triangular, trapezoidal and Gaussian
               readings of the same three elicited points (CC-LNS, and
               OR-Tools on the crisp equivalent of each reading, 10 s).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import common
from common import instances, AllowanceBook, common_scale, make_solver, verify, solver_seed
from aggregators import (OWAFairness, GeneralizedOWAFairness, MeanRisk, MaxRisk,
                         CVaRRisk, GiniPenalisedRisk, JainPenalisedRisk, max_risk)
from crisp_equivalent import CredibilisticCeiling
from ortools_reference import ORToolsCeilingSolver
from solver import EpsilonArchive, hypervolume_2d

AGGREGATORS = {"FFF": OWAFairness, "gOWA2": lambda: GeneralizedOWAFairness(2.0),
               "gOWA3": lambda: GeneralizedOWAFairness(3.0), "Mean": MeanRisk,
               "Max": MaxRisk, "CVaR": lambda: CVaRRisk(0.9),
               "Gini": lambda: GiniPenalisedRisk(1.0), "Jain": lambda: JainPenalisedRisk(0.5)}
SHAPES = ["tfn", "trapezoidal", "gaussian"]
SHAPE_CEILINGS = [0.30, 0.40]
SHAPE_ORTOOLS_SECONDS = 10.0


def sweep_job(job):
    family, seed, method, agg_tag = job
    inst = instances().build(family, seed)
    budget = AllowanceBook().get(family, seed)
    scale = common_scale(inst)
    archive = EpsilonArchive()
    ref_out, evals = None, 0
    for a in common.ALPHAS:
        s = make_solver(method, inst, alpha=a, seed=seed * 97 + int(a * 10),
                        budget=budget, scale=scale, aggregator=AGGREGATORS[agg_tag]())
        out = s.run(archive=archive)
        evals += out["evaluations"]
        if abs(a - common.REPORT_ALPHA) < 1e-9:
            chk = verify(inst, out)
            ref_out = dict(out, **chk)
    return dict(family=family, seed=seed, method=method, aggregator=agg_tag,
                allowance=budget, evaluations=evals, points=archive.points(),
                makespan=ref_out["makespan"], fff=ref_out["fff"],
                max_risk=ref_out["max_risk"], gini=ref_out["gini"],
                jain=ref_out["jain"], cvar=ref_out["cvar"],
                feasible=bool(ref_out["feasible"]))


def finish(runs):
    ref = {}
    for f in {r["family"] for r in runs}:
        mk = [p[0] for r in runs if r["family"] == f for p in r["points"]]
        ref[f] = (max(mk) * 1.05 if mk else 1.0, 1.0)
    for r in runs:
        r["hv"] = hypervolume_2d(r["points"], ref[r["family"]]) / ref[r["family"]][0]
        r["n_points"] = len(r.pop("points"))
    return {k: list(v) for k, v in ref.items()}


def shape_job(job):
    family, seed, shape, r_bar = job
    inst = instances().build(family, seed, shape)
    budget = AllowanceBook().get(family, seed)
    s = make_solver("CC-LNS", inst, alpha=common.REPORT_ALPHA,
                    seed=solver_seed(seed, r_bar), budget=budget, r_bar=r_bar,
                    scale=common_scale(inst))
    out = s.run()
    chk = verify(inst, out)
    c = CredibilisticCeiling(r_bar)
    f = inst.factory
    g = ORToolsCeilingSolver(inst, r_bar, time_limit=SHAPE_ORTOOLS_SECONDS).run()
    return dict(family=family, seed=seed, shape=shape, r_bar=r_bar,
                travel_multiplier=c.planning_duration(f.make_travel(1.0)),
                service_multiplier=c.planning_duration(f.make_service(1.0)),
                makespan=chk["makespan"], max_risk=chk["max_risk"],
                feasible=chk["feasible"],
                ceiling_met=bool(chk["feasible"] and chk["max_risk"] <= r_bar + 1e-12),
                ortools_met=bool(g.get("ceiling_met", False)),
                ortools_makespan=g.get("makespan"))


def run_block(name, workers):
    t0 = time.perf_counter()
    if name in ("scalarized", "fleet", "aggregators"):
        if name == "scalarized":
            jobs = [(f, s, m, "FFF") for f in ("C1", "C2", "C3", "C4")
                    for s in common.SEEDS5 for m in common.METHODS]
        elif name == "fleet":
            jobs = [(f, s, m, "FFF") for f in ("F2", "F4") for s in common.SEEDS5
                    for m in ("CC-LNS", "LNS-AO", "NSGA-II")]
        else:
            jobs = [(f, s, "CC-LNS", t) for f in ("C1", "C2", "C3", "C4")
                    for s in common.SEEDS5 for t in AGGREGATORS]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            runs = list(ex.map(sweep_job, jobs, chunksize=1))
        ref = finish(runs)
        extra = dict(alphas=common.ALPHAS, report_alpha=common.REPORT_ALPHA,
                     hv_reference=ref)
    else:
        jobs = [(f, s, sh, r) for f in ("C2", "C4", "F2") for s in common.SEEDS
                for sh in SHAPES for r in SHAPE_CEILINGS]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            runs = list(ex.map(shape_job, jobs, chunksize=1))
        extra = dict(shapes=SHAPES, ceilings=SHAPE_CEILINGS)
    payload = dict(meta=common.meta(block=name, wall_clock_s=time.perf_counter() - t0,
                                    **extra), raw=runs)
    path = common.dump(f"{name}.json", payload)
    print(f"[{name}] {len(runs)} records in {payload['meta']['wall_clock_s']:.0f}s -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks", nargs="*",
                    default=["aggregators"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for b in a.blocks:
        run_block(b, a.workers)


if __name__ == "__main__":
    main()
