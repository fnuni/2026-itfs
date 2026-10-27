"""Certified small-instance study.

    python3 experiments/run_exact_small.py --workers 2      # results/exact_small.json

For every small instance (families S10, S12, F12, F16; ten seeds) and ceiling:
  * the certified optimum (or certified infeasibility) of the ceiling problem
    and the exact smallest attainable worst-cell credibility r_min
    (exact_small.ExactCeilingSolver, HiGHS set partitioning);
  * the six methods under the common protocol (counted allowance);
  * the OR-Tools reference on the crisp equivalent (2 s wall clock).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import common
from common import (instances, AllowanceBook, common_scale, make_solver, verify,
                    solver_seed)
from exact_small import ExactCeilingSolver
from ortools_reference import ORToolsCeilingSolver

ORTOOLS_SECONDS = 2.0


def run_instance(job):
    family, seed = job
    book = AllowanceBook()
    inst = instances().build(family, seed)
    budget = book.get(family, seed)
    scale = common_scale(inst)
    ex = ExactCeilingSolver(inst)
    t0 = time.perf_counter()
    exact = {}
    for r in common.CEILINGS:
        o = ex.solve(r)
        exact[str(r)] = {k: v for k, v in o.items() if k != "solution"}
        exact[str(r)]["solution"] = o.get("solution")
    rmin = ex.min_max_risk()["r_min"]
    exact_seconds = time.perf_counter() - t0
    runs = []
    for r in common.CEILINGS:
        for m in common.METHODS:
            s = make_solver(m, inst, alpha=common.REPORT_ALPHA,
                            seed=solver_seed(seed, r), budget=budget, r_bar=r,
                            scale=scale)
            t1 = time.perf_counter()
            out = s.run()
            wall = time.perf_counter() - t1
            chk = verify(inst, out)
            runs.append(dict(method=m, r_bar=r, allowance=budget,
                             evaluations=out["evaluations"], wall_clock_s=wall,
                             makespan=chk["makespan"], max_risk=chk["max_risk"],
                             feasible=chk["feasible"],
                             ceiling_met=bool(chk["feasible"] and
                                              chk["max_risk"] <= r + 1e-12)))
    return dict(family=family, seed=seed, n=inst.n, K=inst.K, P=inst.P,
                allowance=budget, r_min=rmin, exact=exact,
                exact_seconds=exact_seconds, tsp_seconds=ex.tsp_seconds, runs=runs)


def ortools_instance(job):
    """OR-Tools runs; executed in worker processes that never import HiGHS
    (the two libraries bundle incompatible HiGHS symbols)."""
    family, seed = job
    inst = instances().build(family, seed)
    runs = []
    for r in common.CEILINGS:
        o = ORToolsCeilingSolver(inst, r, time_limit=ORTOOLS_SECONDS).run()
        runs.append(dict(method="OR-Tools", r_bar=r, wall_clock_s=o["wall_clock_s"],
                         makespan=o.get("makespan"), max_risk=o.get("max_risk"),
                         feasible=o.get("feasible", False),
                         ceiling_met=bool(o.get("ceiling_met", False))))
        from hgs_reference import HGSCeilingSolver
        h = HGSCeilingSolver(inst, r, time_limit=ORTOOLS_SECONDS).run()
        runs.append(dict(method="HGS", r_bar=r, wall_clock_s=h["wall_clock_s"],
                         makespan=h.get("makespan"), max_risk=h.get("max_risk"),
                         feasible=h.get("feasible", False),
                         ceiling_met=bool(h.get("ceiling_met", False))))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--families", nargs="*", default=common.SMALL_FAMILIES)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="exact_small.json")
    a = ap.parse_args()
    jobs = [(f, s) for f in a.families for s in common.SEEDS[:a.seeds]]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(run_instance, jobs, chunksize=1))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:      # fresh workers
        for rec, runs in zip(recs, ex.map(ortools_instance, jobs, chunksize=1)):
            rec["runs"].extend(runs)
    payload = dict(meta=common.meta(block="exact_small", families=a.families,
                                    seeds=common.SEEDS[:a.seeds],
                                    ceilings=common.CEILINGS, methods=common.METHODS,
                                    ortools_seconds=ORTOOLS_SECONDS,
                                    wall_clock_s=time.perf_counter() - t0),
                   raw=recs)
    path = common.dump(a.out, payload)
    print(f"[exact_small] {len(recs)} instances in {payload['meta']['wall_clock_s']:.0f}s -> {path}")


if __name__ == "__main__":
    main()
