"""Ceiling study: six methods, five families, ten seeds, four ceilings.

    python3 experiments/run_ceiling.py --workers 2           # results/ceiling.json
    python3 experiments/run_ceiling.py --families C2 --seeds 1 --out smoke.json

Every method solves  min workload  s.t.  max_kh r_kh <= r_bar  (and
possibilistic feasibility) under the common protocol of common.py.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import common
from common import (instances, AllowanceBook, common_scale, make_solver, verify,
                    solver_seed)

_BOOK = None


def run_one(job):
    global _BOOK
    family, seed, method, r_bar = job
    _BOOK = _BOOK or AllowanceBook()
    inst = instances().build(family, seed)
    budget = _BOOK.get(family, seed)
    scale = common_scale(inst)
    solver = make_solver(method, inst, alpha=common.REPORT_ALPHA,
                         seed=solver_seed(seed, r_bar), budget=budget,
                         r_bar=r_bar, scale=scale)
    t0 = time.perf_counter()
    out = solver.run()
    wall = time.perf_counter() - t0
    chk = verify(inst, out)
    return dict(family=family, seed=seed, method=method, r_bar=r_bar,
                allowance=budget, evaluations=out["evaluations"],
                wall_clock_s=wall, workload_scale=scale,
                makespan=chk["makespan"], max_risk=chk["max_risk"], fff=chk["fff"],
                feasible=chk["feasible"],
                ceiling_met=bool(chk["feasible"] and chk["max_risk"] <= r_bar + 1e-12),
                R=[[round(v, 12) for v in row] for row in out["R"]],
                solution=out["solution"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--families", nargs="*", default=common.BENCH_FAMILIES)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--methods", nargs="*", default=common.METHODS)
    ap.add_argument("--out", default="ceiling.json")
    ap.add_argument("--merge", action="store_true",
                    help="replace the records of --methods in an existing --out file")
    a = ap.parse_args()
    jobs = [(f, s, m, r) for f in a.families for s in common.SEEDS[:a.seeds]
            for r in common.CEILINGS for m in a.methods]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        runs = list(ex.map(run_one, jobs, chunksize=1))
    wall = time.perf_counter() - t0
    if a.merge and (common.RESULTS / a.out).exists():
        old = common.load(a.out)
        keep = [r for r in old["raw"] if r["method"] not in a.methods]
        meta = old["meta"]
        meta.setdefault("updates", []).append(dict(
            created=time.strftime("%Y-%m-%d %H:%M:%S"), methods=a.methods,
            wall_clock_s=wall, note="records of these methods regenerated; "
            "all runs are deterministic functions of the seeds"))
        payload = dict(meta=meta, raw=keep + runs)
    else:
        payload = dict(meta=common.meta(block="ceiling", families=a.families,
                                        seeds=common.SEEDS[:a.seeds], methods=a.methods,
                                        ceilings=common.CEILINGS, wall_clock_s=wall,
                                        protocol=common.__doc__.strip()),
                       raw=runs)
    runs = payload["raw"]
    path = common.dump(a.out, payload)
    print(f"[ceiling] {len(runs)} verified runs ({wall:.0f}s) -> {path}")


if __name__ == "__main__":
    main()
