"""Routing-engine references on the benchmark families through the crisp equivalent.

    python3 experiments/run_reference.py --workers 2 --seconds 10   # results/ortools.json
    python3 experiments/run_reference.py --engine hgs --seconds 1 --out hgs_1s.json

Run on an otherwise idle machine: the stopping rule is wall-clock time.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import common
from common import instances
from ortools_reference import ORToolsCeilingSolver

ENGINES = {"ortools": ORToolsCeilingSolver}

SECONDS = 10.0


def run_one(job):
    family, seed, r_bar, seconds, engine = job
    inst = instances().build(family, seed)
    if engine == "hgs":
        from hgs_reference import HGSCeilingSolver
        o = HGSCeilingSolver(inst, r_bar, time_limit=seconds).run()
    else:
        o = ORToolsCeilingSolver(inst, r_bar, time_limit=seconds).run()
    o.update(family=family, seed=seed)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--families", nargs="*", default=common.BENCH_FAMILIES)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="ortools.json")
    ap.add_argument("--engine", choices=("ortools", "hgs"), default="ortools")
    a = ap.parse_args()
    jobs = [(f, s, r, a.seconds, a.engine) for f in a.families for s in common.SEEDS[:a.seeds]
            for r in common.CEILINGS]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        runs = list(ex.map(run_one, jobs, chunksize=1))
    payload = dict(meta=common.meta(block=a.engine, seconds=a.seconds,
                                    families=a.families, ceilings=common.CEILINGS,
                                    wall_clock_s=time.perf_counter() - t0),
                   raw=runs)
    path = common.dump(a.out, payload)
    print(f"[{a.engine}] {len(runs)} runs -> {path}")


if __name__ == "__main__":
    main()
