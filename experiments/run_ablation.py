"""Component ablation of CC-LNS in ceiling mode (r_bar = 0.30).

    python3 experiments/run_ablation.py --workers 2     # results/ablation.json
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import common
from common import instances, AllowanceBook, common_scale, make_solver, verify

CEILING = 0.30
FAMILIES = ["C2", "C4", "F2"]
VARIANTS = {
    "full": dict(),
    "generic_destroy": dict(generic_destroy=True),
    "no_lexicographic": dict(lexicographic=False),
    "no_surrogate": dict(use_surrogate=False),
    "no_polish": dict(polish=False),
    "constant_penalty": dict(schedule=(1.0, 1.0, 20)),
    "cost_only_repair": dict(cost_only_repair=True),
    "no_annealing": dict(t0=1e-9),
    "random_positions": dict(guided_positions=False),
    "two_positions": dict(positions_per_cell=2),
    "no_random_removal": dict(random_removal=0.0),
}


def run_one(job):
    family, seed, variant = job
    inst = instances().build(family, seed)
    budget = AllowanceBook().get(family, seed)
    s = make_solver("CC-LNS", inst, alpha=common.REPORT_ALPHA, seed=seed * 31,
                    budget=budget, r_bar=CEILING, scale=common_scale(inst),
                    **VARIANTS[variant])
    out = s.run()
    chk = verify(inst, out)
    return dict(family=family, seed=seed, variant=variant, r_bar=CEILING,
                makespan=chk["makespan"], max_risk=chk["max_risk"], fff=chk["fff"],
                feasible=chk["feasible"],
                ceiling_met=bool(chk["feasible"] and chk["max_risk"] <= CEILING + 1e-12),
                evaluations=out["evaluations"], allowance=budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = [(f, s, v) for f in FAMILIES for s in common.SEEDS for v in VARIANTS]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        runs = list(ex.map(run_one, jobs, chunksize=1))
    payload = dict(meta=common.meta(block="ablation", variants=list(VARIANTS),
                                    families=FAMILIES, r_bar=CEILING,
                                    wall_clock_s=time.perf_counter() - t0),
                   raw=runs)
    path = common.dump("ablation.json", payload)
    print(f"[ablation] {len(runs)} runs -> {path}")


if __name__ == "__main__":
    main()
