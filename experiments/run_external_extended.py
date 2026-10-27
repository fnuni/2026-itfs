"""Extended replay on the eight public Amazon routes.

    python3 experiments/run_external_extended.py --block equal  --workers 2
    python3 experiments/run_external_extended.py --block long   --workers 2
    python3 experiments/run_external_extended.py --block shapes --workers 2
    python3 experiments/run_external_extended.py --block iterations --workers 2

Three questions left open by the four-second comparison are tested on one
machine, with every method run in the same block:

  equal   all methods at 4 s, three repetitions; adds a dense twelve-ceiling
          sweep and an adaptive sweep to the four-ceiling reference;
  long    all methods at 16 s, one repetition (seed 10): does the advantage
          of the modal-position ablation persist with a larger allowance?
  shapes  all methods at 4 s, one repetition, under three further membership
          constructions (symmetric 25/20 %, R1 expert median spreads, seeded random
          heterogeneous spreads): do the conclusions depend on the scenario
          memberships?
  iterations  CC-full and CC-modal with the same 150 destroy-repair iterations
          (about the CC-full count in 4 s), three repetitions: is the lead of the
          modal-position ablation a per-iteration or a throughput effect?

Every allowance includes instance and model construction, seeding and search;
final verification is outside it.  The same feasible-first J_alpha selects and
reports every plan.  Records: results/external_extended_<block>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from external_instances_v4 import ExternalMatrixInstance, load_external_records  # noqa: E402
from external_solver_v4 import EndpointCCLNS                                    # noqa: E402
from fuzzy_core import TriangularFN                                              # noqa: E402
from problem import MembershipFactory                                            # noqa: E402
import run_external_v4 as base                                                   # noqa: E402

ALPHAS = (0.1, 0.3, 0.7)
COARSE = (0.25, 0.40, 0.60, 0.80)
DENSE = tuple(round(0.25 + 0.05 * i, 2) for i in range(12))          # 0.25 ... 0.80
METHODS = ("OR-sweep", "OR-dense", "OR-adaptive", "CC-full", "CC-modal", "OR+CC")


# =============================================================================
# membership constructions
# =============================================================================
class MembershipScenario(ABC):
    """Relative left/right spreads of directed travel and service legs."""

    name = "scenario"

    @abstractmethod
    def travel(self, i: int, j: int, rec: dict) -> tuple[float, float]:
        ...

    @abstractmethod
    def service(self, i: int, rec: dict) -> tuple[float, float]:
        ...


class PublishedPattern(MembershipScenario):
    """The frozen heterogeneous pattern of the main replay."""

    name = "published"

    def travel(self, i, j, rec):
        z = rec["zones"]
        cross = bool(z[i] and z[j] and z[i] != z[j])
        return (0.10 + 0.05 * ((i + 2 * j) % 3), 0.20 + 0.10 * ((3 * i + j) % 4) + 0.10 * cross)

    def service(self, i, rec):
        c = rec["package_counts"][i]
        return (0.10 + 0.025 * (c % 3), 0.20 + 0.05 * min(c, 5))


class UniformSymmetric(MembershipScenario):
    """The convention of the synthetic families: 25 % travel, 20 % service."""

    name = "symmetric"

    def travel(self, i, j, rec):
        return (0.25, 0.25)

    def service(self, i, rec):
        return (0.20, 0.20)


class ElicitedMedian(MembershipScenario):
    """Median relative spreads of the R1 expert triangles (S1 travel, S3 service)."""

    name = "elicited"

    def __init__(self):
        s = json.loads((ROOT / "results" / "elicited_summary.json").read_text())["summary"]["scenarios"]
        self.t = (s["s1"]["left_spread"]["median"], s["s1"]["right_spread"]["median"])
        self.s = (s["s3"]["left_spread"]["median"], s["s3"]["right_spread"]["median"])

    def travel(self, i, j, rec):
        return self.t

    def service(self, i, rec):
        return self.s


class SeededRandom(MembershipScenario):
    """Independent seeded spreads per leg: left U[.05,.30], right U[.10,.80]."""

    name = "random"

    def _u(self, *key):
        h = hashlib.sha256(("|".join(map(str, key))).encode()).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    def travel(self, i, j, rec):
        r = self._u("t", rec["id"], i, j)
        return (r.uniform(0.05, 0.30), r.uniform(0.10, 0.80))

    def service(self, i, rec):
        r = self._u("s", rec["id"], i)
        return (r.uniform(0.05, 0.30), r.uniform(0.10, 0.80))


SCENARIOS = {c.name: c for c in (PublishedPattern, UniformSymmetric, ElicitedMedian, SeededRandom)}


class ScenarioInstance(ExternalMatrixInstance):
    """Amazon scaffold whose leg memberships follow a MembershipScenario."""

    def __init__(self, rec, scenario: MembershipScenario):
        self.n, self.K, self.P, self.L = rec["n"], rec["K"], rec["P"], rec["L"]
        self.coords, self.service_modal = rec["coordinates"], rec["service_minutes"]
        self.factory, self.cell_evals = MembershipFactory("tfn"), 0
        self.speed, self._dist, self.meta = 1.0, rec["travel_minutes"], rec
        self.scenario = scenario

        def travel(i, j):
            m = self._dist[i][j]
            if i == j:
                return TriangularFN(0, 0, 0)
            left, right = scenario.travel(i, j, rec)
            return TriangularFN(m * (1 - left), m, m * (1 + right))

        self._travel = [[travel(i, j) for j in range(self.n + 1)] for i in range(self.n + 1)]
        self._service = [TriangularFN(0, 0, 0)]
        for i, m in enumerate(self.service_modal[1:], 1):
            left, right = scenario.service(i, rec)
            self._service.append(TriangularFN(m * (1 - left), m, m * (1 + right)))
        self.tfn_travel = [[(f.a, f.b, f.c) for f in row] for row in self._travel]
        self.tfn_service = [(f.a, f.b, f.c) for f in self._service]


# =============================================================================
# reference sweeps
# =============================================================================
class CeilingSweep(ABC):
    """Epsilon-constraint reference: OR-Tools at several ceilings, best J kept."""

    def __init__(self, inst, alpha, scale):
        self.inst, self.alpha, self.scale = inst, alpha, scale
        self.stages = []
        self.best = None

    def solve(self, ceiling, until):
        out = base.ortools_one(self.inst, ceiling, until)
        if out.get("found") and out["feasible"]:
            out["J"] = self.alpha * out["makespan"] / self.scale + (1 - self.alpha) * out["fff"]
            if self.best is None or out["J"] < self.best["J"]:
                self.best = out
        self.stages.append({k: v for k, v in out.items() if k in
                            ("ceiling", "found", "feasible", "J", "elapsed_s", "max_risk", "fff")})
        return out

    @abstractmethod
    def run(self, start, deadline):
        ...


class FixedSweep(CeilingSweep):
    """Equal cumulative time checkpoints over a fixed list of ceilings."""

    def __init__(self, inst, alpha, scale, ceilings):
        super().__init__(inst, alpha, scale)
        self.ceilings = ceilings

    def run(self, start, deadline):
        span = deadline - start
        for i, c in enumerate(self.ceilings):
            if time.perf_counter() >= deadline:
                break
            self.solve(c, start + span * (i + 1) / len(self.ceilings))
        return self.best


class AdaptiveSweep(CeilingSweep):
    """Coarse sweep on 40 % of the time, then local refinement around the best.

    Each refinement step tries the midpoints between the incumbent ceiling and
    its neighbours at half the previous step, keeping the best J.
    """

    def run(self, start, deadline):
        span = deadline - start
        coarse_end = start + 0.4 * span
        for i, c in enumerate(COARSE):
            self.solve(c, start + 0.4 * span * (i + 1) / len(COARSE))
        if self.best is None:
            return None
        planned, done, step = 6, 0, 0.075
        slot = (deadline - coarse_end) / planned
        tried = set(COARSE)
        while done < planned and step >= 0.005 and time.perf_counter() < deadline:
            c0 = self.best["ceiling"]
            for c in (round(c0 - step, 4), round(c0 + step, 4)):
                if done >= planned or not 0.05 <= c <= 0.95 or c in tried:
                    continue
                tried.add(c)
                done += 1
                self.solve(c, coarse_end + slot * done)
            step /= 2
        return self.best


# =============================================================================
def run_method(method, rec, scenario, alpha, seed, seconds):
    start = time.perf_counter()
    deadline = start + seconds
    inst = ScenarioInstance(rec, scenario)
    scale = rec["workload_scale"]
    sweep, seed_out = None, None
    if method in ("OR-sweep", "OR+CC"):
        sweep = FixedSweep(inst, alpha, scale, COARSE)
        seed_out = sweep.run(start, deadline if method == "OR-sweep" else start + seconds / 2)
    elif method == "OR-dense":
        sweep = FixedSweep(inst, alpha, scale, DENSE)
        seed_out = sweep.run(start, deadline)
    elif method == "OR-adaptive":
        sweep = AdaptiveSweep(inst, alpha, scale)
        seed_out = sweep.run(start, deadline)
    if method.startswith("OR-"):
        out = dict(seed_out) if seed_out else dict(found=False, feasible=False)
    else:
        solver = EndpointCCLNS(inst, alpha=alpha, seed=seed, workload_scale=scale, deadline=deadline,
                               all_positions=method != "CC-modal",
                               initial_cells=seed_out["solution"] if seed_out else None)
        out = solver.run()
        out["found"] = True
    elapsed = time.perf_counter() - start
    if out.get("solution"):
        base.verify(inst, out)
        out["J"] = alpha * out["makespan"] / scale + (1 - alpha) * out["fff"] if out["feasible"] else None
    if seed_out and method == "OR+CC" and out["feasible"]:
        assert out["J"] <= seed_out["J"] + 1e-9
    keep = ("found", "feasible", "J", "makespan", "fff", "max_risk", "ceiling", "iterations",
            "endpoint_candidates", "evaluations")
    rec_out = {k: out.get(k) for k in keep}
    rec_out['solution']=out.get('solution')
    rec_out.update(instance=rec["id"], method=method, scenario=scenario.name, alpha=alpha, seed=seed,
                   budget_s=seconds, wall_clock_s=elapsed, workload_scale=scale,
                   normalized_workload=(out["makespan"] / scale) if out.get("makespan") else None,
                   OR_stages=sweep.stages if sweep else None,
                   seed_J=seed_out["J"] if (seed_out and method == "OR+CC") else None)
    return rec_out


def run_iteration_matched(method, rec, scenario, alpha, seed, iterations):
    """CC-full and CC-modal with the same number of destroy-repair iterations."""
    start = time.perf_counter()
    inst = ScenarioInstance(rec, scenario)
    scale = rec["workload_scale"]
    solver = EndpointCCLNS(inst, alpha=alpha, seed=seed, workload_scale=scale, deadline=start + 900.0,
                           all_positions=method != "CC-modal", max_iterations=iterations)
    out = solver.run()
    elapsed = time.perf_counter() - start
    base.verify(inst, out)
    out["J"] = alpha * out["makespan"] / scale + (1 - alpha) * out["fff"] if out["feasible"] else None
    keep = ("feasible", "J", "makespan", "fff", "max_risk", "iterations", "endpoint_candidates", "evaluations")
    rec_out = {k: out.get(k) for k in keep}
    rec_out.update(instance=rec["id"], method=method, scenario=scenario.name, alpha=alpha, seed=seed,
                   iteration_budget=iterations, wall_clock_s=elapsed, workload_scale=scale,
                   normalized_workload=out["makespan"] / scale)
    return rec_out


def job(args):
    idx, rec, scen_name, alpha, seed, seconds = args
    scenario = SCENARIOS[scen_name]()
    if isinstance(seconds, dict):                                  # iteration-matched block
        return [run_iteration_matched(m, rec, scenario, alpha, seed, seconds["iterations"])
                for m in ("CC-full", "CC-modal")]
    order = METHODS[idx % len(METHODS):] + METHODS[:idx % len(METHODS)]
    return [run_method(m, rec, scenario, alpha, seed, seconds) for m in order]


BLOCKS = {
    "equal": dict(scenarios=("published",), seeds=(10, 20, 30), seconds=4.0),
    "long": dict(scenarios=("published",), seeds=(10,), seconds=16.0),
    "shapes": dict(scenarios=("symmetric", "elicited", "random"), seeds=(10,), seconds=4.0),
    "iterations": dict(scenarios=("published",), seeds=(10, 20, 30), seconds=dict(iterations=150)),
}


def check_published_pattern(records):
    """The published scenario reproduces the memberships of the main replay."""
    a, b = ExternalMatrixInstance(records[0]), ScenarioInstance(records[0], PublishedPattern())
    assert a.tfn_travel == b.tfn_travel and a.tfn_service == b.tfn_service


def warm_libraries():
    """Load solver libraries in every worker before any per-run timer starts."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=tuple(BLOCKS), required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument('--scenario', choices=tuple(SCENARIOS), help='Refresh one scenario, retaining unaffected scenario runs')
    a = ap.parse_args()
    records = load_external_records()
    check_published_pattern(records)
    cfg = dict(BLOCKS[a.block])
    if a.scenario:
        if a.scenario not in cfg['scenarios']:
            raise ValueError('Scenario does not belong to this block')
        cfg['scenarios'] = (a.scenario,)
    methods = ("CC-full", "CC-modal") if isinstance(cfg["seconds"], dict) else METHODS
    protocol = dict(block=a.block, methods=methods, coarse_ceilings=COARSE, dense_ceilings=DENSE,
                    adaptive="coarse sweep on 40% of the allowance, then up to six refinement ceilings "
                             "at halving steps from 0.075 around the incumbent",
                    alphas=ALPHAS, workers=a.workers, python=sys.version.split()[0],
                    platform=platform.platform(), cpu_count=os.cpu_count(), **cfg,
                    software_startup="Worker library imports precede per-run timers for every method",
                    time_includes="instance construction, OR model construction and search, seeding, "
                                  "CC-LNS search; final verification outside the allowance")
    if 'elicited' in cfg['scenarios']:
        summary=json.loads((ROOT/'results/elicited_summary.json').read_text())
        protocol['elicitation']=dict(session=summary['summary']['session'],
            participants=summary['summary']['n'],
            data_sha256=hashlib.sha256((ROOT/'data/reliability20/responses_numeric.csv').read_bytes()).hexdigest(),
            travel_spreads=ElicitedMedian().t, service_spreads=ElicitedMedian().s)
    jobs = [(i, r, s, al, sd, cfg["seconds"]) for i, (r, s, al, sd) in enumerate(
        (r, s, al, sd) for s in cfg["scenarios"] for r in records for al in ALPHAS for sd in cfg["seeds"])]
    out_path = ROOT / "results" / f"external_extended_{a.block}.json"
    original=json.loads(out_path.read_text()) if a.scenario and out_path.exists() else None
    working_path=out_path.with_suffix('.partial') if original else out_path
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=warm_libraries) as ex:
        for k, res in enumerate(ex.map(job, jobs, chunksize=1), 1):
            rows.extend(res)
            working_path.write_text(json.dumps(dict(protocol=protocol, raw=rows), indent=1, allow_nan=False))
            print(f"{a.block}: {k}/{len(jobs)} groups, {time.perf_counter() - t0:.0f}s", flush=True)
    if original:
        retained=[r for r in original['raw'] if r['scenario']!=a.scenario]
        executions=original['protocol'].get('executions',{})
        for sc in {r['scenario'] for r in retained}:
            if sc not in executions:
                executions[sc]=dict(original['protocol'],scenarios=[sc],
                    provenance='Scenario records of an earlier execution of the same block')
        executions[a.scenario]=protocol
        merged=dict(block=a.block,scenarios=sorted(executions),executions=executions,
            comparisons='Paired method comparisons within each scenario execution only')
        out_path.write_text(json.dumps(dict(protocol=merged,raw=retained+rows),indent=1,allow_nan=False))
        working_path.unlink()


if __name__ == "__main__":
    main()
