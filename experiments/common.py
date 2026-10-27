"""
Shared experimental protocol (all drivers import this module).

Common conventions, applied to every method in every experiment:
  * allowance   number of fuzzy cell evaluations spent by a 120-iteration
                reference RALNS (seed 0, alpha 0.5, FFF) on the instance;
                stored once in data/allowances.json;
  * normalisation  the workload term of every scalarised objective is divided
                by the workload of the same deterministic initial plan;
  * incumbent   in ceiling mode every method reports its best *evaluated*
                plan by (possibilistic feasibility, ceiling excess, workload);
  * verification  every returned plan is re-evaluated independently
                (exactly-once coverage, workload, worst-cell credibility,
                optimistic-duration feasibility).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregators import OWAFairness, max_risk, FFF_REPORT          # noqa: E402
from baselines import SOLVER_REGISTRY                             # noqa: E402
from cc_lns import CCLNS                                          # noqa: E402
from instances_io import CanonicalInstances, ALL_DESIGN           # noqa: E402
from problem import Solution                                      # noqa: E402
from solver import RALNS, _State                                  # noqa: E402

RESULTS = REPO / "results"
TABLES = REPO / "tables"
DATA = REPO / "data"
RESULTS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEEDS5 = SEEDS[:5]
CEILINGS = [0.25, 0.30, 0.40, 0.50]
BENCH_FAMILIES = ["C1", "C2", "C3", "C4", "F2"]
SMALL_FAMILIES = ["S10", "S12", "F12", "F16"]
METHODS = ["CC-LNS", "LNS-AO", "IBEA", "NSGA-II", "MOEA/D", "FuzzMH"]
REF_ITERS = 120
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
REPORT_ALPHA = 0.3

_CI: dict[float, CanonicalInstances] = {}


def instances(rho: float = 0.85) -> CanonicalInstances:
    if rho not in _CI:
        _CI[rho] = CanonicalInstances(tightness=rho)
    return _CI[rho]


def solver_seed(seed: int, r_bar: float | None, salt: int = 0) -> int:
    """Seed rule shared by all methods."""
    return seed * 41 + (int(round(r_bar * 100)) if r_bar is not None else 0) + salt


# --- allowances --------------------------------------------------------------
class AllowanceBook:
    """Persistent record of the counted-evaluation allowance per instance."""

    path = DATA / "allowances.json"

    def __init__(self):
        self.book = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(family, seed, shape="tfn", rho=0.85):
        return f"{family}|{seed}|{shape}|{rho:.2f}"

    @staticmethod
    def compute(inst) -> int:
        probe = RALNS(inst, OWAFairness(), alpha=0.5, iterations=REF_ITERS, seed=0)
        probe.run()
        return int(probe.evaluations)

    def get(self, family, seed, shape="tfn", rho=0.85):
        k = self.key(family, seed, shape, rho)
        if k not in self.book:
            inst = instances(rho).build(family, seed, shape)
            self.book[k] = self.compute(inst)
        return self.book[k]

    def save(self):
        self.path.write_text(json.dumps(dict(sorted(self.book.items())), indent=1))


def common_scale(inst) -> float:
    """Workload of the deterministic initial plan (common normalisation)."""
    init = RALNS(inst, OWAFairness(), seed=0).initial()
    return max(_State(inst, init).makespan, 1.0)


# --- methods -----------------------------------------------------------------
def make_solver(method, inst, *, alpha, seed, budget, r_bar=None, scale=None,
                aggregator=None, **variant):
    agg = aggregator or OWAFairness()
    if method == "CC-LNS":
        return CCLNS(inst, agg, alpha=alpha, iterations=10 ** 9, seed=seed,
                     budget=budget, r_bar=r_bar, use_controller=False,
                     workload_scale=scale, **variant)
    cls = SOLVER_REGISTRY[method]
    return cls(inst, agg, alpha=alpha, budget=budget, seed=seed, r_bar=r_bar,
               workload_scale=scale, ceiling_incumbent=r_bar is not None)


def verify(inst, out: dict) -> dict:
    """Independent re-evaluation of a returned plan."""
    sol = Solution(inst.K, inst.P)
    sol.cells = [[list(c) for c in row] for row in out["solution"]]
    tasks = sorted(t for row in sol.cells for c in row for t in c)
    assert tasks == list(range(1, inst.n + 1)), "coverage violated"
    st = _State(inst, sol)
    assert abs(st.makespan - out["makespan"]) < 1e-6 * max(1.0, st.makespan)
    assert abs(max_risk(st.risk) - out["max_risk"]) < 1e-9
    feasible = st.violation <= 1e-9
    assert feasible == bool(out["feasible"])
    return dict(makespan=st.makespan, max_risk=max_risk(st.risk),
                fff=FFF_REPORT(st.risk), feasible=feasible)


def meta(**extra) -> dict:
    return dict(python=sys.version.split()[0], platform=platform.platform(),
                cpu_count=os.cpu_count(), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                **extra)


def dump(name: str, payload: dict) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=1))
    return path


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())
