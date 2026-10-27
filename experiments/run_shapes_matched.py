"""Endpoint-matched membership readings.

    python3 experiments/run_shapes_matched.py        # results/shapes_matched.json

The support-matched study (run_biobjective.py shapes) reads the same triple
(a, b, c) as a triangle, a trapezoid and a Gaussian; their planning durations
differ, so their feasible sets differ.  Here the trapezoid and the Gaussian are
instead calibrated to share the triangle's planning duration at the matching
ceiling r* = 0.30 (Theorem: the ceiling reads only that endpoint):

  trapezoid  core [b - 0.2(b - a), b + 0.2(c - b)], right support end chosen so
             that R(2 r*) equals the triangle's;
  Gaussian   mode b, spread s = (1 - 2 r*)(c - b) / sqrt(-2 ln 2 r*).

The OR-Tools plans of the ceiling study (10 s, crisp equivalent of the
triangle) at r = 0.30 and 0.40 are re-evaluated under the three readings: at
r* the theorem predicts identical ceiling decisions while the FFF, which reads
whole memberships, changes; at 0.40 the readings no longer share the endpoint.
No search is run; every number is a deterministic re-evaluation.
"""

from __future__ import annotations

import math
import statistics as st

import common
from common import instances, load, dump
from aggregators import FFF_REPORT, max_risk
from crisp_equivalent import CredibilisticCeiling
from fuzzy_core import TriangularFN, TrapezoidalFN, GaussianFN
from problem import Instance, Solution

R_MATCH = 0.30
FAMILIES = ("C2", "C4", "F2")
READINGS = ("tfn", "trap-matched", "gauss-matched")
CEILINGS = (0.25, 0.30, 0.40)


class MatchedFactory:
    """Per-leg memberships sharing the triangle's planning duration at R_MATCH."""

    def __init__(self, reading, travel_spread, service_spread, r_match=R_MATCH):
        self.reading, self.t, self.s, self.a0 = reading, travel_spread, service_spread, 2 * r_match

    def zero(self):
        if self.reading == "tfn":
            return TriangularFN(0.0, 0.0, 0.0)
        if self.reading == "trap-matched":
            return TrapezoidalFN(0.0, 0.0, 0.0, 0.0)
        return GaussianFN(0.0, 1e-9)

    def _make(self, modal, spread):
        a, b, c = modal * (1 - spread), modal, modal * (1 + spread)
        if self.reading == "tfn":
            return TriangularFN(a, b, c)
        if self.reading == "trap-matched":
            b1, b2 = b - 0.2 * (b - a), b + 0.2 * (c - b)
            c2 = (c - self.a0 * (c - b) - self.a0 * b2) / (1 - self.a0)
            return TrapezoidalFN(a, b1, b2, c2)
        s = (1 - self.a0) * (c - b) / math.sqrt(-2.0 * math.log(self.a0))
        return GaussianFN(b, max(s, 1e-9))

    def make(self, modal):
        return self._make(modal, self.t)

    def make_travel(self, modal):
        return self._make(modal, self.t)

    def make_service(self, modal):
        return self._make(modal, self.s)


def rebuild(inst, factory):
    return Instance(inst.n, inst.K, inst.P, inst.L, inst.coords,
                    inst.service_modal, inst.speed, factory)


def main():
    plans = [r for r in load("ortools.json")["raw"]
             if r["family"] in FAMILIES and r["r_bar"] in (0.30, 0.40) and r.get("found")]
    design = common.instances().design if hasattr(common.instances(), "design") else None
    from instances_io import ALL_DESIGN
    recs = []
    mult = {}
    for fam in FAMILIES:
        d = ALL_DESIGN[fam]
        for rd in READINGS:
            f = MatchedFactory(rd, d["travel_spread"], d["service_spread"])
            mult[f"{fam}|{rd}"] = {str(c): [CredibilisticCeiling(c).planning_duration(f.make_travel(1.0)),
                                            CredibilisticCeiling(c).planning_duration(f.make_service(1.0))]
                                   for c in CEILINGS}
    for r in plans:
        base = instances().build(r["family"], r["seed"])
        d = ALL_DESIGN[r["family"]]
        sol = Solution(base.K, base.P)
        sol.cells = [[list(c) for c in row] for row in r["solution"]]
        rec = dict(family=r["family"], seed=r["seed"], r_bar=r["r_bar"])
        for rd in READINGS:
            inst = rebuild(base, MatchedFactory(rd, d["travel_spread"], d["service_spread"]))
            R, _ = inst.evaluate(sol)
            rec[rd] = dict(max_risk=max_risk(R), fff=FFF_REPORT(R),
                           meets=bool(max_risk(R) <= r["r_bar"] + 1e-9))
        recs.append(rec)
    # summary
    summ = {}
    for rb in (0.30, 0.40):
        rs = [x for x in recs if x["r_bar"] == rb]
        summ[str(rb)] = dict(
            plans=len(rs),
            meets={rd: sum(x[rd]["meets"] for x in rs) for rd in READINGS},
            same_decision={rd: sum(x[rd]["meets"] == x["tfn"]["meets"] for x in rs)
                           for rd in READINGS},
            fff_change_median_percent={rd: 100 * st.median(x[rd]["fff"] / x["tfn"]["fff"] - 1
                                                          for x in rs if x["tfn"]["fff"] > 0)
                                       for rd in READINGS},
            fff_change_range_percent={rd: [100 * min(x[rd]["fff"] / x["tfn"]["fff"] - 1 for x in rs),
                                           100 * max(x[rd]["fff"] / x["tfn"]["fff"] - 1 for x in rs)]
                                      for rd in READINGS})
    path = dump("shapes_matched.json", dict(meta=common.meta(block="shapes_matched",
                                                             r_match=R_MATCH, readings=READINGS,
                                                             source="results/ortools.json"),
                                            multipliers=mult, summary=summ, raw=recs))
    print(summ)
    print(f"[shapes_matched] {len(recs)} plans re-evaluated -> {path}")


if __name__ == "__main__":
    main()
