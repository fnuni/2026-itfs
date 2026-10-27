"""
First-session (R1) answers of the twenty-expert campaign as admissible sets.

Each returned R1 form gives, for four hypothetical scenarios (S1, S2 travel;
S3, S4 service), a representative duration b, direct thresholds at
possibility of exceedance 0.80, 0.60, 0.30 and 0.10 (credibility ceilings
0.40, 0.30, 0.15 and 0.05 on the right branch, manuscript crisp-equivalent
theorem) and, afterwards, the bounds a, c of the possible support when the
respondent judged them finite.  Two readings turn a judgment into the
planning duration of a leg at r = 0.30:

  DirectReading      q060 as answered;
  TriangularReading  q_tri = b + 0.4 (c - b) of the triangle (a, b, c)
                     (triangular corollary), only for complete coherent triples.

Leg memberships of synthetic instances are expressed relative to a reference
modal duration; the reference of a scenario is the median b over the
coherent judgments, so that the planning multiplier of judgment j is
q_j / median(b).  The R2 session is used only for test-retest agreement
(experiments/run_reliability20.py); it does not enter this panel.
"""

from __future__ import annotations

import csv
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from expert_panel import ExpertAssessment, ExpertPanel

LEVEL = 0.60            # possibility level of the primary direct question
CEILING = 0.30          # credibility ceiling read by the primary question
SCENARIOS = ("s1", "s2", "s3", "s4")
LEG_TYPE = {"s1": "travel", "s2": "travel", "s3": "service", "s4": "service"}
THRESHOLDS = (("q080", 0.80), ("q060", 0.60), ("q030", 0.30), ("q010", 0.10))
RESPONSES = Path(__file__).resolve().parents[1] / "data" / "reliability20" / "responses_numeric.csv"


# =============================================================================
# judgments and respondents
# =============================================================================
def _num(x: str):
    x = (x or "").strip()
    return None if x in ("", "--") else float(x.replace(",", "."))


@dataclass(frozen=True)
class Judgment:
    """Answers of one respondent for one scenario (minutes)."""

    scenario: str
    a: float | None
    b: float | None
    c: float | None
    q: float | None
    qprime: float | None = None
    q080: float | None = None
    q010: float | None = None
    confidence: float | None = None
    c_unbounded: bool = False

    @property
    def flags(self) -> tuple[str, ...]:
        """Semantic inconsistency flags of the prespecified protocol."""
        a, b, c, q = self.a, self.b, self.c, self.q
        out = []
        for key in ("a", "b", "c", "q", "qprime", "q080", "q010"):
            value = getattr(self, key)
            if value is not None and value < 0:
                out.append(key + "_negative")
        chain = [v for v in (b, self.q080, q, self.qprime, self.q010) if v is not None]
        if chain != sorted(chain):
            out.append("thresholds_nonmonotone")
        if self.qprime is not None:
            if b is not None and self.qprime < b: out.append("qprime_below_b")
            if q is not None and self.qprime < q: out.append("qprime_below_q")
            if c is not None and self.qprime > c: out.append("qprime_exceeds_c")
        if b is not None and b <= 0:
            out.append("nonpositive_mode")
        if a is not None and b is not None and a > b:
            out.append("a_exceeds_b")
        if b is not None and c is not None and b > c:
            out.append("b_exceeds_c")
        if a is not None and c is not None and a > c:
            out.append("a_exceeds_c")
        if q is not None and b is not None and q < b:
            out.append("q_below_b")
        if q is not None and c is not None and q > c:
            out.append("q_exceeds_c")
        return tuple(out)

    @property
    def coherent(self) -> bool:
        return not self.flags

    @property
    def complete_triangle(self) -> bool:
        return self.coherent and None not in (self.a, self.b, self.c)

    @property
    def q_triangular(self) -> float | None:
        if self.b is None or self.c is None or not self.coherent:
            return None
        return self.b + (1.0 - 2.0 * CEILING) * (self.c - self.b)

    @property
    def delta(self) -> float | None:
        """Direct minus triangular planning duration (complete finite triangle and q)."""
        if self.q is None or self.q_triangular is None or not self.complete_triangle:
            return None
        return self.q - self.q_triangular

    def branch_residual(self, key: str) -> float | None:
        """Departure of a threshold from the straight right branch through b and q060.

        A triangular right branch gives q_p - b = (1 - p)/0.4 (q060 - b);
        the residual is zero for every level exactly when the branch is linear.
        """
        level = dict(THRESHOLDS)[key]
        x = getattr(self, "qprime" if key == "q030" else key)
        if None in (x, self.b, self.q) or not self.coherent:
            return None
        return (x - self.b) - (1.0 - level) / (1.0 - LEVEL) * (self.q - self.b)


@dataclass
class Respondent:
    """One returned form (module code only; no identity)."""

    rid: str
    answers: dict[str, str]
    judgments: dict[str, Judgment]

    def answer(self, key: str) -> str:
        return (self.answers.get(key) or "").strip()

    @property
    def passes_checks(self) -> bool:
        """Both final semantic/monotonicity checks answered with the keys."""
        return self.answer("checks_passed") == "1"

    @property
    def passes_ceiling_item(self) -> bool:
        return self.answer("ceiling_check_passed") == "1"


# =============================================================================
# readings: how a judgment yields a planning duration at r = 0.30
# =============================================================================
class PlanningReading(ABC):
    """Strategy that turns one judgment into a planning duration (or None)."""

    name = "reading"

    @abstractmethod
    def planning(self, j: Judgment) -> float | None:
        ...

    def values(self, judgments) -> list[float]:
        return [v for v in (self.planning(j) for j in judgments) if v is not None]


class DirectReading(PlanningReading):
    """The planning duration as answered to the direct question."""

    name = "direct"

    def planning(self, j):
        return j.q if (j.coherent and j.q is not None) else None


class TriangularReading(PlanningReading):
    """The linear interpolation of a complete coherent triangle."""

    name = "triangular"

    def planning(self, j):
        return j.q_triangular if j.complete_triangle else None


class PairedDirectReading(DirectReading):
    """Direct answer of a respondent who also gave a complete coherent triangle."""

    name = "paired_direct"

    def planning(self, j):
        return j.q if (j.complete_triangle and j.q is not None) else None


class PairedTriangularReading(TriangularReading):
    """Triangular endpoint of a respondent who also answered q directly."""

    name = "paired_triangular"

    def planning(self, j):
        return j.q_triangular if (j.complete_triangle and j.q is not None) else None


READINGS = {"direct": DirectReading(), "triangular": TriangularReading(),
            "paired_direct": PairedDirectReading(), "paired_triangular": PairedTriangularReading()}


# =============================================================================
# the panel
# =============================================================================
@dataclass(frozen=True)
class PlanningBasis:
    """Planning multipliers of unit travel and service legs at r = 0.30."""

    name: str
    travel: float
    service: float
    note: str = ""

    def as_dict(self):
        return dict(name=self.name, travel=self.travel, service=self.service, note=self.note)


@dataclass(frozen=True)
class LegContext:
    """Which scenario represents travel legs and which service legs."""

    name: str
    travel: str
    service: str


CONTEXTS = (LegContext("S1+S3", "s1", "s3"), LegContext("S1+S4", "s1", "s4"),
            LegContext("S2+S3", "s2", "s3"), LegContext("S2+S4", "s2", "s4"))


class ElicitationPanel:
    """One session of the expert campaign and the admissible sets its coherent answers define."""

    def __init__(self, respondents: list[Respondent], name: str = "campaign"):
        self.respondents = respondents
        self.name = name

    # --- construction -------------------------------------------------------
    @classmethod
    def from_files(cls, responses: Path = RESPONSES, session: str = "R1") -> "ElicitationPanel":
        """One session of the public numeric campaign file (default: first session)."""
        if session not in ("R1", "R2"):
            raise ValueError("session must be R1 or R2")
        with Path(responses).open(newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r["session"] == session]
        ids = [r["code"] for r in rows]
        if len(ids) != len(set(ids)) or not rows:
            raise ValueError("Duplicate or missing respondent codes")
        out = []
        for row in rows:
            js = {s: Judgment(s, _num(row[f"{s}_a"]), _num(row[f"{s}_b"]), _num(row[f"{s}_c"]),
                              _num(row[f"{s}_q060"]), qprime=_num(row[f"{s}_q030"]),
                              q080=_num(row[f"{s}_q080"]), q010=_num(row[f"{s}_q010"]),
                              confidence=_num(row[f"{s}_confidence"]),
                              c_unbounded=row[f"{s}_c_unbounded"] == "1") for s in SCENARIOS}
            out.append(Respondent(row["code"], dict(row), js))
        return cls(out, name=f"twenty-expert {session}")

    def subset(self, keep, name: str) -> "ElicitationPanel":
        return ElicitationPanel([r for r in self.respondents if keep(r)], name)

    def __len__(self):
        return len(self.respondents)

    # --- scenario quantities ---------------------------------------------------
    def judgments(self, s: str) -> list[Judgment]:
        return [r.judgments[s] for r in self.respondents]

    def reference_mode(self, s: str) -> float:
        """Median mode over the coherent judgments of the scenario."""
        return statistics.median(j.b for j in self.judgments(s) if j.coherent and j.b is not None)

    def planning_values(self, s: str, reading: PlanningReading) -> list[float]:
        return reading.values(self.judgments(s))

    def consensus_triangle(self, s: str) -> tuple[float, float, float]:
        """Parameter-wise median of the complete coherent triangles."""
        tri = [(j.a, j.b, j.c) for j in self.judgments(s) if j.complete_triangle]
        return tuple(statistics.median(t[i] for t in tri) for i in range(3))

    def multiplier(self, s: str, reading: PlanningReading, how: str,
                   reference: float | None = None) -> float:
        ref = self.reference_mode(s) if reference is None else reference
        if how == "max":
            return max(self.planning_values(s, reading)) / ref
        if how == "median":
            if reading.name == "triangular":
                a, b, c = self.consensus_triangle(s)
                return (b + (1.0 - 2.0 * CEILING) * (c - b)) / ref
            return statistics.median(self.planning_values(s, reading)) / ref
        raise ValueError(how)

    def binding(self, s: str, reading: PlanningReading) -> list[str]:
        """Module codes whose planning duration equals the envelope."""
        vals = {r.rid: reading.planning(r.judgments[s]) for r in self.respondents}
        top = max(v for v in vals.values() if v is not None)
        return [k for k, v in vals.items() if v is not None and abs(v - top) < 1e-12]

    # --- planning bases for a context ---------------------------------------
    def bases(self, ctx: LegContext, reference: "ElicitationPanel | None" = None) -> dict[str, PlanningBasis]:
        """Consensus and envelope multipliers; references from `reference` panel."""
        ref = reference or self
        rt, rs = ref.reference_mode(ctx.travel), ref.reference_mode(ctx.service)
        d, t = READINGS["direct"], READINGS["triangular"]
        return {
            "consensus_direct": PlanningBasis("consensus_direct",
                                              self.multiplier(ctx.travel, d, "median", rt),
                                              self.multiplier(ctx.service, d, "median", rs),
                                              "median direct planning duration"),
            "envelope_direct": PlanningBasis("envelope_direct",
                                             self.multiplier(ctx.travel, d, "max", rt),
                                             self.multiplier(ctx.service, d, "max", rs),
                                             "largest direct planning duration"),
            "consensus_triangular": PlanningBasis("consensus_triangular",
                                                  self.multiplier(ctx.travel, t, "median", rt),
                                                  self.multiplier(ctx.service, t, "median", rs),
                                                  "parameter-wise median triangle"),
            "envelope_triangular": PlanningBasis("envelope_triangular",
                                                 self.multiplier(ctx.travel, t, "max", rt),
                                                 self.multiplier(ctx.service, t, "max", rs),
                                                 "largest triangular planning duration"),
        }

    def bases_paired(self, ctx: LegContext, reference: "ElicitationPanel | None" = None) -> dict[str, PlanningBasis]:
        """Direct and triangular readings on the same respondents.

        Only judgments with both a complete coherent triangle and a direct q
        enter, so the comparison isolates the interpolation from panel coverage.
        """
        ref = reference or self
        out = {}
        for kind in ("paired_direct", "paired_triangular"):
            rd = READINGS[kind]
            for how in ("median", "max"):
                vals = []
                for s in (ctx.travel, ctx.service):
                    r = ref.reference_mode(s)
                    if how == "max":
                        vals.append(max(self.planning_values(s, rd)) / r)
                    elif kind == "paired_direct":
                        vals.append(statistics.median(self.planning_values(s, rd)) / r)
                    else:
                        tri = [(j.a, j.b, j.c) for j in self.judgments(s)
                               if j.complete_triangle and j.q is not None]
                        a_, b_, c_ = (statistics.median(t[i] for t in tri) for i in range(3))
                        vals.append((b_ + (1.0 - 2.0 * CEILING) * (c_ - b_)) / r)
                name = ("consensus_" if how == "median" else "envelope_") + kind
                out[name] = PlanningBasis(name, vals[0], vals[1], f"{how} of {kind} answers")
        return out

    def paired_rids(self, s: str) -> list[str]:
        return [r.rid for r in self.respondents
                if r.judgments[s].complete_triangle and r.judgments[s].q is not None]

    # --- triangular panel in the format of expert_panel ------------------------
    def triangular_panel(self, ctx: LegContext) -> ExpertPanel:
        """Complete coherent triangles relative to the reference modes."""
        def assessments(s):
            ref = self.reference_mode(s)
            return [ExpertAssessment(j.b / ref, 1.0 - j.a / j.b, j.c / j.b - 1.0)
                    for j in self.judgments(s) if j.complete_triangle]
        return ExpertPanel(assessments(ctx.travel), assessments(ctx.service),
                           name=f"elicited triangles {ctx.name}")
