"""
Expert panels and worst-case (envelope) credibility of a cell.

A panel gives, for each leg type (travel, service), several triangular
assessments expressed relative to the reference modal value b of a leg:

    expert j:  (b * m_j * (1 - l_j),  b * m_j,  b * m_j * (1 + u_j)).

Modal disagreement (m_j != 1), spread disagreement and asymmetric spreads are
all allowed.  The same panel can encode a hesitant linguistic assessment whose
terms carry triangular semantics.

PanelFN is the cell-level object.  It stores the modal travel and service sums
of the cell, from which every expert's sup--min sum follows, and evaluates the
*worst-case* credibility over all leg-wise expert choices.  By the robust
envelope corollary this equals the credibility of the fuzzy number whose
alpha-cut endpoints are the leg-wise maxima, so the right and left endpoints
of PanelFN are exactly those envelopes and CredibilisticCeiling applies
unchanged (polymorphism through the FuzzyNumber interface).
"""

from __future__ import annotations

from dataclasses import dataclass

from fuzzy_core import FuzzyNumber, TriangularFN
from problem import MembershipFactory


@dataclass(frozen=True)
class ExpertAssessment:
    """One triangular assessment relative to the reference modal value."""

    mode: float = 1.0      # modal multiplier m_j
    left: float = 0.25     # relative left spread l_j
    right: float = 0.25    # relative right spread u_j

    def normalised(self) -> tuple[float, float, float]:
        m = self.mode
        return (m * (1.0 - self.left), m, m * (1.0 + self.right))


class _Envelope:
    """Upper envelope of the alpha-cut endpoints of one leg type."""

    def __init__(self, experts: list[ExpertAssessment]):
        self.tfn = [e.normalised() for e in experts]

    def right(self, alpha: float) -> float:
        return max(c - alpha * (c - b) for a, b, c in self.tfn)

    def left(self, beta: float) -> float:
        return max(a + beta * (b - a) for a, b, c in self.tfn)

    def consensus(self) -> tuple[float, float, float]:
        k = len(self.tfn)
        return tuple(sum(t[i] for t in self.tfn) / k for i in range(3))

    def breakpoints(self) -> list[float]:
        """Alpha values in (0,1) where the maximising line may change."""
        pts = set()
        for side in (0, 1):
            lines = ([(c, -(c - b)) for a, b, c in self.tfn] if side == 0 else
                     [(a, (b - a)) for a, b, c in self.tfn])
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    (p1, s1), (p2, s2) = lines[i], lines[j]
                    if abs(s1 - s2) > 1e-15:
                        x = (p2 - p1) / (s1 - s2)
                        if 0.0 < x < 1.0:
                            pts.add(x)
        return sorted(pts)


class ExpertPanel:
    """Assessments per leg type; consensus and envelope views."""

    def __init__(self, travel: list[ExpertAssessment],
                 service: list[ExpertAssessment], name: str = "panel"):
        self.name = name
        self.env = {"travel": _Envelope(travel), "service": _Envelope(service)}
        grid = set(self.env["travel"].breakpoints()) | set(self.env["service"].breakpoints())
        self.grid = [0.0] + sorted(grid) + [1.0]

    def consensus_spreads(self, kind: str) -> tuple[float, float, float]:
        return self.env[kind].consensus()

    def as_dict(self):
        return {k: [list(t) for t in v.tfn] for k, v in self.env.items()}


class PanelFN(FuzzyNumber):
    """Worst-case (envelope) fuzzy duration of a sum of panel-assessed legs."""

    __slots__ = ("panel", "T", "S")

    def __init__(self, panel: ExpertPanel, travel_modal: float = 0.0,
                 service_modal: float = 0.0):
        self.panel = panel
        self.T = float(travel_modal)
        self.S = float(service_modal)

    # --- envelopes of the cut endpoints ---------------------------------
    def right_endpoint(self, alpha):
        e = self.panel.env
        return self.T * e["travel"].right(alpha) + self.S * e["service"].right(alpha)

    def left_endpoint(self, beta):
        e = self.panel.env
        return self.T * e["travel"].left(beta) + self.S * e["service"].left(beta)

    # --- shape descriptors ------------------------------------------------
    @property
    def core(self):
        e = self.panel.env
        lo = self.left_endpoint(1.0)
        return (lo, self.right_endpoint(1.0))

    @property
    def modal(self):
        return self.left_endpoint(1.0)  # actual envelope core; workload uses instance reference costs

    @property
    def support(self):
        # support[0] is read by the solvers as the optimistic duration of the
        # possibilistic feasibility constraint; the robust value is the
        # largest optimistic duration over all admissible choices.
        return (self.left_endpoint(0.0), self.right_endpoint(0.0))

    def _solve(self, f, target, increasing):
        """Root of a monotone piecewise-linear envelope on the panel grid."""
        g = self.panel.grid
        vals = [f(x) for x in g]
        for i in range(len(g) - 1):
            v0, v1 = vals[i], vals[i + 1]
            lo, hi = (v0, v1) if increasing else (v1, v0)
            if lo - 1e-12 <= target <= hi + 1e-12:
                if abs(v1 - v0) < 1e-15:
                    return g[i]
                return g[i] + (target - v0) * (g[i + 1] - g[i]) / (v1 - v0)
        return g[-1] if (increasing and target >= vals[-1]) or \
            (not increasing and target <= vals[-1]) else g[0]

    def overtime_risk(self, limit: float) -> float:
        """sup over leg-wise expert choices of Cr{sigma > limit}."""
        if self.T == 0.0 and self.S == 0.0:
            return float(0.0 > limit)
        if limit >= self.right_endpoint(0.0):
            return 0.0
        if limit >= self.right_endpoint(1.0):
            alpha = self._solve(self.right_endpoint, limit, increasing=False)
            return 0.5 * alpha
        if limit < self.left_endpoint(0.0):
            return 1.0
        beta = self._solve(self.left_endpoint, limit, increasing=True)
        return 1.0 - 0.5 * beta

    def credibility_le(self, t: float) -> float:
        return 1.0 - self.overtime_risk(t)

    def membership(self, x: float) -> float:
        """Membership of the envelope fuzzy number (worst-case reading)."""
        if self.T == 0.0 and self.S == 0.0:
            return 1.0 if x == 0.0 else 0.0
        if x >= self.right_endpoint(1.0):
            if x >= self.right_endpoint(0.0):
                return 0.0
            return self._solve(self.right_endpoint, x, increasing=False)
        if x <= self.left_endpoint(0.0):
            return 0.0
        return self._solve(self.left_endpoint, x, increasing=True)

    def __add__(self, other):
        return PanelFN(self.panel, self.T + other.T, self.S + other.S)

    def __repr__(self):
        return f"PanelFN(T={self.T:.1f},S={self.S:.1f})"


class PanelFactory(MembershipFactory):
    """Membership factory producing envelope legs for a given panel."""

    def __init__(self, panel: ExpertPanel):
        super().__init__("tfn")
        self.panel = panel

    def zero(self):
        return PanelFN(self.panel)

    def make(self, modal: float):
        return PanelFN(self.panel, travel_modal=modal)

    def make_travel(self, modal: float):
        return PanelFN(self.panel, travel_modal=modal)

    def make_service(self, modal: float):
        return PanelFN(self.panel, service_modal=modal)


class ConsensusFactory(MembershipFactory):
    """Single triangular membership per leg type: the panel mean."""

    def __init__(self, panel: ExpertPanel):
        super().__init__("tfn")
        self.t = panel.consensus_spreads("travel")
        self.s = panel.consensus_spreads("service")

    def _tfn(self, modal, spec):
        a, b, c = spec
        return TriangularFN(modal * a, modal * b, modal * c)

    def zero(self):
        return TriangularFN(0.0, 0.0, 0.0)

    def make(self, modal):
        return self._tfn(modal, self.t)

    def make_travel(self, modal):
        return self._tfn(modal, self.t)

    def make_service(self, modal):
        return self._tfn(modal, self.s)


def default_panel() -> ExpertPanel:
    """Three simulated assessments per leg type used in the scenario study.

    Expert 2 is the nominal assessment of the C2/F2 families (25%/20%
    symmetric spreads); experts 1 and 3 disagree on the mode (-3%/+3%) and on
    the spreads, including an asymmetric pessimistic assessment.
    """
    travel = [ExpertAssessment(0.97, 0.20, 0.20),
              ExpertAssessment(1.00, 0.25, 0.25),
              ExpertAssessment(1.03, 0.25, 0.30)]
    service = [ExpertAssessment(0.97, 0.15, 0.15),
               ExpertAssessment(1.00, 0.20, 0.20),
               ExpertAssessment(1.03, 0.20, 0.25)]
    return ExpertPanel(travel, service, name="simulated three-assessor panel")
