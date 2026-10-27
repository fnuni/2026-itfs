"""
Crisp (deterministic) equivalent of a cellwise credibility ceiling.

Theorem (manuscript, Model and Crisp Equivalent). Let the legs of a vehicle--shift cell have
upper semicontinuous, normal, unimodal memberships with compact cuts and let the cell
duration be their sup--min sum sigma = xi_1 (+) ... (+) xi_m.  Then

    0 <= r < 1/2 :  Cr{sigma > L} <= r  <=>  sum_e R_e(2r)    <= L,
    1/2 <= r < 1 :  Cr{sigma > L} <= r  <=>  sum_e l_e(2-2r)  <= L,

with R_e(a) = sup{x : mu_e(x) > a} and l_e(b) = inf{x : mu_e(x) >= b}.
The leg quantity q_e(r) on the left-hand side is called the *planning
duration* of the leg at ceiling r.

Corollary (robust envelope).  If leg e admits a set M_e of memberships
(an expert panel, a hesitant linguistic assessment, a parametric box) and the
admissible choices are independent across legs, the ceiling holds for every
admissible choice iff sum_e max_{m in M_e} q_e^m(r) <= L.

Corollary (crisp routing equivalent).  Replacing every leg by its planning
duration turns the credibilistic ceiling problem into a deterministic
duration-constrained routing problem with K*P interchangeable route slots:
modal durations define the cost, planning durations the route-length limit.

Only the alpha-cut interface of fuzzy_core.FuzzyNumber is used, so the class
works unchanged for triangular, trapezoidal, Gaussian and panel (envelope)
durations.
"""

from __future__ import annotations

from dataclasses import dataclass

from fuzzy_core import FuzzyNumber


@dataclass(frozen=True)
class CredibilisticCeiling:
    """A ceiling r_bar on the overtime credibility of every cell."""

    r_bar: float
    tol: float = 1e-9

    def __post_init__(self):
        if not 0.0 <= self.r_bar < 1.0:
            raise ValueError("the ceiling must satisfy 0 <= r_bar < 1")

    # --- which alpha-cut endpoint the ceiling reads -----------------------
    @property
    def side(self) -> str:
        return "right" if self.r_bar < 0.5 else "left"

    @property
    def level(self) -> float:
        return 2.0 * self.r_bar if self.r_bar < 0.5 else 2.0 - 2.0 * self.r_bar

    # --- planning durations -------------------------------------------------
    def planning_duration(self, leg: FuzzyNumber) -> float:
        """q_e(r_bar): the leg's alpha-cut endpoint selected by the ceiling."""
        if self.side == "right":
            return leg.right_endpoint(self.level)
        return leg.left_endpoint(self.level)

    def envelope_duration(self, admissible: list[FuzzyNumber]) -> float:
        """max over admissible memberships of the planning duration."""
        return max(self.planning_duration(m) for m in admissible)

    # --- checks ---------------------------------------------------------
    def holds(self, legs: list[FuzzyNumber], limit: float) -> bool:
        """Crisp check of Cr{sum legs > limit} <= r_bar (Theorem)."""
        return sum(self.planning_duration(e) for e in legs) <= limit + self.tol

    def holds_robust(self, admissible_sets: list[list[FuzzyNumber]],
                     limit: float) -> bool:
        """Crisp check of the ceiling for every admissible leg-wise choice."""
        return sum(self.envelope_duration(s) for s in admissible_sets) <= limit + self.tol


class PlanningDurationMatrix:
    """Planning durations of every arc and service leg of an instance.

    For a canonical instance (one membership per leg) this is the nominal
    crisp equivalent; for a panel instance (expert_panel.PanelFN legs) the
    alpha-cut endpoints are already envelopes, so the same class returns the
    robust crisp equivalent.
    """

    def __init__(self, instance, ceiling: CredibilisticCeiling):
        self.inst = instance
        self.ceiling = ceiling
        n = instance.n
        q = ceiling.planning_duration
        self.travel = [[0.0 if i == j else q(instance.travel(i, j))
                        for j in range(n + 1)] for i in range(n + 1)]
        self.service = [0.0] + [q(instance.service(i)) for i in range(1, n + 1)]

    def route_duration(self, seq) -> float:
        """Planning duration of one cell visiting `seq` from and to the depot."""
        if not seq:
            return 0.0
        total = self.travel[0][seq[0]]
        for idx, t in enumerate(seq):
            nxt = seq[idx + 1] if idx + 1 < len(seq) else 0
            total += self.service[t] + self.travel[t][nxt]
        return total

    def plan_satisfies(self, cells, limit=None) -> bool:
        limit = self.inst.L if limit is None else limit
        return all(self.route_duration(c) <= limit + self.ceiling.tol
                   for row in cells for c in row)
