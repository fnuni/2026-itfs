"""
Possibilistic primitives for MS-VRPPU-FR.

Class hierarchy (abstract base + concrete membership shapes) implementing the
unified credibility distribution of a normal, unimodal fuzzy number:

    Cr{xi <= t} = mu(t)/2                if t <= core_lo
                = 1/2                    if core_lo <= t <= core_hi
                = 1 - mu(t)/2            if t >= core_hi

derived from Cr = (Pos + Nec)/2 with
    Pos{xi <= t} = sup_{x<=t} mu(x),  Nec{xi <= t} = 1 - sup_{x>t} mu(x).

Every shape also exposes the endpoints of its alpha-cuts,

    right_endpoint(alpha) = sup{x : mu(x) > alpha}   (strict cut, alpha in [0,1))
    left_endpoint(beta)   = inf{x : mu(x) >= beta}   (closed cut, beta in (0,1])

which is all the crisp equivalent of a credibility ceiling requires
(crisp_equivalent.CredibilisticCeiling).  Under sup-min addition the
endpoints of a sum are the sums of the endpoints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math


class FuzzyNumber(ABC):
    """Normal, unimodal fuzzy number on the reals."""

    __slots__ = ()

    # --- membership -------------------------------------------------------
    @abstractmethod
    def membership(self, x: float) -> float:
        """Membership degree mu(x) in [0, 1]."""

    @property
    @abstractmethod
    def core(self) -> tuple[float, float]:
        """Interval [core_lo, core_hi] where mu = 1."""

    @property
    @abstractmethod
    def modal(self) -> float:
        """Representative modal (centre-of-core) value."""

    # --- possibilistic measures ------------------------------------------
    def possibility_le(self, t: float) -> float:
        lo, _ = self.core
        return 1.0 if t >= lo else self.membership(t)

    def necessity_le(self, t: float) -> float:
        _, hi = self.core
        return 1.0 - (1.0 if t < hi else self.membership(t))

    def credibility_le(self, t: float) -> float:
        """Cr{xi <= t}; closed form, O(1)."""
        lo, hi = self.core
        if t <= lo:
            return 0.5 * self.membership(t)
        if t >= hi:
            return 1.0 - 0.5 * self.membership(t)
        return 0.5

    def overtime_risk(self, limit: float) -> float:
        """r = Cr{xi > limit} = 1 - Cr{xi <= limit}."""
        return 1.0 - self.credibility_le(limit)

    # --- reference (non closed-form) implementation, for verification -----
    def credibility_le_numeric(self, t: float, grid: int = 200001) -> float:
        lo_s, hi_s = self.numeric_bracket
        xs = [lo_s + (hi_s - lo_s) * i / (grid - 1) for i in range(grid)]
        pos = max((self.membership(x) for x in xs if x <= t), default=0.0)
        sup_gt = max((self.membership(x) for x in xs if x > t), default=0.0)
        return 0.5 * (pos + 1.0 - sup_gt)

    @property
    @abstractmethod
    def support(self) -> tuple[float, float]:
        """True support endpoints; may be infinite."""

    @property
    def numeric_bracket(self) -> tuple[float, float]:
        """Finite interval for numerical checks, distinct from true support."""
        return self.support

    # --- alpha-cuts (used by the crisp equivalent of a ceiling) -------------
    @abstractmethod
    def right_endpoint(self, alpha: float) -> float:
        """sup{x : mu(x) > alpha} for alpha in [0, 1)."""

    @abstractmethod
    def left_endpoint(self, beta: float) -> float:
        """inf{x : mu(x) >= beta} for beta in (0, 1]."""

    @abstractmethod
    def __add__(self, other: "FuzzyNumber") -> "FuzzyNumber":
        """Extension-principle addition (closed within each family)."""


class TriangularFN(FuzzyNumber):
    """TFN (a, b, c): optimistic, modal, pessimistic."""

    __slots__ = ("a", "b", "c")

    def __init__(self, a: float, b: float, c: float):
        if not (a <= b <= c):
            raise ValueError(f"TFN requires a <= b <= c, got ({a}, {b}, {c})")
        self.a, self.b, self.c = float(a), float(b), float(c)

    def membership(self, x: float) -> float:
        if x == self.b:
            return 1.0
        if x < self.a or x > self.c:
            return 0.0
        if x <= self.b:
            return 1.0 if self.b == self.a else (x - self.a) / (self.b - self.a)
        return 1.0 if self.c == self.b else (self.c - x) / (self.c - self.b)

    @property
    def core(self):
        return (self.b, self.b)

    @property
    def modal(self):
        return self.b

    @property
    def support(self):
        return (self.a, self.c)

    def necessity_le(self, t: float) -> float:
        if t >= self.c:
            return 1.0
        return 0.0 if t < self.b else 1.0 - self.membership(t)

    def credibility_le(self, t: float) -> float:
        # Exact Pos/Nec definition handles crisp and discontinuous shoulders.
        return 0.5 * (self.possibility_le(t) + self.necessity_le(t))

    def right_endpoint(self, alpha):
        return self.c - alpha * (self.c - self.b)

    def left_endpoint(self, beta):
        return self.a + beta * (self.b - self.a)

    def __add__(self, other):
        return TriangularFN(self.a + other.a, self.b + other.b, self.c + other.c)

    def __repr__(self):
        return f"TFN({self.a:.2f},{self.b:.2f},{self.c:.2f})"


class TrapezoidalFN(FuzzyNumber):
    """Trapezoidal fuzzy number (a, b1, b2, c)."""

    __slots__ = ("a", "b1", "b2", "c")

    def __init__(self, a, b1, b2, c):
        if not (a <= b1 <= b2 <= c):
            raise ValueError("Trapezoidal requires a <= b1 <= b2 <= c")
        self.a, self.b1, self.b2, self.c = map(float, (a, b1, b2, c))

    def membership(self, x: float) -> float:
        if self.b1 <= x <= self.b2:
            return 1.0
        if x < self.a or x > self.c:
            return 0.0
        if x < self.b1:
            return (x - self.a) / (self.b1 - self.a)
        return (self.c - x) / (self.c - self.b2)

    def necessity_le(self, t: float) -> float:
        if t >= self.c:
            return 1.0
        return 0.0 if t < self.b2 else 1.0 - self.membership(t)

    def credibility_le(self, t: float) -> float:
        return 0.5 * (self.possibility_le(t) + self.necessity_le(t))

    @property
    def core(self):
        return (self.b1, self.b2)

    @property
    def modal(self):
        return 0.5 * (self.b1 + self.b2)

    @property
    def support(self):
        return (self.a, self.c)

    def right_endpoint(self, alpha):
        return self.c - alpha * (self.c - self.b2)

    def left_endpoint(self, beta):
        return self.a + beta * (self.b1 - self.a)

    def __add__(self, other):
        return TrapezoidalFN(self.a + other.a, self.b1 + other.b1,
                             self.b2 + other.b2, self.c + other.c)

    def __repr__(self):
        return f"Trap({self.a:.1f},{self.b1:.1f},{self.b2:.1f},{self.c:.1f})"


class GaussianFN(FuzzyNumber):
    """Gaussian fuzzy number with mode m and spread s: mu(x)=exp(-(x-m)^2/(2 s^2))."""

    __slots__ = ("m", "s")
    _CUT = 3.5  # numerical bracket only, not support truncation

    def __init__(self, m: float, s: float):
        if s <= 0:
            raise ValueError("Gaussian spread must be positive")
        self.m, self.s = float(m), float(s)

    def membership(self, x: float) -> float:
        z = (x - self.m) / self.s
        return math.exp(-0.5 * z * z)

    @property
    def core(self):
        return (self.m, self.m)

    @property
    def modal(self):
        return self.m

    @property
    def support(self):
        return (-math.inf, math.inf)

    @property
    def numeric_bracket(self):
        return (self.m - self._CUT * self.s, self.m + self._CUT * self.s)

    def right_endpoint(self, alpha):
        if alpha <= 0.0:
            return math.inf
        return self.m + self.s * math.sqrt(-2.0 * math.log(min(alpha, 1.0)))

    def left_endpoint(self, beta):
        if beta <= 0.0:
            return -math.inf
        return self.m - self.s * math.sqrt(-2.0 * math.log(min(beta, 1.0)))

    def __add__(self, other):
        # Sup-min extension principle: for Gaussian membership functions the
        # MODES and the SPREADS add linearly,
        #   sup_{x+y=z} min(mu_A(x), mu_B(y)) = exp(-(z-m1-m2)^2 / (2 (s1+s2)^2)).
        # NOTE: the quadrature rule s = hypot(s1, s2) is the *probability*
        # convolution of two Gaussian densities and is NOT the possibilistic
        # sum; using it here would understate the spread of an accumulated
        # shift duration by a factor of up to sqrt(n).
        return GaussianFN(self.m + other.m, self.s + other.s)

    def __repr__(self):
        return f"Gauss({self.m:.1f},{self.s:.2f})"


class IntervalType2FN(FuzzyNumber):
    """
    Interval type-2 TFN with a footprint of uncertainty generated by scaling the
    spread of the primary TFN by (1 -/+ delta). The credibility of the embedded
    type-1 sets is bracketed; the representative value used downstream is the
    midpoint of the endpoint credibility values, a transparent sensitivity
    proxy. This is not a Nie-Tan centroid or a general IT2 type-reduction.
    """

    __slots__ = ("lower", "upper", "b")

    def __init__(self, a: float, b: float, c: float, delta: float = 0.25):
        self.b = float(b)
        self.lower = TriangularFN(b - (b - a) * (1.0 - delta), b,
                                  b + (c - b) * (1.0 - delta))
        self.upper = TriangularFN(b - (b - a) * (1.0 + delta), b,
                                  b + (c - b) * (1.0 + delta))

    def membership(self, x: float) -> float:
        return self.upper.membership(x)

    @property
    def core(self):
        return (self.b, self.b)

    @property
    def modal(self):
        return self.b

    @property
    def support(self):
        return self.upper.support

    def right_endpoint(self, alpha):
        return max(self.lower.right_endpoint(alpha), self.upper.right_endpoint(alpha))

    def left_endpoint(self, beta):
        return max(self.lower.left_endpoint(beta), self.upper.left_endpoint(beta))

    def credibility_interval(self, t: float) -> tuple[float, float]:
        lo = min(self.lower.credibility_le(t), self.upper.credibility_le(t))
        hi = max(self.lower.credibility_le(t), self.upper.credibility_le(t))
        return (lo, hi)

    def credibility_le(self, t: float) -> float:
        lo, hi = self.credibility_interval(t)
        return 0.5 * (lo + hi)

    def __add__(self, other):
        out = IntervalType2FN.__new__(IntervalType2FN)
        out.b = self.b + other.b
        out.lower = self.lower + other.lower
        out.upper = self.upper + other.upper
        return out

    def __repr__(self):
        return f"IT2({self.b:.1f})"
