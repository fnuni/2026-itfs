"""
Reference solvers for MS-VRPPU-FR, all on the shared encoding and evaluator.

Every solver derives from BudgetedSolver and is charged per fuzzy cell
evaluation. Equal allowances do not imply equal computational work or runtime.
None of these is a verbatim reimplementation of a published system: each follows
the algorithmic scheme named in its docstring on our encoding. This is a
controlled implementation comparison, with the limitations in README.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import random

from aggregators import FFF_REPORT, max_risk, gini, jain, cvar
from problem import Solution
from solver import polish, two_opt_cell, EpsilonArchive, _State


class BudgetedSolver(ABC):
    """Common interface: minimise J_alpha under a fixed evaluation budget."""

    name = "abstract"

    def __init__(self, instance, aggregator, alpha=0.3, budget=2500, seed=0,
                 r_bar=None, workload_scale=None, ceiling_incumbent=False):
        self.inst = instance
        self.agg = aggregator
        self.alpha = alpha
        self.budget = budget
        self.rng = random.Random(seed)
        self._eval0 = instance.cell_evals
        self._scale = 1.0
        self.r_bar = r_bar          # when set, solve the constrained problem
        self.penalty = 4.0
        self.workload_scale = workload_scale
        self.ceiling_incumbent = ceiling_incumbent
        self._best_observed = None

    # --- shared machinery -------------------------------------------------
    @property
    def evaluations(self):
        """Fuzzy cell evaluations spent (same unit and budget as FIC-LNS)."""
        return self.inst.cell_evals - self._eval0

    def _state(self, sol):
        st = _State(self.inst, sol)
        self._observe(st)
        return st

    def _observe(self, st):
        """Optional shared ceiling incumbent used only in fresh validation.

        The default preserves the historical implementations exactly. The
        validation option reports the best *evaluated* plan by the same
        feasibility/excess/workload order used by CC-LNS, independently of
        population survival or the scalar selection merit.
        """
        if not self.ceiling_incumbent or self.r_bar is None:
            return
        def rank(s):
            over = [max(0.0, v-self.r_bar) for row in s.risk for v in row]
            exc = max(over) + 0.25*sum(over)/len(over)
            return (s.violation > 1e-9, round(exc, 6), s.makespan)
        if self._best_observed is None or rank(st) < rank(self._best_observed):
            self._best_observed = st.clone()

    def polish(self, sol):
        """Intra-cell 2-opt shared by every solver (memetic offspring)."""
        return polish(self.inst, sol)

    INFEASIBILITY_PENALTY = 10.0

    def _scalar(self, st):
        base = self.INFEASIBILITY_PENALTY * st.violation
        if self.r_bar is None:
            return (self.alpha * (st.makespan / (self.workload_scale or self._scale))
                    + (1 - self.alpha) * self.agg(st.risk) + base)
        over = [max(0.0, v - self.r_bar) for row in st.risk for v in row]
        excess = max(over) + 0.25 * (sum(over) / len(over))
        return st.makespan / (self.workload_scale or self._scale) + self.penalty * excess + base

    def objectives(self, st):
        """Common two-objective vector; infeasibility is charged to both."""
        pen = self.INFEASIBILITY_PENALTY * st.violation
        if self.r_bar is not None:
            over = [max(0.0, v - self.r_bar) for row in st.risk for v in row]
            pen += self.penalty * (max(over) + 0.25 * (sum(over) / len(over)))
        return (st.makespan / (self.workload_scale or self._scale) + pen, self.agg(st.risk) + pen)

    def random_solution(self):
        inst = self.inst
        sol = Solution(inst.K, inst.P)
        tasks = list(range(1, inst.n + 1))
        self.rng.shuffle(tasks)
        cells = [(k, h) for k in range(inst.K) for h in range(inst.P)]
        for i, t in enumerate(tasks):
            k, h = cells[i % len(cells)]
            sol.cells[k][h].append(t)
        return sol

    def greedy_solution(self):
        inst = self.inst
        sol = Solution(inst.K, inst.P)
        unvisited = set(range(1, inst.n + 1))
        per_cell = math.ceil(inst.n / (inst.K * inst.P))
        for k in range(inst.K):
            for h in range(inst.P):
                cur = 0
                for _ in range(per_cell):
                    if not unvisited:
                        break
                    nxt = min(unvisited, key=lambda j: inst.modal_travel(cur, j))
                    sol.cells[k][h].append(nxt)
                    unvisited.discard(nxt)
                    cur = nxt
        for t in list(unvisited):
            sol.cells[self.rng.randrange(inst.K)][self.rng.randrange(inst.P)].append(t)
        return sol

    # --- variation operators shared by the evolutionary baselines ---------
    def mutate(self, sol, strength=3):
        inst = self.inst
        for _ in range(strength):
            if self.rng.random() < 0.5:
                k1, h1 = self.rng.randrange(inst.K), self.rng.randrange(inst.P)
                k2, h2 = self.rng.randrange(inst.K), self.rng.randrange(inst.P)
                c1, c2 = sol.cells[k1][h1], sol.cells[k2][h2]
                if c1 and c2:
                    i, j = self.rng.randrange(len(c1)), self.rng.randrange(len(c2))
                    c1[i], c2[j] = c2[j], c1[i]
            else:
                k1, h1 = self.rng.randrange(inst.K), self.rng.randrange(inst.P)
                c1 = sol.cells[k1][h1]
                if c1:
                    t = c1.pop(self.rng.randrange(len(c1)))
                    c2 = sol.cells[self.rng.randrange(inst.K)][self.rng.randrange(inst.P)]
                    c2.insert(self.rng.randint(0, len(c2)), t)
        return sol

    def crossover(self, a, b):
        """Cell-wise inheritance with repair, valid for any K and P."""
        inst = self.inst
        child = Solution(inst.K, inst.P)
        taken = set()
        for k in range(inst.K):
            for h in range(inst.P):
                src = a if self.rng.random() < 0.5 else b
                seq = [t for t in src.cells[k][h] if t not in taken]
                child.cells[k][h] = seq
                taken.update(seq)
        missing = [t for t in range(1, inst.n + 1) if t not in taken]
        self.rng.shuffle(missing)
        for t in missing:
            c = child.cells[self.rng.randrange(inst.K)][self.rng.randrange(inst.P)]
            c.insert(self.rng.randint(0, len(c)), t)
        return child

    # --- reporting --------------------------------------------------------
    def report(self, best_state, archive):
        self._observe(best_state)
        if self.ceiling_incumbent and self._best_observed is not None:
            best_state = self._best_observed
        R = best_state.R
        out_extra = {}
        if self.r_bar is not None:
            out_extra = dict(r_bar=self.r_bar,
                             ceiling_met=bool(best_state.violation <= 1e-9
                                              and max_risk(R) <= self.r_bar + 1e-12))
        return dict(**out_extra, R=R,
                    solution=[[list(c) for c in row] for row in best_state.sol.cells],
                    makespan=best_state.makespan, fff=FFF_REPORT(R),
                    violation=best_state.violation,
                    feasible=best_state.violation <= 1e-9,
                    max_risk=max_risk(R), gini=gini(R), jain=jain(R),
                    cvar=cvar(R), archive=archive, evaluations=self.evaluations)

    @abstractmethod
    def run(self, archive=None):
        ...


# ---------------------------------------------------------------------------
# Population-based multi-objective baselines
# ---------------------------------------------------------------------------

def _dominates(p, q):
    return all(a <= b for a, b in zip(p, q)) and any(a < b for a, b in zip(p, q))


def fast_nondominated_sort(points):
    fronts, S, n = [[]], [[] for _ in points], [0] * len(points)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            if _dominates(p, q):
                S[i].append(j)
            elif _dominates(q, p):
                n[i] += 1
        if n[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(points, idx):
    d = {i: 0.0 for i in idx}
    if len(idx) <= 2:
        return {i: float("inf") for i in idx}
    for m in range(len(points[0])):
        order = sorted(idx, key=lambda i: points[i][m])
        d[order[0]] = d[order[-1]] = float("inf")
        lo, hi = points[order[0]][m], points[order[-1]][m]
        span = hi - lo
        if span <= 1e-12:
            continue
        for a in range(1, len(order) - 1):
            d[order[a]] += (points[order[a + 1]][m] - points[order[a - 1]][m]) / span
    return d


class NSGA2(BudgetedSolver):
    """Non-dominated sorting genetic algorithm II (Deb et al., 2002)."""

    name = "NSGA-II"

    def __init__(self, *a, pop_size=40, **kw):
        super().__init__(*a, **kw)
        self.pop_size = pop_size

    def run(self, archive=None):
        archive = archive if archive is not None else EpsilonArchive()
        pop = [self._state(self.greedy_solution() if i == 0 else self.random_solution())
               for i in range(self.pop_size)]
        self._scale = max(max(s.makespan for s in pop), 1.0)
        obj = [self.objectives(s) for s in pop]
        for s in pop:
            if s.violation <= 1e-9:
                archive.add((s.makespan, FFF_REPORT(s.risk)))
        while self.evaluations < self.budget:
            fronts = fast_nondominated_sort(obj)
            rank = {i: r for r, f in enumerate(fronts) for i in f}
            cd = {}
            for f in fronts:
                cd.update(crowding_distance(obj, f))

            def tournament():
                i, j = self.rng.randrange(len(pop)), self.rng.randrange(len(pop))
                if rank[i] != rank[j]:
                    return i if rank[i] < rank[j] else j
                return i if cd[i] >= cd[j] else j

            children = []
            for _ in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                c = self.polish(self.mutate(self.crossover(pop[tournament()].sol,
                                               pop[tournament()].sol)))
                st = self._state(c)
                if st.violation <= 1e-9:
                    archive.add((st.makespan, FFF_REPORT(st.risk)))
                children.append(st)
            merged = pop + children
            mobj = obj + [self.objectives(s) for s in children]
            fronts = fast_nondominated_sort(mobj)
            new_idx = []
            for f in fronts:
                if len(new_idx) + len(f) <= self.pop_size:
                    new_idx += f
                else:
                    cdf = crowding_distance(mobj, f)
                    f = sorted(f, key=lambda i: -cdf[i])
                    new_idx += f[: self.pop_size - len(new_idx)]
                    break
            pop = [merged[i] for i in new_idx]
            obj = [mobj[i] for i in new_idx]
        best = min(pop, key=self._scalar)
        return self.report(best, archive)


class MOEAD(BudgetedSolver):
    """MOEA/D with Tchebycheff decomposition (Zhang and Li, 2007)."""

    name = "MOEA/D"

    def __init__(self, *a, pop_size=40, neighbours=8, **kw):
        super().__init__(*a, **kw)
        self.pop_size = pop_size
        self.T = neighbours

    def run(self, archive=None):
        archive = archive if archive is not None else EpsilonArchive()
        N = self.pop_size
        lam = [(i / (N - 1), 1 - i / (N - 1)) for i in range(N)]
        B = [sorted(range(N), key=lambda j: abs(lam[i][0] - lam[j][0]))[: self.T]
             for i in range(N)]
        pop = [self._state(self.random_solution()) for _ in range(N)]
        self._scale = max(max(s.makespan for s in pop), 1.0)
        obj = [self.objectives(s) for s in pop]
        for s in pop:
            if s.violation <= 1e-9:
                archive.add((s.makespan, FFF_REPORT(s.risk)))
        z = [min(o[m] for o in obj) for m in range(2)]

        def tcheby(o, w):
            return max(w[m] * abs(o[m] - z[m]) for m in range(2))

        while self.evaluations < self.budget:
            for i in range(N):
                if self.evaluations >= self.budget:
                    break
                a, b = self.rng.sample(B[i], 2)
                child = self.polish(self.mutate(self.crossover(pop[a].sol, pop[b].sol)))
                st = self._state(child)
                o = self.objectives(st)
                if st.violation <= 1e-9:
                    archive.add((st.makespan, FFF_REPORT(st.risk)))
                z = [min(z[m], o[m]) for m in range(2)]
                for j in B[i]:
                    if tcheby(o, lam[j]) <= tcheby(obj[j], lam[j]):
                        pop[j], obj[j] = st, o
        best = min(pop, key=self._scalar)
        return self.report(best, archive)


class IBEA(BudgetedSolver):
    """Indicator-based EA with the additive binary epsilon indicator
    (Zitzler and Kunzli, 2004), with the standard incremental fitness update."""

    name = "IBEA"

    def __init__(self, *a, pop_size=40, kappa=0.05, **kw):
        super().__init__(*a, **kw)
        self.pop_size = pop_size
        self.kappa = kappa

    def _normalise(self, obj):
        lo = [min(o[m] for o in obj) for m in range(2)]
        hi = [max(o[m] for o in obj) for m in range(2)]
        span = [max(hi[m] - lo[m], 1e-12) for m in range(2)]
        return [tuple((o[m] - lo[m]) / span[m] for m in range(2)) for o in obj]

    @staticmethod
    def _eps(p, q):
        return max(p[m] - q[m] for m in range(len(p)))

    def _build(self, obj):
        nrm = self._normalise(obj)
        n = len(nrm)
        I = [[self._eps(nrm[i], nrm[j]) for j in range(n)] for i in range(n)]
        c = max(abs(I[i][j]) for i in range(n) for j in range(n)) or 1.0
        fit = [-sum(math.exp(-I[j][i] / (c * self.kappa))
                    for j in range(n) if j != i) for i in range(n)]
        return I, c, fit

    def run(self, archive=None):
        archive = archive if archive is not None else EpsilonArchive()
        pop = [self._state(self.greedy_solution() if i == 0 else self.random_solution())
               for i in range(self.pop_size)]
        self._scale = max(max(s.makespan for s in pop), 1.0)
        obj = [self.objectives(s) for s in pop]
        for s in pop:
            if s.violation <= 1e-9:
                archive.add((s.makespan, FFF_REPORT(s.risk)))
        I, c, fit = self._build(obj)

        while self.evaluations < self.budget:
            # environmental selection: drop the worst, update the others
            worst = min(range(len(pop)), key=lambda i: fit[i])
            for i in range(len(pop)):
                if i != worst:
                    fit[i] += math.exp(-I[worst][i] / (c * self.kappa))
            pop.pop(worst); obj.pop(worst); fit.pop(worst)
            for row in I:
                row.pop(worst)
            I.pop(worst)

            def binary_tournament():
                i, j = self.rng.randrange(len(pop)), self.rng.randrange(len(pop))
                return i if fit[i] >= fit[j] else j

            child = self.polish(self.mutate(self.crossover(pop[binary_tournament()].sol,
                                               pop[binary_tournament()].sol)))
            st = self._state(child)
            if st.violation <= 1e-9:
                archive.add((st.makespan, FFF_REPORT(st.risk)))
            pop.append(st); obj.append(self.objectives(st))
            I, c, fit = self._build(obj)      # O(pop^2) once per generation step
        best = min(pop, key=self._scalar)
        return self.report(best, archive)


class FuzzMH(BudgetedSolver):
    """
    Fuzzy-aware hybrid metaheuristic in the spirit of the fuzzy multi-objective
    formulation plus metaheuristic search of Mehlawat et al.: a possibilistic
    fitness built from the credibility profile, greedy fuzzy construction, and
    a first-improvement local search on the scalarised objective.
    """

    name = "FuzzMH"

    def __init__(self, *a, pop_size=24, **kw):
        super().__init__(*a, **kw)
        self.pop_size = pop_size

    def _fuzzy_construct(self):
        """Assign tasks to the cell with the lowest current fuzzy load."""
        inst = self.inst
        sol = Solution(inst.K, inst.P)
        load = {(k, h): 0.0 for k in range(inst.K) for h in range(inst.P)}
        order = sorted(range(1, inst.n + 1),
                       key=lambda t: -inst.service_modal[t])
        for t in order:
            k, h = min(load, key=load.get)
            c = sol.cells[k][h]
            if c:
                pos = min(range(len(c) + 1),
                          key=lambda p: inst.modal_travel(c[p - 1] if p else 0, t))
            else:
                pos = 0
            c.insert(pos, t)
            load[(k, h)] += inst.service_modal[t] + inst.modal_travel(0, t)
        return sol

    def run(self, archive=None):
        archive = archive if archive is not None else EpsilonArchive()
        seeds = [self._fuzzy_construct()] + [self.random_solution()
                                             for _ in range(self.pop_size - 1)]
        pop = [self._state(s) for s in seeds]
        self._scale = max(max(s.makespan for s in pop), 1.0)
        for s in pop:
            if s.violation <= 1e-9:
                archive.add((s.makespan, FFF_REPORT(s.risk)))
        best = min(pop, key=self._scalar)
        best_val = self._scalar(best)
        cur, cur_val = best, best_val
        while self.evaluations < self.budget:
            cand = self._state(self.polish(self.mutate(cur.sol.clone(), strength=2)))
            if cand.violation <= 1e-9:
                archive.add((cand.makespan, FFF_REPORT(cand.risk)))
            val = self._scalar(cand)
            if val < cur_val:
                cur, cur_val = cand, val
                if val < best_val:
                    best, best_val = cand, val
            elif self.rng.random() < 0.02:
                cur, cur_val = cand, val          # restart-style diversification
        return self.report(best, archive)


class AdaptiveLNS(BudgetedSolver):
    """
    Learning-augmented LNS without preference conditioning: destroy-operator
    selection is a sliding-window bandit over random / worst-cell / boundary
    removal, repair is greedy insertion.  This is the role Neural-LNS plays in
    the comparison -- a learnt operator-selection policy on top of LNS -- and is
    labelled LNS-AO in the tables to avoid implying a verbatim reimplementation.
    """

    name = "LNS-AO"

    def run(self, archive=None):
        archive = archive if archive is not None else EpsilonArchive()
        inst = self.inst
        st = self._state(self.greedy_solution())
        self._scale = max(st.makespan, 1.0)
        cur_val = self._scalar(st)
        best, best_val = st.clone(), cur_val
        if st.violation <= 1e-9:
            archive.add((st.makespan, FFF_REPORT(st.risk)))
        w = [1.0, 1.0, 1.0]
        score = [0.0, 0.0, 0.0]
        cnt = [1e-9, 1e-9, 1e-9]
        temp = 0.04
        n_iter = 0
        while self.evaluations < self.budget:
            q = max(2, int(self.rng.uniform(0.10, 0.25) * inst.n))
            u = self.rng.random() * sum(w)
            op, acc = 0, 0.0
            for i, wi in enumerate(w):
                acc += wi
                if u <= acc:
                    op = i
                    break
            cand = st.clone()
            if op == 0:
                pool = [t for row in cand.sol.cells for c in row for t in c]
                self.rng.shuffle(pool)
                removed = pool[:q]
            elif op == 1:
                order = sorted(((cand.risk[k][h], k, h) for k in range(inst.K)
                                for h in range(inst.P)), reverse=True)
                removed = []
                for _, k, h in order:
                    cc = list(cand.sol.cells[k][h])
                    self.rng.shuffle(cc)
                    removed += cc
                    if len(removed) >= q:
                        break
                removed = removed[:q]
            else:
                pool = []
                for k in range(inst.K):
                    for h in range(inst.P):
                        c = cand.sol.cells[k][h]
                        pool += c[:2] + c[-2:]
                self.rng.shuffle(pool)
                removed = list(dict.fromkeys(pool))[:q]
            if not removed:
                continue
            touched = [(k, h) for k in range(inst.K) for h in range(inst.P)
                       if any(t in removed for t in cand.sol.cells[k][h])]
            cand.sol.remove(removed)
            for k, h in touched:
                cand.refresh(k, h)
            touched = list(touched)
            for t in removed:                      # greedy modal-cost insertion
                bk, bh, bp, bc = 0, 0, 0, float("inf")
                for k in range(inst.K):
                    for h in range(inst.P):
                        c = cand.sol.cells[k][h]
                        for pos in range(len(c) + 1):
                            prev = c[pos - 1] if pos else 0
                            nxt = c[pos] if pos < len(c) else 0
                            delta = (inst.modal_travel(prev, t)
                                     + inst.modal_travel(t, nxt)
                                     - inst.modal_travel(prev, nxt))
                            if delta < bc:
                                bk, bh, bp, bc = k, h, pos, delta
                cand.sol.cells[bk][bh].insert(bp, t)
                touched.append((bk, bh))
            for k, h in set(touched):
                two_opt_cell(inst, cand.sol.cells[k][h])
                cand.refresh(k, h)
            n_iter += 1
            cval = self._scalar(cand)
            self._observe(cand)
            if cand.violation <= 1e-9:
                archive.add((cand.makespan, FFF_REPORT(cand.risk)))
            cnt[op] += 1
            if cval < cur_val or self.rng.random() < math.exp(
                    -(cval - cur_val) / max(temp, 1e-9)):
                st, cur_val = cand, cval
                score[op] += 1.0 if cval < best_val else 0.3
            if cval < best_val:
                best, best_val = cand.clone(), cval
            temp *= 0.995
            if n_iter % 40 == 0:
                w = [max(0.05, 0.7 * a + 0.3 * (s / c))
                     for a, s, c in zip(w, score, cnt)]
                score = [0.0, 0.0, 0.0]
                cnt = [1e-9, 1e-9, 1e-9]
        return self.report(best, archive)


SOLVER_REGISTRY = {
    "NSGA-II": NSGA2,
    "MOEA/D": MOEAD,
    "IBEA": IBEA,
    "FuzzMH": FuzzMH,
    "LNS-AO": AdaptiveLNS,
}
