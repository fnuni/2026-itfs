"""Independent checks of the endpoint minimax results.

1. Sharpness of the endpoint certificates: random legs and grids; the
   credibility of a sum of staircase memberships is computed by level-set
   addition of step functions (no crisp-equivalent formula) and compared with
   the endpoint bounds.
2. Count representation of the OWA on random grid-valued vectors.
3. Sharpness of the ceiling-sweep gap on the cluster construction of the
   supplement (exact models).
4. Sharpness of the regret constant on the two-tour construction.
5. Consistency of the representative- and optimistic-feasibility records (bounds, enumeration).
6. Complete-membership admissibility, strong endpoints and direct Pos/Nec
   calculations, including the representative sentinel and endpoint equalities.
"""
import itertools, json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from endpoint_minimax import (EndpointGrid, StaircaseCompletion, LowerStaircaseCompletion, SupMinSum,
                              OWAProfile, CountRepresentation, EndpointColumns, CountMILP, GridSweep)


class Check:
    def __init__(self):
        self.out = {}

    def record(self, name, ok, **info):
        self.out[name] = dict(passed=bool(ok), **info)
        print(name, 'passed' if ok else 'FAILED', info)
        return ok


def random_legs(rng, grid, m):
    legs = []
    for _ in range(m):
        b = rng.uniform(5, 20)
        inc = np.sort(rng.uniform(0, 10, len(grid.levels) - 1))[::-1]
        legs.append([b + x for x in inc] + [b])
    return legs


def sharpness(chk, rng, trials=3000, eps=1e-4):
    bad = 0
    grids = [(0, .05, .15, .30, .40, .5), (.1, .2, .5), (0, .25, .5), (.05, .5)]
    for lv in grids:
        g = EndpointGrid(lv)
        for _ in range(trials):
            legs = random_legs(rng, g, int(rng.integers(1, 5)))
            Q = np.sum(legs, axis=0)
            L = rng.uniform(Q[-1], Q[0] + 3)          # representative sum <= L: right branch
            up = SupMinSum([StaircaseCompletion(g, q) for q in legs]).right_credibility(L)
            lo = SupMinSum([LowerStaircaseCompletion(g, q, eps) for q in legs]).right_credibility(L)
            U, Lo = min(g.upper(Q, L), .5), g.lower(Q, L)
            ok_up = abs(up - U) <= 1e-12
            # With no lower-completion tail, risk is exactly zero beyond q_1
            # even when the first elicited ceiling is positive.
            violated = any(x > L + 1e-9 for x in Q)
            ok_lo = (Lo < lo <= Lo + eps + 1e-12) if violated else abs(lo) <= 1e-12
            bad += not (ok_up and ok_lo)
    return chk.record('sharp_endpoint_bounds', bad == 0, trials=trials * len(grids), failures=bad)


def direct_pos_nec(legs, L):
    """Independent sup-min event calculation over products of closed plateaux.

    For each product rectangle, the attainable sum is the interval between
    the sums of its endpoints and its membership is at least the minimum
    plateau height. Taking the supremum over all rectangles is exact. This
    does not call cuts, planning endpoints, risk bounds or SupMinSum.
    """
    above = below = 0.0
    for pieces in itertools.product(*(leg.breakpoints() for leg in legs)):
        level = min(v for _, _, v in pieces)
        if sum(hi for _, hi, _ in pieces) > L + 1e-9:
            above = max(above, level)
        if sum(lo for lo, _, _ in pieces) <= L + 1e-9:
            below = max(below, level)
    return above, 1.0 - below


def complete_memberships(chk, rng, trials=120, eps=1e-4):
    bad = memberships = events = boundary_events = 0
    grids = [(0, .05, .15, .30, .40, .5), (.1, .2, .5), (0, .25, .5), (.05, .5)]
    for lv in grids:
        grid = EndpointGrid(lv)
        for trial in range(trials):
            m = 1 + trial % 3
            q = random_legs(rng, grid, m)
            if trial % 3 == 0:
                # Tied endpoints and a degenerate right branch are admissible.
                q[0][-2] = q[0][-1]
            if trial % 11 == 0:
                q[0] = [q[0][-1]] * len(lv)
            Q = np.sum(q, axis=0)
            Lmax = float(Q[0] + 10)
            # One finite completion per fixed domain, reused for every cell.
            upp = [StaircaseCompletion(grid, x, tail=max(x[0], Lmax) + 1) for x in q]
            low = [LowerStaircaseCompletion(grid, x, eps) for x in q]
            for leg in upp + low:
                memberships += 1
                spans = leg.breakpoints()
                points = sorted({x for a, b, _ in spans for x in (a, b)})
                bad += leg.membership(leg.q[-1]) != 1.0
                bad += leg.membership(-1.0) != 0.0
                bad += leg.membership(max(points) + 1) != 0.0
                bad += any(not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b
                                and 0 < v <= 1) for a, b, v in spans)
                # Normality, upper semicontinuity and unimodality checked at
                # all jumps and interval interiors, rather than only on a mesh.
                for i, x in enumerate(points):
                    near = min([abs(y - x) for y in points if y != x] or [1.0]) / 3
                    bad += leg.membership(x) + 1e-12 < max(leg.membership(x - near), leg.membership(x + near))
                right = sorted(set([leg.q[-1]] + points + [(a + b) / 2 for a, b in zip(points, points[1:])]))
                vals = [leg.membership(x) for x in right if x >= leg.q[-1]]
                bad += any(a < b - 1e-12 for a, b in zip(vals, vals[1:]))
                for rho, wanted in zip(grid.levels, leg.q):
                    if rho < .5:
                        got = max(b for a, b, v in spans if v > 2 * rho)
                    else:
                        got = min(a for a, b, v in spans if v == 1)
                    bad += abs(got - wanted) > 1e-10
                    bad += abs(leg.planning_endpoint(rho) - wanted) > 1e-10
                if isinstance(leg, LowerStaircaseCompletion):
                    bad += leg.cut(1.0) != (leg.q[-1], leg.q[-1])
                for alpha in sorted({v for _, _, v in spans}):
                    a, b = leg.cut(alpha)
                    bad += not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b)
            horizons = sorted(set([0.0, float(Q[-1] - .5), Lmax] +
                                  [float(x + shift) for x in Q for shift in (-1e-6, 0, 1e-6)]))
            for L in horizons:
                boundary_events += any(abs(L - x) < 1e-12 for x in Q)
                for legs in (upp, low):
                    pos, nec = direct_pos_nec(legs, L)
                    model = SupMinSum(legs)
                    p, complement = model.possibility_pair(L)
                    bad += abs(p - pos) > 1e-12 or abs(1 - complement - nec) > 1e-12
                    bad += abs(model.credibility(L) - .5 * (pos + nec)) > 1e-12
                    events += 1
                upper_risk, lower_risk = SupMinSum(upp).credibility(L), SupMinSum(low).credibility(L)
                bad += abs(upper_risk - grid.upper(Q, L)) > 1e-12
                if L >= Q[-1] - 1e-9:
                    bad += not (grid.lower(Q, L) - 1e-12 <= lower_risk <= grid.lower(Q, L) + eps + 1e-12)
            # Shared legs, different cells and one fixed L: the same upper
            # completion must certify each possible cell, without refitting.
            for size in range(1, m + 1):
                for selected in itertools.combinations(range(m), size):
                    L = float(Q[-1])
                    sums = np.sum([q[i] for i in selected], axis=0)
                    got = SupMinSum([upp[i] for i in selected]).credibility(L)
                    bad += abs(got - grid.upper(sums, L)) > 1e-12
    # Adversarial regression: replacing {b} by [0,b] destroys q(1/2)=b.
    g = EndpointGrid((0.0, .25, .5))
    lower = LowerStaircaseCompletion(g, (10, 8, 5), .01)
    bad += lower.planning_endpoint(.5) != 5 or SupMinSum([lower]).credibility(3) != 1
    # A bounded upper tail is not a certificate uniformly over all horizons.
    g = EndpointGrid((.1, .3, .5))
    upper = StaircaseCompletion(g, (10, 8, 5), tail=20)
    tail_counterexample = SupMinSum([upper]).credibility(21) == 0 and g.upper(upper.q, 21) == .1
    bad += not tail_counterexample
    return chk.record('complete_membership_admissibility_and_events', bad == 0,
                      memberships=memberships, event_checks=events, endpoint_equalities=boundary_events,
                      failures=bad, lower_core_regression='q(1/2)=5 and Cr(X>3)=1',
                      fixed_horizon_tail_regression='bounded tail fails beyond its chosen domain as expected')


def invalid_inputs(chk):
    cases = [lambda: EndpointGrid([.5]), lambda: EndpointGrid([0, float('nan'), .5]),
             lambda: EndpointGrid([-.1, .5]), lambda: EndpointGrid([.2, .1, .5]),
             lambda: OWAProfile(0), lambda: OWAProfile(2, [1.1, -.1]),
             lambda: OWAProfile(2, [1.0]), lambda: OWAProfile(2, [float('nan'), .5]),
             lambda: OWAProfile(2).owa([.1, .2, .3]),
             lambda: StaircaseCompletion(EndpointGrid([0, .5]), [2, -1]),
             lambda: StaircaseCompletion(EndpointGrid([0, .5]), [2, 1], tail=1),
             lambda: StaircaseCompletion(EndpointGrid([0, .5]), [2, 1, 0])]
    caught = 0
    for f in cases:
        try:
            f()
        except ValueError:
            caught += 1
    return chk.record('invalid_inputs', caught == len(cases), rejected=caught, cases=len(cases))


def secondary_status_provenance(chk):
    """A nonoptimal secondary status must not be relabelled as a timeout."""
    class FakeMILP:
        grid = type('Grid', (), {'values': (.5,)})()
        time_limit = 99.0

        def __init__(self, secondary):
            self.secondary, self.calls = secondary, 0

        def solve(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return dict(status='Optimal', seconds=2., workload=10., fff_upper=.2)
            return dict(status=self.secondary, seconds=3., workload=10., fff_upper=.1)

    statuses = ('Optimal', 'Time limit reached', 'Solve error', 'Infeasible')
    for status in statuses:
        fake = FakeMILP(status)
        result = GridSweep(fake).plans(10.)[.5]
        assert result['secondary_status'] == status and result['secondary_seconds'] == 3.
        assert result['seconds'] == 5. and fake.time_limit == 99.
        assert result['fff_upper'] == (.1 if status == 'Optimal' else .2)
        assert result['tiebreak'] == ('optimal' if status == 'Optimal'
                                     else 'uncertified; first minimum-workload plan kept')
    return chk.record('secondary_solver_status_provenance', True, cases=len(statuses),
                      actual_status_preserved=True, fallback_plan_preserved=True, elapsed_time_checked=True)


def count_identity(chk, rng, trials=5000):
    worst = 0.0
    for _ in range(trials):
        lv = sorted(set([0.0] + list(np.round(rng.uniform(0, .5, 3), 3)) + [.5]))
        g = EndpointGrid(lv)
        S = int(rng.integers(1, 10))
        w = np.sort(rng.dirichlet(np.ones(S)))[::-1]
        prof = OWAProfile(S, w)
        k = int(rng.integers(0, S + 1))
        vals = list(rng.choice(g.values, size=k)) + [0.0] * (S - k)
        rep = CountRepresentation(g, prof)
        worst = max(worst, abs(prof.owa(vals) - rep.value(g.violation_counts(vals))))
        W = prof.W
        assert all(W[i + 1] - W[i] >= W[i + 2] - W[i + 1] - 1e-15 for i in range(S - 1))   # concave
    return chk.record('count_representation', worst <= 1e-12, trials=trials, max_abs_error=worst)


class ClusterConstruction:
    """n_c clusters, one per cell; cluster 1 risk M on both tours, the others a
    cheap tour of upper risk M or a dearer tour of risk 0 (supplement)."""

    def __init__(self, grid, slots, M, eta, L=100.0, base=60.0):
        self.grid, self.slots, self.M, self.eta, self.L = grid, slots, M, eta, L
        h = grid.levels.index(M)
        masks, work, Q = [], [], []
        risky = [L + 5.0 if r < M else L - 5.0 for r in grid.levels[:-1]] + [base]
        safe = [L - 5.0] * (len(grid.levels) - 1) + [base + eta]
        for i in range(slots):
            masks.append(1 << i); work.append(base); Q.append(risky)
            if i > 0:
                masks.append(1 << i); work.append(base + eta); Q.append(safe)
        Q = np.array(Q)
        self.cols = EndpointColumns(np.array(masks), np.array(work), Q,
                                    np.array([grid.upper(q, L) for q in Q]),
                                    np.array([grid.lower(q, L) for q in Q]), slots)
        assert all(abs(u - M) < 1e-12 or u == grid.levels[0] for u in self.cols.upper)


def sweep_gap_sharpness(chk):
    g = EndpointGrid((0.0, .05, .15, .30, .40, .5))
    rows, ok = [], True
    for S in (2, 3, 4, 6, 9):
        prof = OWAProfile(S)
        for M in (.15, .30, .40):
            for alpha in (0.0, .1, .5, .9):
                cons = ClusterConstruction(g, S, M, eta=1e-4)
                milp = CountMILP(cons.cols, S, g, prof)
                scale = milp.solve(1.0, 1.0)['workload']
                opt = milp.solve(alpha, scale)
                sp, sj = GridSweep.best(GridSweep(milp).plans(scale), alpha, scale)
                bound = (1 - alpha) * (1 - prof.w[0]) * opt['max_upper']
                gap = sj - opt['J']
                ratio = sj / opt['J']
                ok &= -1e-12 <= gap <= bound + 1e-12 and gap >= bound - 1e-3
                ok &= ratio <= 1 / prof.w[0] + 1e-9
                rows.append(dict(slots=S, M=M, alpha=alpha, gap=gap, bound=bound, ratio=ratio,
                                 ratio_bound=1 / prof.w[0]))
    a0 = [r for r in rows if r['alpha'] == 0.0]
    ok &= all(abs(r['ratio'] - r['ratio_bound']) <= 1e-3 * r['ratio_bound'] for r in a0)
    return chk.record('sweep_gap_sharp', ok, cases=len(rows),
                      min_gap_over_bound=min(r['gap'] / r['bound'] for r in rows if r['bound'] > 0),
                      alpha0_ratio_vs_bound=[(r['slots'], round(r['ratio'], 4), round(r['ratio_bound'], 4)) for r in a0 if r['M'] == .30])


def regret_sharpness(chk, eps=1e-6, eta=1e-6):
    """One cell, two tours on disjoint legs with the same upper risk rho_h in the
    widest gap; the cheaper tour is chosen and its legs follow mu^+, the other
    tour's legs mu^-_eps (credibilities by level-set addition)."""
    g = EndpointGrid((0.0, .05, .15, .30, .40, .5))
    pts = list(g.levels)
    k = int(np.argmax(np.diff(pts)))
    lo_r, hi_r = pts[k], pts[k + 1]
    delta = hi_r - lo_r
    L = 100.0
    # tour sums: violate lo_r, meet hi_r ; two legs per tour
    tour = [L + 4.0 if r <= lo_r else L - 4.0 for r in g.levels[:-1]] + [60.0]
    legs = [[x / 2 for x in tour]] * 2
    ok = True
    worst = 0.0
    for alpha in (0.0, .1, .5, .9):
        up = SupMinSum([StaircaseCompletion(g, q) for q in legs]).right_credibility(L)
        legs_b = [q[:-1] + [q[-1] + eta / len(legs)] for q in legs]
        down = SupMinSum([LowerStaircaseCompletion(g, q, eps) for q in legs_b]).right_credibility(L)
        scale = 120.0
        # plan A (chosen by J^+: cheaper) realizes up, plan B realizes down
        JA = alpha * 60.0 / scale + (1 - alpha) * up
        JB = alpha * (60.0 + eta) / scale + (1 - alpha) * down
        regret = JA - min(JA, JB)
        worst = max(worst, abs(regret - (1 - alpha) * delta))
        ok &= abs(up - hi_r) < 1e-12 and regret <= (1 - alpha) * delta + 1e-12
    ok &= worst <= 1e-5
    return chk.record('regret_constant_sharp', ok, delta=delta, max_shortfall=worst)


def records(chk):
    ok_all = True
    for name in ('endpoint_minimax.json', 'endpoint_minimax_optimistic.json'):
        path = ROOT / 'results' / name
        if not path.exists():
            ok_all &= chk.record('records:' + name, False, missing=str(path))
            continue
        d = json.loads(path.read_text())
        representative = d['protocol'].get('feasibility') == 'representative'
        n = viol = enum = enum_bad = over = uncertified_cases = uncertified_losses = 0
        for case in d['raw']:
            for rec in case['records']:
                certified = all(s.get('tiebreak') == 'optimal' for s in rec['sweep'].values()
                                if s['status'] == 'Optimal')
                uncertified_cases += not certified
                for a, r in rec['alphas'].items():
                    n += 1
                    viol += not (-1e-9 <= r['sweep_gap'] <= r['bound'] + 1e-9)
                    viol += not (r['fff_lower'] - 1e-12 <= r['fff_upper'])
                    over += r['counts'][-1] > 0
                    uncertified_losses += not certified and r['sweep_gap'] > 1e-7
                    if 'enumeration_J' in r:
                        enum += 1
                        enum_bad += abs(r['enumeration_J'] - r['J']) > 1e-7 or abs(r['enumeration_sweep_J'] - r['sweep_J']) > 1e-7
        ok = viol == 0 and enum_bad == 0 and (over == 0 or not representative)
        ok &= not representative or uncertified_cases == 0
        ok_all &= chk.record('records:' + name, ok, comparisons=n, violations=viol, enumeration_checked=enum,
                             enumeration_mismatch=enum_bad, optima_with_representative_overrun=over,
                             cases_without_certified_sweep_tiebreak=uncertified_cases,
                             losses_with_uncertified_sweep_tiebreak=uncertified_losses)
    return ok_all


def main():
    rng = np.random.default_rng(20261015)
    chk = Check()
    sharpness(chk, rng)
    complete_memberships(chk, rng)
    invalid_inputs(chk)
    secondary_status_provenance(chk)
    count_identity(chk, rng)
    sweep_gap_sharpness(chk)
    regret_sharpness(chk)
    records(chk)
    status = 'passed' if all(v['passed'] for v in chk.out.values()) else 'failed'
    (ROOT / 'results/endpoint_minimax_validation.json').write_text(json.dumps(dict(status=status, checks=chk.out), indent=1) + '\n')
    print(status)
    if status != 'passed':
        sys.exit(1)


if __name__ == '__main__':
    main()
