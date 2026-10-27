"""Exhaustive endpoint-information experiment on directed seven-stop restrictions.

Only the modal travel/service inputs and support bounds are read from the
released Amazon scenario. Interior endpoints and evaluation completions are
explicit deterministic stress scenarios, not elicited or fitted observations.
No route dominance is used: every ordered route and every unlabelled partition
into at most three routes is retained before common feasibility filtering.
"""
from __future__ import annotations
import itertools
import numpy as np
from exact_fff import partitions
from external_instances_v4 import ExternalMatrixInstance

N, SLOTS, THETA, CAP, TOL = 7, 3, 1.2, .30, 1e-9
LEVELS = np.array([0., .05, .15, .30, .40, .50])
GRIDS = {'two': np.array([0, 2, 3, 5]), 'four': np.arange(6)}
PROFILES = {'arithmetic': np.array([.5, 1/3, 1/6]),
            'top_heavy': np.array([.7, .2, .1])}
ALPHAS = (.1, .3, .7)
TEST_POWERS = (.45, 1.0, 2.2)


def choose(values, workload, maximum, eligible):
    """Deterministic objective, workload, max-risk, enumeration-index tie break."""
    ids = np.flatnonzero(eligible)
    if not len(ids):
        return None
    return int(ids[np.lexsort((ids, maximum[ids], workload[ids], values[ids]))[0]])


class DecisionCase:
    def __init__(self, record):
        inst = ExternalMatrixInstance(record)
        self.id = record['id'] + '-7'
        legs, keys, left_supports = [], [], []
        # Arc-specific shape exponents make modal and endpoint route ordering
        # differ; every case uses the same released deterministic formula.
        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    f = inst.travel(i, j)
                    legs.append((f.b, f.c, (.5, 1., 2., 3.)[(3*i+5*j) % 4]))
                    left_supports.append(f.a)
                    keys.append(('t', i, j))
        for i in range(1, N + 1):
            f = inst.service(i)
            legs.append((f.b, f.c, (.5, 1., 2., 3.)[(3*i+1) % 4]))
            left_supports.append(f.a)
            keys.append(('s', i))
        self.legs = np.array(legs, float)
        self.left_supports = left_supports
        self.keys = keys
        self.key_to_id = {k: i + 1 for i, k in enumerate(keys)}
        self.endpoints = np.zeros((len(legs) + 1, len(LEVELS)))
        self.endpoints[1:] = (self.legs[:, 0, None]
            + (self.legs[:, 1] - self.legs[:, 0])[:, None]
            * (1 - 2*LEVELS[None, :]) ** self.legs[:, 2, None])
        self.sequences = [()]
        incidence = [np.zeros(2*N+1, dtype=np.int16)]
        by_mask = {}
        for k in range(1, N + 1):
            for seq in itertools.permutations(range(1, N + 1), k):
                ids, prev = [], 0
                for node in seq:
                    ids.extend([self.key_to_id[('t', prev, node)], self.key_to_id[('s', node)]])
                    prev = node
                ids.append(self.key_to_id[('t', prev, 0)])
                row = np.zeros(2*N+1, dtype=np.int16)
                row[:len(ids)] = ids
                rid = len(self.sequences)
                self.sequences.append(seq)
                incidence.append(row)
                by_mask.setdefault(sum(1 << (i-1) for i in seq), []).append(rid)
        self.incidence = np.array(incidence)
        self.Q = self.endpoints[self.incidence].sum(axis=1)
        plans = []
        for groups in partitions((1 << N)-1, SLOTS):
            for ids in itertools.product(*(by_mask[g] for g in groups)):
                plans.append(ids + (0,)*(SLOTS-len(ids)))
        all_plans = np.array(plans, dtype=np.int32)
        self.total_plans = len(all_plans)
        maxima = self.Q[all_plans, -1].max(axis=1)
        self.bottleneck = float(maxima.min())
        self.L = THETA * self.bottleneck
        self.plans = all_plans[maxima <= self.L + TOL]
        self.workload = self.Q[self.plans, -1].sum(axis=1)
        self.scale = float(self.workload.min())
        self.modal_ok = self.Q[:, -1] <= self.L + TOL
        width = self.Q[:, 0] - self.Q[:, -1]
        self.tri_risk = np.divide(.5*(self.Q[:, 0]-self.L), width,
                                  out=np.zeros(len(width)), where=width > 0)
        self.tri_risk = np.maximum(0., np.minimum(.5, self.tri_risk))
        self.tri_risk[0] = 0.

    def route_bounds(self, grid):
        ix = GRIDS[grid]
        levels, Q = LEVELS[ix], self.Q[:, ix]
        satisfied = Q <= self.L + TOL
        upper = levels[np.argmax(satisfied, axis=1)].copy()
        upper[~satisfied.any(axis=1)] = 1.
        lower = np.max(np.where(~satisfied, levels[None, :], 0.), axis=1)
        upper[0] = lower[0] = 0.
        return lower, upper

    def hidden_risk(self, template):
        """Invert the exact sup--min sum of leg endpoint functions by bisection.

        Between adjacent full-grid endpoints interpolate q_e(rho) with an
        exponent depending on leg and template. All three completions match
        all four interior endpoints, but differ inside the unelicited gaps.
        They are constructed only for evaluation, never passed to choose().
        """
        ids = self.incidence[self.modal_ok]
        lo, hi = np.zeros(len(ids)), np.full(len(ids), .5)
        legpower = np.r_[1., TEST_POWERS[template] *
                        (1 + .12*(np.arange(1, len(self.legs)+1) % 3 - 1))]
        for _ in range(48):
            mid = (lo + hi)/2
            h = np.clip(np.searchsorted(LEVELS, mid, side='right')-1, 0, len(LEVELS)-2)
            z = (mid - LEVELS[h])/(LEVELS[h+1]-LEVELS[h])
            qa = self.endpoints[ids, h[:, None]]
            qb = self.endpoints[ids, (h+1)[:, None]]
            Q = (qa - (qa-qb)*z[:, None]**legpower[ids]).sum(axis=1)
            above = Q > self.L
            lo[above], hi[~above] = mid[above], mid[~above]
        risk = np.full(len(self.sequences), np.nan)
        risk[self.modal_ok] = (lo+hi)/2
        risk[self.Q[:, 0] <= self.L] = 0.
        risk[0] = 0.
        return risk

    def plan_owa(self, route_risks, weights):
        return np.sort(route_risks[self.plans], axis=1)[:, ::-1] @ weights

    def sweep_candidates(self, upper, upper_owa, eligible):
        maximum = upper[self.plans].max(axis=1)
        result = []
        for ceiling in sorted(set(maximum[eligible])):
            feasible = eligible & (maximum <= ceiling + TOL)
            ids = np.flatnonzero(feasible)
            # Exact workload optimum, then OWA, then enumeration index.
            selected = int(ids[np.lexsort((ids, upper_owa[ids], self.workload[ids]))[0]])
            result.append(selected)
        return np.array(sorted(set(result)), dtype=int)

    def run(self):
        hidden = [self.hidden_risk(t) for t in range(len(TEST_POWERS))]
        records = []
        for grid in GRIDS:
            lower, upper = self.route_bounds(grid)
            maximum = upper[self.plans].max(axis=1)
            for risk in hidden:
                assert np.all(risk[self.modal_ok] <= upper[self.modal_ok] + 1e-8)
                assert np.all(risk[self.modal_ok] >= lower[self.modal_ok] - 1e-8)
            for pname, weights in PROFILES.items():
                upper_owa = self.plan_owa(upper, weights)
                lower_owa = self.plan_owa(lower, weights)
                tri_owa = self.plan_owa(self.tri_risk, weights)
                tests = [self.plan_owa(r, weights) for r in hidden]
                for policy in ('uncapped', 'capped'):
                    eligible = np.ones(len(self.plans), dtype=bool) if policy == 'uncapped' else maximum <= CAP + TOL
                    sweep = self.sweep_candidates(upper, upper_owa, eligible) if eligible.any() else []
                    for alpha in ALPHAS:
                        common = alpha*self.workload/self.scale
                        upper_obj = common + (1-alpha)*upper_owa
                        tri_obj = common + (1-alpha)*tri_owa
                        optimum = choose(upper_obj, self.workload, maximum, eligible)
                        selections = {'minimax': optimum,
                            'triangular': choose(tri_obj, self.workload, maximum, eligible)}
                        sweep_ok = np.zeros(len(self.plans), dtype=bool)
                        sweep_ok[sweep] = True
                        selections['sweep'] = choose(upper_obj, self.workload, maximum, sweep_ok)
                        for method, index in selections.items():
                            rec = dict(grid=grid, profile=pname, weights=weights.tolist(), policy=policy,
                                alpha=alpha, method=method, eligible_plans=int(eligible.sum()),
                                status='Optimal' if index is not None else 'Infeasible')
                            if index is not None:
                                routes = self.plans[index]
                                rec.update(plan_index=index, routes=[list(self.sequences[i]) for i in routes],
                                    workload=float(self.workload[index]), upper_owa=float(upper_owa[index]),
                                    lower_owa=float(lower_owa[index]), upper_max=float(maximum[index]),
                                    upper_J=float(upper_obj[index]), minimax_J=float(upper_obj[optimum]),
                                    upper_gap=float(upper_obj[index]-upper_obj[optimum]),
                                    upper_gap_pct=float(100*(upper_obj[index]/upper_obj[optimum]-1)), tests=[])
                                for t, (risk, owa) in enumerate(zip(hidden, tests)):
                                    obj = common + (1-alpha)*owa
                                    oracle = choose(obj, self.workload, risk[self.plans].max(axis=1), eligible)
                                    uncap = float(obj.min())
                                    regret = float(obj[index]-obj[oracle])
                                    if method == 'minimax':
                                        assert regret <= (1-alpha)*float(np.diff(LEVELS[GRIDS[grid]]).max()) + 1e-8
                                    rec['tests'].append(dict(completion=t, J=float(obj[index]), oracle_J=float(obj[oracle]),
                                        oracle_index=oracle, unconstrained_oracle_J=uncap,
                                        regret=regret, regret_pct=float(100*(obj[index]/obj[oracle]-1)),
                                        loss_vs_unconstrained=float(obj[index]-uncap),
                                        worst_risk=float(risk[routes].max()),
                                        fff=float(owa[index])))
                            records.append(rec)
        return dict(id=self.id, n=N, slots=SLOTS, L=self.L, modal_bottleneck=self.bottleneck,
            scale=self.scale, total_routes=len(self.sequences)-1, total_plans=self.total_plans,
            modal_feasible_plans=len(self.plans), legs=[dict(key=list(k), a=float(self.left_supports[i]), b=float(x[0]), c=float(x[1]),
                gamma=float(x[2]), endpoints=self.endpoints[i+1].tolist())
                for i, (k, x) in enumerate(zip(self.keys, self.legs))], records=records)
