"""Certified interpolation and neutral admissible completions (release 12).

Theorem 13 certifies the worst-case objective J+ of every plan on the fixed
finite domain, not only of the minimax plan. CI-tau uses this certificate as a
constraint: it optimizes the piecewise-linear point completion among plans
whose J+ is within a factor (1+tau) of the minimax value. Archived v11
decisions are reused and asserted; new completions are evaluation-only.
"""
from __future__ import annotations

import numpy as np
from endpoint_decision_v11 import GRIDS, LEVELS, PROFILES, ALPHAS, CAP, TOL, choose
from endpoint_information_v11 import piecewise_linear_risks, completion_risks

METHODS = ('triangular', 'minimax', 'sweep', 'piecewise_linear')


def gap_exponent_risks(case, exponents):
    """Invert the sup--min sum for leg- and interval-specific exponents.

    exponents has shape (number of legs + 1, len(LEVELS) - 1); row zero is the
    padding leg. Each interval keeps both full-grid endpoints, so the
    completion is admissible for every grid that is a subset of LEVELS.
    """
    ids = case.incidence[case.modal_ok]
    lo, hi = np.zeros(len(ids)), np.full(len(ids), .5)
    for _ in range(48):
        mid = (lo+hi)/2
        h = np.clip(np.searchsorted(LEVELS, mid, side='right')-1, 0, len(LEVELS)-2)
        z = (mid-LEVELS[h])/(LEVELS[h+1]-LEVELS[h])
        qa = case.endpoints[ids, h[:, None]]
        qb = case.endpoints[ids, (h+1)[:, None]]
        q = (qa-(qa-qb)*z[:, None]**exponents[ids, h[:, None]]).sum(axis=1)
        above = q > case.L
        lo[above], hi[~above] = mid[above], mid[~above]
    risk = np.full(len(case.sequences), np.nan)
    risk[case.modal_ok] = (lo+hi)/2
    risk[case.Q[:, 0] <= case.L] = 0.
    risk[0] = 0.
    return risk


def neutral_exponents(case, rng, draws, low, high):
    shape = (draws, len(case.legs)+1, len(LEVELS)-1)
    return np.exp(rng.uniform(np.log(low), np.log(high), size=shape))


def evaluate_case(case, original, information, frozen, case_index):
    assert case.id == original['id'] == information['id']
    assert case.total_plans == original['total_plans']
    rng = np.random.default_rng(frozen['neutral_seed']+case_index)
    expo = neutral_exponents(case, rng, frozen['neutral_draws'], *frozen['neutral_exponent_range'])
    families = {
        'smooth': [completion_risks(case, p) for p in (.45, 1., 2.2)],
        'stress': [completion_risks(case, p) for p in (.1, 10.)],
        'neutral': [gap_exponent_risks(case, expo[d]) for d in range(len(expo))],
    }
    taus = frozen['taus']
    rows = []
    for grid in GRIDS:
        lower, upper = case.route_bounds(grid)
        for risks in families['neutral']:
            assert np.all(risks[case.modal_ok] <= upper[case.modal_ok]+1e-8)
            assert np.all(risks[case.modal_ok] >= lower[case.modal_ok]-1e-8)
        linear = piecewise_linear_risks(case, grid)
        maximum = upper[case.plans].max(axis=1)
        for profile, weights in PROFILES.items():
            upper_owa = case.plan_owa(upper, weights)
            linear_owa = case.plan_owa(linear, weights)
            fam_owa = {f: [case.plan_owa(r, weights) for r in rs] for f, rs in families.items()}
            fam_max = {f: [r[case.plans].max(axis=1) for r in rs] for f, rs in families.items()}
            for policy in ('uncapped', 'capped'):
                eligible = np.ones(len(case.plans), bool) if policy == 'uncapped' else maximum <= CAP+TOL
                for alpha in ALPHAS:
                    common = alpha*case.workload/case.scale
                    upper_obj = common+(1-alpha)*upper_owa
                    pl_obj = common+(1-alpha)*linear_owa
                    opt = float(upper_obj[eligible].min())
                    archived = {r['method']: r for r in information['records']
                                if r['grid'] == grid and r['profile'] == profile
                                and r['policy'] == policy and r['alpha'] == alpha}
                    assert set(archived) == set(METHODS)
                    selected = {m: archived[m]['plan_index'] for m in METHODS}
                    assert selected['piecewise_linear'] == choose(pl_obj, case.workload, maximum, eligible)
                    assert abs(archived['minimax']['upper_J']-opt) < 1e-8
                    for tau in taus:
                        ok = eligible & (upper_obj <= (1+tau)*opt+TOL)
                        selected[f'ci_{tau:g}'] = choose(pl_obj, case.workload, maximum, ok)
                    for method, index in selected.items():
                        if method in archived:
                            assert abs(archived[method]['upper_J']-float(upper_obj[index])) < 1e-8
                        row = dict(grid=grid, profile=profile, policy=policy, alpha=alpha, method=method,
                                   plan_index=int(index), workload=float(case.workload[index]),
                                   upper_J=float(upper_obj[index]), minimax_J=opt,
                                   robust_gap_pct=100*(float(upper_obj[index])/opt-1),
                                   upper_max=float(maximum[index]))
                        for fam in families:
                            reg = []
                            for owa, mx in zip(fam_owa[fam], fam_max[fam]):
                                obj = common+(1-alpha)*owa
                                oracle = choose(obj, case.workload, mx, eligible)
                                reg.append(100*(float(obj[index])/float(obj[oracle])-1))
                            row[fam+'_regret_pct'] = reg
                        if method in archived and method != 'piecewise_linear':
                            for t, value in enumerate(row['smooth_regret_pct']):
                                assert abs(value-archived[method]['tests'][t]['regret_pct']) < 1e-6
                        rows.append(row)
    return dict(id=case.id, L=case.L, modal_feasible_plans=len(case.plans), records=rows)
