"""Same-information interpolation and fixed additional completion stresses.

The original 576 decisions and their smooth-test values remain the reference.
This module adds one point-completion method and evaluates all four methods
on two fixed stress completions. Upper-objective values also give the common
least-favourable-completion certificate; that identity is not a new test.
"""
from __future__ import annotations

import copy
import numpy as np
from endpoint_decision_v11 import GRIDS, LEVELS, PROFILES, ALPHAS, CAP, TOL, choose


def piecewise_linear_risks(case, grid):
    """Invert the sum of the piecewise-linear available endpoint functions."""
    indices = GRIDS[grid]
    q, levels = case.Q[:, indices], LEVELS[indices]
    satisfied = q <= case.L + TOL
    h = np.argmax(satisfied, axis=1)
    lower = np.maximum(0, h-1)
    rows = np.arange(len(h))
    width = q[rows, lower]-q[rows, h]
    fraction = np.divide(q[rows, lower]-case.L, width,
                         out=np.zeros_like(width), where=width > 0)
    risks = levels[lower]+(levels[h]-levels[lower])*fraction
    risks[h == 0] = 0.
    risks[~satisfied.any(axis=1)] = np.nan
    risks[0] = 0.
    return risks


def completion_risks(case, power):
    """Full-grid-compatible completion with the frozen leg-specific exponent."""
    ids = case.incidence[case.modal_ok]
    lo, hi = np.zeros(len(ids)), np.full(len(ids), .5)
    exponent = np.r_[1., power*(1+.12*(np.arange(1, len(case.legs)+1) % 3-1))]
    for _ in range(48):
        mid = (lo+hi)/2
        h = np.clip(np.searchsorted(LEVELS, mid, side='right')-1, 0, len(LEVELS)-2)
        z = (mid-LEVELS[h])/(LEVELS[h+1]-LEVELS[h])
        qa = case.endpoints[ids, h[:, None]]
        qb = case.endpoints[ids, (h+1)[:, None]]
        q = (qa-(qa-qb)*z[:, None]**exponent[ids]).sum(axis=1)
        above = q > case.L
        lo[above], hi[~above] = mid[above], mid[~above]
    risk = np.full(len(case.sequences), np.nan)
    risk[case.modal_ok] = (lo+hi)/2
    risk[case.Q[:, 0] <= case.L] = 0.
    risk[0] = 0.
    return risk


def extend_case(case, original, frozen):
    """Keep original methods anchored, optimize PL, then evaluate fixed tests."""
    assert case.id == original['id']
    assert case.total_plans == original['total_plans']
    assert len(case.plans) == original['modal_feasible_plans']
    assert abs(case.L-original['L']) < TOL
    smooth = [completion_risks(case, p) for p in frozen['smooth_powers']]
    stress = [completion_risks(case, p) for p in frozen['stress_powers']]
    records = []
    for grid in GRIDS:
        lower, upper = case.route_bounds(grid)
        linear = piecewise_linear_risks(case, grid)
        maximum = upper[case.plans].max(axis=1)
        for risks in smooth+stress+[linear]:
            assert np.all(risks[case.modal_ok] <= upper[case.modal_ok]+1e-8)
            assert np.all(risks[case.modal_ok] >= lower[case.modal_ok]-1e-8)
        for profile, weights in PROFILES.items():
            linear_owa = case.plan_owa(linear, weights)
            upper_owa = case.plan_owa(upper, weights)
            lower_owa = case.plan_owa(lower, weights)
            smooth_owas = [case.plan_owa(r, weights) for r in smooth]
            stress_owas = [case.plan_owa(r, weights) for r in stress]
            for policy in ('uncapped', 'capped'):
                eligible = np.ones(len(case.plans), bool) if policy == 'uncapped' else maximum <= CAP+TOL
                for alpha in ALPHAS:
                    common = alpha*case.workload/case.scale
                    upper_obj = common+(1-alpha)*upper_owa
                    originals = {r['method']: r for r in original['records']
                        if r['grid'] == grid and r['profile'] == profile
                        and r['policy'] == policy and r['alpha'] == alpha}
                    assert set(originals) == {'triangular', 'minimax', 'sweep'}
                    optimum = originals['minimax']['upper_J']
                    assert abs(optimum-float(upper_obj[eligible].min())) < 1e-8
                    index = choose(common+(1-alpha)*linear_owa, case.workload, maximum, eligible)
                    pl = dict(grid=grid, profile=profile, weights=weights.tolist(), policy=policy,
                        alpha=alpha, method='piecewise_linear', eligible_plans=int(eligible.sum()),
                        status='Optimal', plan_index=index,
                        routes=[list(case.sequences[i]) for i in case.plans[index]],
                        workload=float(case.workload[index]), upper_owa=float(upper_owa[index]),
                        lower_owa=float(lower_owa[index]), upper_max=float(maximum[index]),
                        upper_J=float(upper_obj[index]), minimax_J=optimum,
                        upper_gap=float(upper_obj[index]-optimum),
                        upper_gap_pct=100*(float(upper_obj[index])/optimum-1), tests=[])
                    selected = [copy.deepcopy(r) for r in originals.values()]+[pl]
                    for t, (risk, owa) in enumerate(zip(smooth, smooth_owas)):
                        obj = common+(1-alpha)*owa
                        oracle = choose(obj, case.workload, risk[case.plans].max(axis=1), eligible)
                        oracle_value = originals['minimax']['tests'][t]['oracle_J']
                        assert abs(float(obj[oracle])-oracle_value) < 1e-8
                        for original_rec in originals.values():
                            assert abs(float(obj[original_rec['plan_index']])-original_rec['tests'][t]['J']) < 1e-8
                        value = float(obj[index])
                        pl['tests'].append(dict(completion=t, J=value, oracle_J=oracle_value,
                            oracle_index=oracle, unconstrained_oracle_J=float(obj.min()),
                            regret=value-oracle_value, regret_pct=100*(value/oracle_value-1),
                            loss_vs_unconstrained=value-float(obj.min()),
                            worst_risk=float(risk[case.plans[index]].max()), fff=float(owa[index])))
                    for rec in selected:
                        idx = rec['plan_index']
                        assert rec['routes'] == [list(case.sequences[i]) for i in case.plans[idx]]
                        assert abs(rec['workload']-float(case.workload[idx])) < 1e-8
                        assert abs(rec['upper_J']-float(upper_obj[idx])) < 1e-8
                        rec['anchored_original_decision'] = rec['method'] != 'piecewise_linear'
                        rec['stress_tests'] = []
                        rec['common_adversary'] = dict(J=rec['upper_J'], oracle_J=optimum,
                            regret=rec['upper_gap'], regret_pct=rec['upper_gap_pct'],
                            interpretation='Deterministic grid-specific least-favourable-completion certificate; not an additional empirical test.')
                        for t, (power, risk, owa) in enumerate(zip(frozen['stress_powers'], stress, stress_owas)):
                            obj = common+(1-alpha)*owa
                            oracle = choose(obj, case.workload, risk[case.plans].max(axis=1), eligible)
                            value, oracle_value = float(obj[idx]), float(obj[oracle])
                            rec['stress_tests'].append(dict(completion=t, power=power, J=value,
                                oracle_J=oracle_value, oracle_index=oracle,
                                unconstrained_oracle_J=float(obj.min()), regret=value-oracle_value,
                                regret_pct=100*(value/oracle_value-1),
                                loss_vs_unconstrained=value-float(obj.min()),
                                worst_risk=float(risk[case.plans[idx]].max()), fff=float(owa[idx])))
                        records.append(rec)
    return dict({k: copy.deepcopy(v) for k, v in original.items() if k != 'records'}, records=records)
