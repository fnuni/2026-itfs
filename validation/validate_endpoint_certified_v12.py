"""Independent audit and full replay of the release-12 certified-interpolation study.

Route-level quantities (modal duration, grid upper risks, PL risks and the
risks of neutral and smooth completions) are recomputed by scalar bisection on
the released leg endpoints, without calling the evaluated modules. Plans are
enumerated independently by permutations/compositions. Plan-level objectives
are then assembled from these scalar route values to check (i) every CI-tau
selection: the certificate constraint and PL optimality within it, (ii) the
robust gaps of all methods and (iii) all smooth regrets and the neutral
regrets of the first AUDITED_DRAWS draws, including their exact oracles.
Finally the full study is replayed deterministically.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'validation'))
from validate_endpoint_decision_v11 import independent_domains  # noqa: E402

AUDITED_DRAWS = 2
TOL = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScalarRoute:
    """Scalar endpoint sums and inversions for one directed route."""

    def __init__(self, used, levels, L):
        self.used, self.levels, self.L = used, levels, L
        self.knots = [math.fsum(x['endpoints'][h] for _, x in used) for h in range(len(levels))]
        self.modal = self.knots[-1] if used else 0.

    def summed(self, rho, exponent):
        lv = self.levels
        h = next((i for i in range(len(lv)-1) if lv[i] <= rho < lv[i+1]), len(lv)-2)
        z = (rho-lv[h])/(lv[h+1]-lv[h])
        return math.fsum(x['endpoints'][h]-(x['endpoints'][h]-x['endpoints'][h+1])*z**exponent(leg_id, h)
                         for leg_id, x in self.used)

    def invert(self, exponent):
        if not self.used or self.summed(0., exponent) <= self.L:
            return 0.
        lo, hi = 0., .5
        for _ in range(55):
            mid = (lo+hi)/2
            if self.summed(mid, exponent) > self.L:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2

    def grid_bounds(self, grid_levels):
        pairs = [(r, self.knots[self.levels.index(r)]) for r in grid_levels]
        upper = next((r for r, q in pairs if q <= self.L+1e-9), 1.)
        lower = max([0.]+[r for r, q in pairs if q > self.L+1e-9])
        return (0., 0.) if not self.used else (lower, upper)

    def pl_risk(self, grid_levels):
        """Invert the piecewise-linear interpolation of the grid endpoints."""
        if not self.used:
            return 0.
        pts = [(r, self.knots[self.levels.index(r)]) for r in grid_levels]
        if pts[0][1] <= self.L+1e-9:
            return 0.
        for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
            if q1 <= self.L+1e-9:
                return r0+(r1-r0)*((q0-self.L)/(q0-q1) if q0 > q1 else 0.)
        return float('nan')


class CaseAuditor:
    def __init__(self, saved, decision_case, information_case, protocol, levels, grids, profiles, cap, alphas):
        self.saved, self.protocol = saved, protocol
        self.levels, self.grids, self.profiles, self.cap, self.alphas = levels, grids, profiles, cap, alphas
        self.L, self.scale = decision_case['L'], decision_case['scale']
        self.legs = {tuple(x['key']): (i+1, x) for i, x in enumerate(decision_case['legs'])}
        self.information = information_case
        self.max_error = 0.

    def route(self, seq):
        used, prev = [], 0
        for node in seq:
            used.extend([self.legs[('t', prev, node)], self.legs[('s', node)]])
            prev = node
        if seq:
            used.append(self.legs[('t', prev, 0)])
        return ScalarRoute(used, self.levels, self.L)

    def run(self, case_index, rebuilt):
        domain = independent_domains(7, 3)
        seqs = sorted({r for p in domain for r in p})
        routes = {s: self.route(s) for s in seqs}
        plans = [p for p in sorted(domain) if all(routes[r].modal <= self.L+TOL for r in p)]
        assert len(plans) == self.saved['modal_feasible_plans']
        rebuilt_plans = [tuple(sorted(rebuilt.sequences[i] for i in row if i)) for row in rebuilt.plans]
        assert set(rebuilt_plans) == set(plans)
        index_of = {p: i for i, p in enumerate(plans)}
        mapped = [index_of[p] for p in rebuilt_plans]
        workload = np.array([math.fsum(routes[r].modal for r in p) for p in plans])
        rng = np.random.default_rng(self.protocol['neutral_seed']+case_index)
        lo_, hi_ = self.protocol['neutral_exponent_range']
        expo = np.exp(rng.uniform(np.log(lo_), np.log(hi_),
                                  size=(self.protocol['neutral_draws'], len(self.legs)+1, len(self.levels)-1)))
        families = {'smooth': [lambda leg, h, p=p: p*(1+.12*(leg % 3-1)) for p in (.45, 1., 2.2)],
                    'neutral': [lambda leg, h, d=d: float(expo[d, leg, h]) for d in range(AUDITED_DRAWS)]}
        fam_risk = {f: [{s: routes[s].invert(e) for s in seqs if routes[s].modal <= self.L+TOL} for e in es]
                    for f, es in families.items()}
        records = {(r['grid'], r['profile'], r['policy'], r['alpha'], r['method']): r for r in self.saved['records']}
        checks = dict(ci_selections=0, robust_gaps=0, smooth_regrets=0, neutral_regrets=0)
        for grid, glev in self.grids.items():
            bounds = {s: routes[s].grid_bounds(glev) for s in seqs}
            pl = {s: routes[s].pl_risk(glev) for s in seqs if routes[s].modal <= self.L+TOL}
            for s in pl:
                lb, ub = bounds[s]
                assert lb-1e-8 <= pl[s] <= ub+1e-8
            def sorted3(vals):
                x = sorted(list(vals)+[0.]*(3-len(vals)), reverse=True)
                return np.array(x)
            U = np.array([sorted3([bounds[r][1] for r in p]) for p in plans])
            P = np.array([sorted3([pl[r] for r in p]) for p in plans])
            M = U[:, 0]
            F = {f: [np.array([sorted3([rk[r] for r in p]) for p in plans]) for rk in rs] for f, rs in fam_risk.items()}
            Fmax = {f: [np.array([max([rk[r] for r in p]+[0.]) for p in plans]) for rk in rs] for f, rs in fam_risk.items()}
            for prof, w in self.profiles.items():
                w = np.array(w)
                for policy in ('uncapped', 'capped'):
                    elig = np.ones(len(plans), bool) if policy == 'uncapped' else M <= self.cap+TOL
                    for alpha in self.alphas:
                        common = alpha*workload/self.scale
                        Jup = common+(1-alpha)*U@w
                        Jpl = common+(1-alpha)*P@w
                        opt = Jup[elig].min()
                        for tau in self.protocol['taus']:
                            rec = records[(grid, prof, policy, alpha, f'ci_{tau:g}')]
                            k = mapped[rec['plan_index']]
                            assert Jup[k] <= (1+tau)*opt+1e-8
                            ok = elig & (Jup <= (1+tau)*opt+TOL)
                            assert abs(Jpl[k]-Jpl[ok].min()) <= 1e-8, (grid, prof, policy, alpha, tau)
                            checks['ci_selections'] += 1
                        for method in ('triangular', 'minimax', 'sweep', 'piecewise_linear',
                                       *[f'ci_{t:g}' for t in self.protocol['taus']]):
                            rec = records[(grid, prof, policy, alpha, method)]
                            k = mapped[rec['plan_index']]
                            gap = 100*(Jup[k]/opt-1)
                            self.max_error = max(self.max_error, abs(gap-rec['robust_gap_pct']))
                            assert abs(gap-rec['robust_gap_pct']) <= 1e-6
                            checks['robust_gaps'] += 1
                            for fam, key in (('smooth', 'smooth_regret_pct'), ('neutral', 'neutral_regret_pct')):
                                for t, (Ft, Mt) in enumerate(zip(F[fam], Fmax[fam])):
                                    J = common+(1-alpha)*Ft@w
                                    cand = np.flatnonzero(elig)
                                    best = J[cand].min()
                                    reg = 100*(J[k]/best-1)
                                    self.max_error = max(self.max_error, abs(reg-rec[key][t]))
                                    assert abs(reg-rec[key][t]) <= 1e-6, (fam, t, method)
                                    checks[fam+'_regrets'] += 1
        return dict(id=self.saved['id'], independently_enumerated_plans=len(plans),
                    independently_audited_routes=len(seqs), max_abs_error_pct=self.max_error, **checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-replay', action='store_true')
    args = parser.parse_args()
    source = ROOT/'results/endpoint_certified_v12.json'
    frozen_path = ROOT/'results/endpoint_certified_v12_protocol.json'
    payload = json.loads(source.read_text())
    frozen = json.loads(frozen_path.read_text())
    pp = payload['protocol']
    assert pp['frozen_protocol_sha256'] == sha(frozen_path)
    assert pp['original_record_sha256'] == sha(ROOT/'results/endpoint_decision_v11.json')
    assert pp['information_record_sha256'] == sha(ROOT/'results/endpoint_information_v11.json')
    for p, expected in pp['code_sha256'].items():
        assert sha(ROOT/p) == expected, p
    decision = json.loads((ROOT/'results/endpoint_decision_v11.json').read_text())
    information = json.loads((ROOT/'results/endpoint_information_v11.json').read_text())
    dp = decision['protocol']
    levels = dp['grids']['four']
    grids = dp['grids']
    profiles = dp['profiles']
    cases = {c['id']: c for c in decision['raw']}
    infos = {c['id']: c for c in information['raw']}
    from external_instances_v4 import load_external_records
    from endpoint_decision_v11 import DecisionCase
    sources = {r['id']+'-7': r for r in load_external_records()}
    checks = []
    for k, saved in enumerate(payload['raw']):
        auditor = CaseAuditor(saved, cases[saved['id']], infos[saved['id']], frozen, levels, grids,
                              profiles, dp['cap'], dp['alphas'])
        result = auditor.run(k, DecisionCase(sources[saved['id']]))
        checks.append(result)
        print(saved['id'], 'independent scalar audit PASS', flush=True)
    replay = None
    if not args.skip_replay:
        from endpoint_certified_v12 import evaluate_case
        dec, inf = {r['id']: r for r in decision['raw']}, {r['id']: r for r in information['raw']}
        for k, src in enumerate(load_external_records()):
            case = DecisionCase(src)
            assert evaluate_case(case, dec[case.id], inf[case.id], frozen, k) == payload['raw'][k]
        replay = 'full deterministic replay identical for all eight inputs'
        print(replay, flush=True)
    out = dict(status='PASS', source_sha256=sha(source), frozen_protocol_sha256=sha(frozen_path),
               validator_sha256=sha(Path(__file__).resolve()), audited_neutral_draws_per_input=AUDITED_DRAWS,
               checks=checks, replay=replay,
               limitations='Numerical exhaustive validation at tolerance 1e-9; the neutral audit covers the first draws per input, the replay covers all.')
    name = 'endpoint_certified_v12_validation_no_replay.json' if args.skip_replay else 'endpoint_certified_v12_validation.json'
    (ROOT/'results'/name).write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
