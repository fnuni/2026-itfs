"""Independent scalar audit and full deterministic replay of the extension.

The scalar audit reconstructs each used leg and inverts its endpoint sum,
without calling the extension's PL or completion evaluator. Original records
are compared field for field; exhaustive-domain generation is independent.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from external_instances_v4 import load_external_records
from endpoint_decision_v11 import DecisionCase
from endpoint_information_v11 import extend_case
from validate_endpoint_decision_v11 import independent_domains, owa_three


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_case(saved, original, protocol, rebuilt):
    full_grid = protocol['grids']['four']
    grids = protocol['grids']
    frozen = protocol['extension']
    legs = {tuple(x['key']): (i+1, x) for i, x in enumerate(saved['legs'])}
    cache = {}
    counts = dict(original_records_anchored=0, selected_tests_audited=0,
                  oracle_tests_audited=0, linear_selected_plans_audited=0,
                  endpoint_equalities_audited=0)
    max_error = 0.

    def close(a, b, label):
        nonlocal max_error
        max_error = max(max_error, abs(a-b))
        assert abs(a-b) <= 2e-8*max(1., abs(a), abs(b)), (saved['id'], label, a, b)

    def endpoint_sum(used, levels, rho, power):
        h = next((i for i in range(len(levels)-1) if levels[i] <= rho < levels[i+1]), len(levels)-2)
        z = (rho-levels[h])/(levels[h+1]-levels[h])
        a, b = full_grid.index(levels[h]), full_grid.index(levels[h+1])
        return math.fsum(x['endpoints'][a]-(x['endpoints'][a]-x['endpoints'][b])
            * z**(1. if power is None else power*(1+.12*(leg_id % 3-1)))
            for leg_id, x in used)

    def invert(used, levels, power, horizon):
        if not used or endpoint_sum(used, levels, 0., power) <= horizon:
            return 0.
        lo, hi = 0., .5
        for _ in range(55):
            rho = (lo+hi)/2
            if endpoint_sum(used, levels, rho, power) > horizon:
                lo = rho
            else:
                hi = rho
        return (lo+hi)/2

    def route(sequence):
        seq = tuple(sequence)
        if seq in cache:
            return cache[seq]
        used, previous = [], 0
        for node in seq:
            used.extend([legs[('t', previous, node)], legs[('s', node)]])
            previous = node
        if seq:
            used.append(legs[('t', previous, 0)])
        knots = [math.fsum(x['endpoints'][h] for _, x in used) for h in range(len(full_grid))]
        for grid, levels in grids.items():
            for rho in levels:
                close(endpoint_sum(used, levels, rho, None), knots[full_grid.index(rho)], 'PL endpoint')
                counts['endpoint_equalities_audited'] += 1
            if used and all(a > b for a, b in zip(knots, knots[1:])):
                for rho in levels:
                    close(invert(used, levels, None, knots[full_grid.index(rho)]), rho, 'PL inverse at endpoint')
                    counts['endpoint_equalities_audited'] += 1
        for power in frozen['smooth_powers']+frozen['stress_powers']:
            for rho in full_grid:
                close(endpoint_sum(used, full_grid, rho, power), knots[full_grid.index(rho)], 'completion endpoint')
                counts['endpoint_equalities_audited'] += 1
        result = dict(modal=knots[-1], knots=knots,
            linear={g: invert(used, levels, None, saved['L']) for g, levels in grids.items()},
            smooth=[invert(used, full_grid, p, saved['L']) for p in frozen['smooth_powers']],
            stress=[invert(used, full_grid, p, saved['L']) for p in frozen['stress_powers']])
        cache[seq] = result
        return result

    def plan(index):
        return [route(rebuilt.sequences[i]) for i in rebuilt.plans[index]]

    original_index = {(r['grid'],r['profile'],r['policy'],r['alpha'],r['method']):r
                      for r in original['records']}
    for rec in saved['records']:
        key = (rec['grid'],rec['profile'],rec['policy'],rec['alpha'],rec['method'])
        if rec['method'] != 'piecewise_linear':
            anchor = original_index[key]
            assert {k: rec[k] for k in anchor} == anchor
            assert rec['anchored_original_decision']
            counts['original_records_anchored'] += 1
        else:
            counts['linear_selected_plans_audited'] += 1
            assert not rec['anchored_original_decision']
        assert sorted(x for r in rec['routes'] for x in r) == list(range(1,8))
        cells = [route(r) for r in rec['routes']]
        wl = math.fsum(c['modal'] for c in cells)
        close(wl, rec['workload'], 'workload')
        assert all(c['modal'] <= saved['L']+1e-9 for c in cells)
        upper, lower = [], []
        for c in cells:
            pairs = [(rho,c['knots'][full_grid.index(rho)]) for rho in grids[rec['grid']]]
            upper.append(next(rho for rho,q in pairs if q <= saved['L']+1e-9))
            lower.append(max([0.]+[rho for rho,q in pairs if q > saved['L']+1e-9]))
            assert lower[-1]-1e-8 <= c['linear'][rec['grid']] <= upper[-1]+1e-8
        u = owa_three(upper, rec['weights'])
        close(u, rec['upper_owa'], 'upper OWA')
        close(owa_three(lower, rec['weights']), rec['lower_owa'], 'lower OWA')
        close(rec['alpha']*wl/saved['scale']+(1-rec['alpha'])*u, rec['upper_J'], 'upper J')
        assert rec['common_adversary']['J'] == rec['upper_J']
        assert rec['common_adversary']['oracle_J'] == rec['minimax_J']
        assert rec['common_adversary']['regret_pct'] == rec['upper_gap_pct']
        if rec['policy'] == 'capped':
            assert max(upper) <= .3+1e-9
        for field, family in [('tests','smooth'),('stress_tests','stress')]:
            for test in rec[field]:
                t = test['completion']
                if family == 'stress':
                    assert test['power'] == frozen['stress_powers'][t]
                risks = [c[family][t] for c in cells]
                assert all(a-1e-8 <= r <= b+1e-8 for a,r,b in zip(lower,risks,upper))
                f = owa_three(risks, rec['weights'])
                close(f, test['fff'], family+' OWA')
                close(max(risks), test['worst_risk'], family+' maximum')
                close(rec['alpha']*wl/saved['scale']+(1-rec['alpha'])*f, test['J'], family+' J')
                counts['selected_tests_audited'] += 1
                oracle = plan(test['oracle_index'])
                owl = math.fsum(c['modal'] for c in oracle)
                of = owa_three([c[family][t] for c in oracle], rec['weights'])
                close(rec['alpha']*owl/saved['scale']+(1-rec['alpha'])*of, test['oracle_J'], family+' oracle J')
                close(test['J']-test['oracle_J'], test['regret'], 'regret')
                assert test['regret'] >= -1e-8
                counts['oracle_tests_audited'] += 1
    return dict(**counts, independently_audited_routes=len(cache), max_scalar_abs_error=max_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    stem = 'endpoint_information_v11_smoke' if args.smoke else 'endpoint_information_v11'
    source = ROOT/'results'/f'{stem}.json'
    payload = json.loads(source.read_text())
    frozen_path = ROOT/'results/endpoint_information_v11_protocol.json'
    frozen = json.loads(frozen_path.read_text())
    original_path = ROOT/'results/endpoint_decision_v11.json'
    original = json.loads(original_path.read_text())
    assert sha(frozen_path) == payload['protocol']['frozen_protocol_sha256']
    assert sha(original_path) == payload['protocol']['original_record_sha256'] == frozen['original_record_sha256']
    for p, expected in original['protocol']['code_sha256'].items():
        assert sha(ROOT/p) == expected, ('Original code changed',p)
    for p, expected in payload['protocol']['extension_code_sha256'].items():
        assert sha(ROOT/p) == expected
    independent = independent_domains(7,3)
    assert len(independent) == 32760
    sources = {r['id']+'-7':r for r in load_external_records()}
    originals = {r['id']:r for r in original['raw']}
    checks = []
    for saved in payload['raw']:
        rebuilt = DecisionCase(sources[saved['id']])
        actual_domain = {tuple(sorted(rebuilt.sequences[i] for i in row if i)) for row in rebuilt.plans}
        modal = {seq:rebuilt.Q[i,-1] for i,seq in enumerate(rebuilt.sequences)}
        expected_domain = {p for p in independent if all(modal[r] <= saved['L']+1e-9 for r in p)}
        assert actual_domain == expected_domain
        counts = audit_case(saved, originals[saved['id']], payload['protocol'], rebuilt)
        assert extend_case(rebuilt, originals[saved['id']], frozen) == saved
        checks.append(dict(id=saved['id'], full_reproduction=True,
                           feasible_plans=len(actual_domain), **counts))
        print(saved['id'], 'scalar audit, original anchors and full replay PASS', flush=True)
    output = dict(status='PASS', source_sha256=sha(source), frozen_protocol_sha256=sha(frozen_path),
        original_record_sha256=sha(original_path), validator_sha256=sha(Path(__file__).resolve()),
        independent_unfiltered_plans=len(independent), checks=checks,
        limitations='Numerical exhaustive validation, not exact arithmetic or operational calibration; common-adversary values are deterministic certificates.')
    (ROOT/'results'/f'{stem}_validation.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
