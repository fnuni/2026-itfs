"""Independent scalar audits plus deterministic exhaustive reproduction.

The audit does not call the optimizer's aggregation, endpoint-sum or hidden
completion evaluators. It also generates the plan domain independently by
permuting all tasks and cutting each permutation into ordered route blocks.
"""
from pathlib import Path
import hashlib, itertools, json, math, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from external_instances_v4 import load_external_records
from endpoint_decision_v11 import DecisionCase


def independent_domains(n, slots):
    out = set()
    for perm in itertools.permutations(range(1, n+1)):
        for k in range(1, slots+1):
            for cuts in itertools.combinations(range(1, n), k-1):
                ends = (0,) + cuts + (n,)
                routes = tuple(sorted(perm[ends[i]:ends[i+1]] for i in range(k)))
                out.add(routes)
    return out


def owa_three(values, weights):
    x = list(values) + [0.]*(3-len(values))
    maximum, minimum = max(x), min(x)
    middle = sum(x) - maximum - minimum
    return weights[0]*maximum + weights[1]*middle + weights[2]*minimum


def audit_case(saved, protocol):
    levels = protocol['grids']['four']
    leg_by_key = {tuple(x['key']): (i+1, x) for i, x in enumerate(saved['legs'])}
    assert all(0 <= x['a'] <= x['b'] <= x['c'] for x in saved['legs'])
    cache = {}

    def audit_route(sequence):
        sequence = tuple(sequence)
        if sequence in cache:
            return cache[sequence]
        if not sequence:
            out = {'modal': 0., 'knots': [0.]*len(levels), 'tests': [0.]*3}
            cache[sequence] = out
            return out
        used, previous = [], 0
        for node in sequence:
            used.extend([leg_by_key[('t', previous, node)], leg_by_key[('s', node)]])
            previous = node
        used.append(leg_by_key[('t', previous, 0)])
        knots = [math.fsum(x['b']+(x['c']-x['b'])*(1-2*r)**x['gamma']
                           for _, x in used) for r in levels]
        tests = []
        for p in protocol['hidden_test_completions']['powers']:
            def summed_endpoint(r):
                h = next((i for i in range(len(levels)-1) if levels[i] <= r < levels[i+1]), len(levels)-2)
                z = (r-levels[h])/(levels[h+1]-levels[h])
                return math.fsum(x['endpoints'][h]-(x['endpoints'][h]-x['endpoints'][h+1])
                    * z**(p*(1+.12*(leg_id%3-1))) for leg_id, x in used)
            if knots[0] <= saved['L']:
                tests.append(0.)
                continue
            lo, hi = 0., .5
            for _ in range(55):
                r = (lo+hi)/2
                if summed_endpoint(r) > saved['L']:
                    lo = r
                else:
                    hi = r
            tests.append((lo+hi)/2)
        out = dict(modal=knots[-1], knots=knots, tests=tests)
        cache[sequence] = out
        return out

    def close(a, b, label):
        assert abs(a-b) <= 2e-8*max(1., abs(a), abs(b)), (saved['id'], label, a, b)

    audits = 0
    for rec in saved['records']:
        if rec['status'] != 'Optimal':
            assert rec['eligible_plans'] == 0
            continue
        routes = rec['routes']
        assert sorted(i for route in routes for i in route) == list(range(1, 8))
        cells = [audit_route(route) for route in routes]
        assert all(c['modal'] <= saved['L']+1e-9 for c in cells)
        wl = math.fsum(c['modal'] for c in cells)
        close(wl, rec['workload'], 'workload')
        upper, lower = [], []
        for c in cells:
            pairs = [(r, c['knots'][levels.index(r)]) for r in protocol['grids'][rec['grid']]]
            upper.append(next(r for r, q in pairs if q <= saved['L']+1e-9))
            lower.append(max([0.] + [r for r, q in pairs if q > saved['L']+1e-9]))
        U, A = owa_three(upper, rec['weights']), owa_three(lower, rec['weights'])
        close(U, rec['upper_owa'], 'upper OWA')
        close(A, rec['lower_owa'], 'lower OWA')
        close(max(upper), rec['upper_max'], 'upper worst cell')
        # Independent layer-cake representation, including empty cells.
        values = sorted(set([0.] + upper))
        count_sum = 0.
        for a, b in zip(values, values[1:]):
            count = sum(u >= b-1e-12 for u in upper)
            count_sum += (b-a)*sum(rec['weights'][:count])
        close(U, count_sum, 'count OWA')
        close(rec['alpha']*wl/saved['scale']+(1-rec['alpha'])*U, rec['upper_J'], 'upper objective')
        if rec['policy'] == 'capped':
            assert max(upper) <= protocol['cap']+1e-9
        for test in rec['tests']:
            t = test['completion']
            risks = [c['tests'][t] for c in cells]
            F = owa_three(risks, rec['weights'])
            close(F, test['fff'], 'test OWA')
            close(max(risks), test['worst_risk'], 'test worst cell')
            close(rec['alpha']*wl/saved['scale']+(1-rec['alpha'])*F, test['J'], 'test objective')
            close(test['J']-test['oracle_J'], test['regret'], 'same-domain regret')
            assert test['regret'] >= -1e-9
            if rec['policy'] == 'capped':
                assert max(risks) <= protocol['cap']+1e-8
            audits += 1
    return audits, len(cache)


def main():
    source = ROOT/'results/endpoint_decision_v11.json'
    payload = json.loads(source.read_text())
    proto = payload['protocol']
    actual = hashlib.sha256((ROOT/'data/amazon_v4/instances.json').read_bytes()).hexdigest()
    assert actual == proto['source_sha256']
    independent = independent_domains(7, 3)
    assert len(independent) == 32760
    records = {r['id']+'-7': r for r in load_external_records()}
    checks = []
    for saved in payload['raw']:
        audits, routes = audit_case(saved, proto)
        rebuilt = DecisionCase(records[saved['id']])
        actual_domain = {tuple(sorted(rebuilt.sequences[i] for i in row if i)) for row in rebuilt.plans}
        route_modal = {seq: rebuilt.Q[i, -1] for i, seq in enumerate(rebuilt.sequences)}
        expected_domain = {plan for plan in independent if all(route_modal[route] <= saved['L']+1e-9 for route in plan)}
        assert actual_domain == expected_domain
        # Full replay compares every optimum, oracle, gap and status, not only
        # the aggregate table; no randomized or deadline-dependent search.
        assert rebuilt.run() == saved, saved['id']
        checks.append(dict(id=saved['id'], independent_test_audits=audits,
                           independently_audited_routes=routes, feasible_plans=len(actual_domain),
                           full_reproduction=True))
        print(saved['id'], 'independent audit and full reproduction passed', flush=True)
    out = dict(status='PASS', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               independent_unfiltered_plans=len(independent), checks=checks,
               limitations='Numerical exhaustive certificates at the documented tolerance; constructed completions do not calibrate operational uncertainty')
    (ROOT/'results/endpoint_decision_v11_validation.json').write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
