"""Independent check of the release-12 expert strata.

Residual counts are recomputed with plain arithmetic from the public CSV; the
'all' stratum must reproduce the published first-session counts and the
published q060 ICC(A,1) of results/reliability20.json; strata must partition
the twenty codes; ICC point values are recomputed with an independent
two-way ANOVA implementation.
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class IndependentICC:
    @staticmethod
    def a1(pairs):
        n, k = len(pairs), 2
        grand = sum(a+b for a, b in pairs)/(n*k)
        rows = [(a+b)/2 for a, b in pairs]
        cols = [sum(p[j] for p in pairs)/n for j in range(k)]
        msr = k*sum((r-grand)**2 for r in rows)/(n-1)
        msc = n*sum((c-grand)**2 for c in cols)/(k-1)
        sse = sum((p[j]-rows[i]-cols[j]+grand)**2 for i, p in enumerate(pairs) for j in range(k))
        mse = sse/((n-1)*(k-1))
        return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)


def main():
    rows = list(csv.DictReader((ROOT/'data/reliability20/responses_numeric.csv').open()))
    strata = {r['code']: r for r in csv.DictReader((ROOT/'data/reliability20/strata.csv').open())}
    saved = json.loads((ROOT/'results/reliability20_strata_v12.json').read_text())['groups']
    published = json.loads((ROOT/'results/reliability20.json').read_text())
    by = {}
    for r in rows:
        by.setdefault(r['code'], {})[r['session']] = r
    assert set(by) == set(strata) and len(by) == 20
    assert {s['area'] for s in strata.values()} <= {'operations', 'research', 'other'}
    assert {s['experience'] for s in strata.values()} <= {'below_2', '2_to_5', 'above_5'}
    groups = {'all': set(by),
              'operations': {c for c, s in strata.items() if s['area'] == 'operations'},
              'research_other': {c for c, s in strata.items() if s['area'] != 'operations'},
              'experience_2plus': {c for c, s in strata.items() if s['experience'] != 'below_2'},
              'experience_below_2': {c for c, s in strata.items() if s['experience'] == 'below_2'}}
    assert groups['operations'] | groups['research_other'] == groups['all']
    assert not groups['operations'] & groups['research_other']
    report = {}
    for g, codes in groups.items():
        assert sorted(codes) == saved[g]['codes']
        for lab, p in (('q030', .30), ('q010', .10)):
            k = (1-p)/.4
            tol = .5*(1+abs(1-k)+k)
            n = exceed = 0
            for c in codes:
                r = by[c]['R1']
                for s in ('s1', 's2', 's3', 's4'):
                    b, q6, qp = num(r[s+'_b']), num(r[s+'_q060']), num(r[s+'_'+lab])
                    if None in (b, q6, qp):
                        continue
                    n += 1
                    exceed += abs((qp-b)-k*(q6-b)) > tol+1e-9
            assert (exceed, n) == (saved[g]['departures'][lab]['exceed'], saved[g]['departures'][lab]['judgments'])
        for s in ('s1', 's2', 's3', 's4'):
            pairs = [(num(by[c]['R1'][s+'_q060']), num(by[c]['R2'][s+'_q060'])) for c in sorted(codes)]
            pairs = [p for p in pairs if None not in p]
            icc = IndependentICC.a1(pairs)
            assert abs(icc-saved[g]['agreement'][s]['icc_A1']) < 1e-10
            if g == 'all':
                assert abs(icc-published['estimates']['all'][s+'_q060']['icc_A1']) < 1e-10
        report[g] = dict(n=len(codes), departures=saved[g]['departures'])
    assert report['all']['departures']['q030'] == {'exceed': 24, 'judgments': 80}
    assert report['all']['departures']['q010'] == {'exceed': 30, 'judgments': 79}
    out = dict(status='PASS', source_sha256=sha(ROOT/'results/reliability20_strata_v12.json'),
               strata_sha256=sha(ROOT/'data/reliability20/strata.csv'),
               validator_sha256=sha(Path(__file__).resolve()), groups=report,
               checks='partition, residual counts, independent ICC(A,1) point values, published all-respondent values')
    (ROOT/'results/reliability20_strata_v12_validation.json').write_text(json.dumps(out, indent=2)+'\n')
    print('strata validation PASS')


if __name__ == '__main__':
    main()
