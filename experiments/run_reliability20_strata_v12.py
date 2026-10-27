"""Stratify the twenty-expert results by declared area and experience band (release 12).

Strata come from the self-reported area and experience items of the R1 forms,
released as coarse categories in data/reliability20/strata.csv. The analysis is
descriptive: strata have 4-13 respondents and no test of group differences is
claimed.
"""
from __future__ import annotations

import csv, importlib.util, json, statistics as st, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('collector', ROOT/'elicitation/reliability20/raccogli_e_analizza.py')
collector = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(collector)
SCEN = ('s1', 's2', 's3', 's4')
LEVELS = {'q080': .80, 'q030': .30, 'q010': .10}


def num(x):
    return collector.numeric(x)


def tolerance(p):
    k = (1-p)/.4
    return .5*(1+abs(1-k)+k)


class StratifiedPanel:
    def __init__(self):
        rows = list(csv.DictReader((ROOT/'data/reliability20/responses_numeric.csv').open()))
        strata = {r['code']: r for r in csv.DictReader((ROOT/'data/reliability20/strata.csv').open())}
        self.pairs = {}
        for r in rows:
            self.pairs.setdefault(r['code'], {})[r['session']] = r
        assert len(self.pairs) == 20 and set(strata) == set(self.pairs)
        self.strata = strata
        self.groups = {
            'all': sorted(self.pairs),
            'operations': sorted(c for c in self.pairs if strata[c]['area'] == 'operations'),
            'research_other': sorted(c for c in self.pairs if strata[c]['area'] != 'operations'),
            'experience_2plus': sorted(c for c in self.pairs if strata[c]['experience'] != 'below_2'),
            'experience_below_2': sorted(c for c in self.pairs if strata[c]['experience'] == 'below_2'),
        }

    def departures(self, codes):
        out = {}
        for p, lab in ((.30, 'q030'), (.10, 'q010')):
            n = k = 0
            for c in codes:
                r = self.pairs[c]['R1']
                for s in SCEN:
                    b, q6, qp = num(r[s+'_b']), num(r[s+'_q060']), num(r[s+'_'+lab])
                    if None in (b, q6, qp):
                        continue
                    e = (qp-b)-(1-p)/.4*(q6-b)
                    n += 1
                    k += abs(e) > tolerance(p)+1e-9
            out[lab] = dict(exceed=k, judgments=n)
        return out

    def ratios(self, codes):
        res = {}
        for s in SCEN:
            vals = [num(self.pairs[c]['R1'][s+'_q060'])/num(self.pairs[c]['R1'][s+'_b']) for c in codes
                    if num(self.pairs[c]['R1'][s+'_q060']) and num(self.pairs[c]['R1'][s+'_b'])]
            res[s] = st.median(vals)
        return res

    def agreement(self, codes):
        res = {}
        for s in SCEN:
            vals = [(num(self.pairs[c]['R1'][s+'_q060']), num(self.pairs[c]['R2'][s+'_q060'])) for c in codes]
            vals = [v for v in vals if None not in v]
            res[s] = collector.agreement(vals)
        return res

    def run(self):
        out = dict(protocol=dict(
            strata='Self-reported area (operations / research / other relevant experience) and experience band (<2, 2-5, >5 years) from the R1 forms',
            primary_item='q060 test-retest ICC(A,1), paired-expert percentile bootstrap as in the main study',
            departures='R1 straight-branch residuals E_p beyond the whole-minute rounding tolerance, p = 0.30 and 0.10',
            inference='Descriptive; strata of 4-13 respondents; no group-difference test'), groups={})
        for g, codes in self.groups.items():
            out['groups'][g] = dict(n=len(codes), codes=codes, departures=self.departures(codes),
                                    median_q060_over_b=self.ratios(codes), agreement=self.agreement(codes))
        (ROOT/'results/reliability20_strata_v12.json').write_text(json.dumps(out, indent=1)+'\n')
        return out


if __name__ == '__main__':
    res = StratifiedPanel().run()
    for g, v in res['groups'].items():
        print(g, v['n'], v['departures'], {s: round(x, 2) for s, x in v['median_q060_over_b'].items()},
              {s: (round(a['icc_A1'], 3), [round(x, 3) for x in a['icc_bootstrap_95']]) if 'icc_A1' in a else a
               for s, a in v['agreement'].items()})
