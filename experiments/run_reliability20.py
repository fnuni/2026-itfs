"""Twenty-expert R1/R2 campaign: import of the returned PDF forms, agreement, planning transfer.

The returned forms (``Test retest 20 esperti/03_Compilati_R1`` and ``04_Compilati_R2``)
are the only data source. Literal answers, dates and checksums stay in
``05_Analisi``; the public release receives pseudonymous numerical answers.

    python3 experiments/run_reliability20.py --import-forms   # PDF forms -> public CSV, then analyses
    python3 experiments/run_reliability20.py                  # analyses from the public CSV
"""
from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, math, statistics as st, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
_spec = importlib.util.spec_from_file_location('collector', ROOT / 'elicitation/reliability20/raccogli_e_analizza.py')
collector = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(collector)
SCENARIOS, ENDPOINTS = collector.SCENARIOS, collector.ENDPOINTS
PUBLIC = ROOT / 'data/reliability20'
KIT = ROOT.parents[1] / 'Test retest 20 esperti'
CONTEXTS = [('s1', 's3'), ('s1', 's4'), ('s2', 's3'), ('s2', 's4')]
SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def coherent(row, scenario):
    vals = [collector.numeric(row[scenario + '_' + e]) for e in ENDPOINTS]
    if None in vals or vals[0] <= 0 or vals != sorted(vals):
        return False
    a, c = (collector.numeric(row[scenario + '_' + e]) for e in ['a', 'c'])
    return (a is None or a <= vals[0]) and (c is None or c >= vals[-1])


class FormImport:
    """Returned PDF forms -> private literal records and public numerical CSV."""

    def __init__(self, kit: Path = KIT):
        self.kit, self.out = kit, kit / '05_Analisi'

    @staticmethod
    def anomalies(rows):
        found = []
        for r in rows:
            for s in SCENARIOS:
                for e in ENDPOINTS:
                    lit = r[s + '_' + e]
                    if collector.numeric(lit) is None:
                        reason = 'unknown' if collector.normal(lit) == 'non so' else 'approximate value'
                        found.append(dict(code=r['code'], session=r['session'], item=s + '_' + e, answer=reason))
                vals = [collector.numeric(r[s + '_' + e]) for e in ENDPOINTS]
                if None not in vals and vals != sorted(vals):
                    found.append(dict(code=r['code'], session=r['session'], item=s,
                                      answer='nonmonotone thresholds ' + '/'.join(f'{v:g}' for v in vals)))
        return found

    def run(self):
        rows, errors = collector.collect([self.kit / '03_Compilati_R1', self.kit / '04_Compilati_R2'])
        if errors:
            raise ValueError(errors)
        by = {}
        for r in rows:
            by.setdefault(r['code'], {})[r['session']] = r
        if len(rows) != 40 or len(by) != 20 or any(set(v) != {'R1', 'R2'} for v in by.values()):
            raise ValueError('Expected 20 paired R1/R2 forms')
        if not all(collector.normal(r['consent']) == 'si' and collector.normal(r['eligibility']) == 'si' for r in rows):
            raise ValueError('Form without consent or declared relevant experience')
        if any(p['R1']['scenario_order'] != p['R2']['scenario_order'] for p in by.values()):
            raise ValueError('Scenario order differs across sessions')
        dates = {(r['code'], r['session']): collector.date(r['session_date']) for r in rows}
        if None in dates.values() or any(dates[(c, 'R2')] <= dates[(c, 'R1')] for c in by):
            raise ValueError('Missing session date or R2 not after R1')
        self.out.mkdir(exist_ok=True)
        with (self.out / 'risposte_originali_PRIVATO.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        numeric_rows = []
        for r in sorted(rows, key=lambda r: (r['session'], r['code'])):
            rr = dict(code=r['code'], session=r['session'],
                      gap_days=(dates[(r['code'], 'R2')] - dates[(r['code'], 'R1')]).days,
                      consulted_r1=int(collector.normal(r['consulted_r1']) == 'si'),
                      remembers_numbers=int(collector.normal(r['memory']) in ['alcune', 'molte']),
                      new_information=int(collector.normal(r['new_information']) == 'si'),
                      checks_passed=int(r['final_c1'].upper() == 'B' and r['final_c2'].upper() == 'A'),
                      ceiling_check_passed=int(collector.normal(r['ceiling030']) == 'no'))
            for s in SCENARIOS:
                rr[s + '_coherent'] = int(coherent(r, s))
                for e in ENDPOINTS + ['a', 'c', 'confidence']:
                    val = collector.numeric(r[s + '_' + e]); rr[s + '_' + e] = val if val is not None else ''
                rr[s + '_c_unbounded'] = int(collector.normal(r[s + '_c']) == 'nessuno')
            for k in ['completion_minutes', 'practicality']:
                val = collector.numeric(r[k]); rr[k] = val if val is not None else ''
            numeric_rows.append(rr)
        PUBLIC.mkdir(parents=True, exist_ok=True)
        with (PUBLIC / 'responses_numeric.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(numeric_rows[0])); w.writeheader(); w.writerows(numeric_rows)
        anomalies = self.anomalies(rows)
        save(self.out / 'audit_importazione_PRIVATO.json', dict(
            source='returned PDF forms', source_manifest=[dict(code=r['code'], session=r['session'], file=r['source_file'],
                                                               sha256=r['source_sha256'], scenario_order=r['scenario_order']) for r in rows],
            form_notes=[dict(code=r['code'], session=r['session'], note=r['form_anomaly']) for r in rows if r['form_anomaly']],
            anomalies=anomalies))
        d = sorted(dates.values())
        save(PUBLIC / 'import_audit.json', dict(
            source='Returned PDF forms of sessions R1 and R2 (20 anonymous experts)', forms=40, experts=20,
            source_bundle_sha256=hashlib.sha256(''.join(sorted(r['source_sha256'] for r in rows)).encode()).hexdigest(),
            collection_range=[d[0].isoformat(), d[-1].isoformat()],
            scenario_rotations={o: sum(r['scenario_order'] == o for r in rows) // 2 for o in sorted({r['scenario_order'] for r in rows})},
            numerical_anomalies=anomalies, imputation='none'))


# --------------------------------------------------------------------------- analyses
def load():
    rows = list(csv.DictReader((PUBLIC / 'responses_numeric.csv').open()))
    by = {}
    for r in rows:
        by.setdefault(r['code'], {})[r['session']] = r
    assert len(rows) == 40 and len(by) == 20 and all(set(v) == {'R1', 'R2'} for v in by.values())
    return rows, [(v['R1'], v['R2']) for _, v in sorted(by.items())]


def cohorts_of(pairs):
    return {'all': pairs,
            'no_recall_or_new_information': [p for p in pairs if p[1]['remembers_numbers'] == '0'
                                             and p[1]['new_information'] == '0']}


def summarize(rows, pairs):
    cohorts = cohorts_of(pairs)
    estimates = {}
    for cohort, pp in cohorts.items():
        estimates[cohort] = {}
        for s in SCENARIOS:
            for e in ENDPOINTS:
                vals = [tuple(collector.numeric(r[s + '_' + e]) for r in p) for p in pp]
                estimates[cohort][s + '_' + e] = collector.agreement([v for v in vals if None not in v])
    estimates['coherent'] = {}
    for s in SCENARIOS:
        pp = [p for p in pairs if all(r[s + '_coherent'] == '1' for r in p)]
        for e in ENDPOINTS:
            estimates['coherent'][s + '_' + e] = collector.agreement(
                [tuple(collector.numeric(r[s + '_' + e]) for r in p) for p in pp])
    gaps = [int(a['gap_days']) for a, b in pairs]
    result = dict(
        source='returned PDF forms', forms=40, experts=20,
        cohort_sizes={k: len(v) for k, v in cohorts.items()},
        session_interval_days=dict(minimum=min(gaps), median=st.median(gaps), maximum=max(gaps)),
        primary_item='q060 separately by scenario',
        bootstrap=dict(replicates=collector.BOOTSTRAP_REPLICATES, seed=collector.BOOTSTRAP_SEED,
                       resampling_unit='paired expert', interval='percentile'),
        r2_consultation=sum(r['consulted_r1'] == '1' for r in rows if r['session'] == 'R2'),
        r2_recall=sum(r['remembers_numbers'] == '1' for r in rows if r['session'] == 'R2'),
        r2_new_information=sum(r['new_information'] == '1' for r in rows if r['session'] == 'R2'),
        final_checks_passed=sum(r['checks_passed'] == '1' for r in rows),
        ceiling_checks_passed=sum(r['ceiling_check_passed'] == '1' for r in rows),
        estimates=estimates,
        scenario_quality={s: dict(
            complete_numeric=sum(all(collector.numeric(r[s + '_' + e]) is not None for e in ENDPOINTS) for r in rows),
            coherent_complete=sum(r[s + '_coherent'] == '1' for r in rows),
            unbounded_upper_support=sum(r[s + '_c_unbounded'] == '1' for r in rows),
            missing_lower_support=sum(r[s + '_a'] == '' for r in rows)) for s in SCENARIOS},
        session_summaries={})
    for session in ['R1', 'R2']:
        rr = [r for r in rows if r['session'] == session]; result['session_summaries'][session] = {}
        for k in ['completion_minutes', 'practicality']:
            vals = [collector.numeric(r[k]) for r in rr]; vals = [v for v in vals if v is not None]
            result['session_summaries'][session][k] = dict(n=len(vals), median=st.median(vals))
    save(ROOT / 'results/reliability20.json', result)
    return cohorts, result


class PlanningTransfer:
    """Certified ceiling-0.30 transfer of paired q060 endpoints to ten S10 instances."""

    def __init__(self, pairs):
        from instances_io import CanonicalInstances
        self.pairs, self.ci = pairs, CanonicalInstances()

    def run(self):
        from exact_small import SubsetTSP
        from exact_fff import partitions
        records, coverage = [], []
        for ts, ss in CONTEXTS:
            pp = [p for p in self.pairs if all(r[s + '_coherent'] == '1' for r in p for s in [ts, ss])]
            bases = [st.median(float(p[0][s + '_b']) for p in pp) for s in [ts, ss]]
            factors = [np.array([[float(p[j][s + '_q060']) / base for s, base in zip([ts, ss], bases)] for p in pp])
                       for j in [0, 1]]
            for j in [0, 1]:
                q = [sorted(float(p[j][s + '_q060']) for p in pp)[math.ceil(.9 * len(pp)) - 1] for s in [ts, ss]]
                marg = [sum(float(p[j][s + '_q060']) <= th for p in pp) for s, th in zip([ts, ss], q)]
                joint = sum(all(float(p[j][s + '_q060']) <= th for s, th in zip([ts, ss], q)) for p in pp)
                coverage.append(dict(context=ts + '+' + ss, session='R' + str(j + 1), n=len(pp),
                                     target_marginal=.9, thresholds=q, marginal_counts=marg, joint_count=joint))
            for seed in SEEDS:
                inst = self.ci.build('S10', seed)
                tsp = SubsetTSP(np.array([[inst.modal_travel(i, j) for j in range(11)] for i in range(11)]))
                travel = tsp.best
                service = np.array([sum(inst.service_modal[i + 1] for i in range(10) if m >> i & 1) for m in range(1024)])
                groups = np.array([tuple(g) + (0,) * (2 - len(g)) for g in partitions(1023, 2)], dtype=int)
                durations = [f[:, 0, None] * travel + f[:, 1, None] * service for f in factors]
                union = np.concatenate(durations, axis=0)
                cons = [np.median(f, axis=0) for f in factors]
                vectors = dict(consensus_R1=cons[0][0] * travel + cons[0][1] * service,
                               consensus_R2=cons[1][0] * travel + cons[1][1] * service,
                               joint_R1=np.max(durations[0], axis=0), joint_R2=np.max(durations[1], axis=0),
                               joint_union=np.max(union, axis=0),
                               cartesian_union=max(f[:, 0].max() for f in factors) * travel
                               + max(f[:, 1].max() for f in factors) * service)
                scores = {k: np.max(v[groups], axis=1) for k, v in vectors.items()}
                idx = {k: int(np.argmin(v)) for k, v in scores.items()}
                minima = {k: float(scores[k][idx[k]]) for k in scores}
                chosen = groups[idx['joint_R1']]; L = minima['joint_R1']
                under_R2 = float(np.max(durations[1][:, chosen])); excess = max(0., under_R2 - L)
                protected = np.all(durations[1][:, chosen] <= L + 1e-7, axis=1)
                assert minima['joint_union'] + 1e-7 >= max(minima['joint_R1'], minima['joint_R2'])
                assert minima['cartesian_union'] + 1e-7 >= minima['joint_union']
                records.append(dict(
                    context=ts + '+' + ss, seed=seed, n=len(pp), reference_b_R1=bases,
                    minimum_shift_minutes=minima, joint_R2_change_pct=100 * (minima['joint_R2'] / L - 1),
                    R1_joint_plan_masks=list(map(int, chosen)), R1_joint_plan_tours=[tsp.tour(int(m)) for m in chosen],
                    R1_joint_plan_R2_duration=under_R2, R1_joint_plan_R2_excess_minutes=excess,
                    R1_joint_plan_R2_excess_pct=100 * excess / L,
                    R1_joint_plan_R2_joint_protected_count=int(protected.sum()),
                    union_over_R1_joint_pct=100 * (minima['joint_union'] / L - 1),
                    cartesian_over_joint_union_pct=100 * (minima['cartesian_union'] / minima['joint_union'] - 1)))
        save(ROOT / 'results/reliability20_planning.json', dict(protocol=dict(
            source='20 paired experts, returned PDF forms', instances='Ten canonical S10 seeds',
            ceiling=.30, endpoint='q060', reference='Context paired-coherent R1 median b, fixed across sessions',
            coherent_filter='All five b/q endpoints numerical and monotone in both sessions for both context scenarios',
            exact='Held-Karp shortest subset tours and exhaustive two-route partitions',
            R1_plan_tie_break='First enumerated minimum-duration two-route partition'), raw=records, coverage=coverage))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--import-forms', action='store_true', help='Read the returned PDF forms (private folders required)')
    args = ap.parse_args()
    if args.import_forms:
        FormImport().run()
    rows, pairs = load()
    cohorts, result = summarize(rows, pairs)
    PlanningTransfer(pairs).run()
    print(json.dumps({k: result[k] for k in ['forms', 'experts', 'cohort_sizes', 'scenario_quality']}, indent=2))


if __name__ == '__main__':
    main()
