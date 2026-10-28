"""Combine complete per-case exact-regret records and audit reported claims."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=7);args=ap.parse_args()
    parts=HERE/('parts' if args.n==7 else 'parts_n'+str(args.n))
    files=[parts/('A'+str(i)+'.json') for i in range(1,9)]
    cases=[json.loads(p.read_text()) for p in files]
    rows=[dict(r,case=c['id']) for c in cases for r in c['records']]
    assert len(rows)==96
    assert len({(r['case'],r['profile'],r['policy'],r['alpha']) for r in rows})==96
    methods=sorted({m['method'] for r in rows for m in r['methods']})
    collect=lambda method:[next(m for m in r['methods'] if m['method']==method) for r in rows]
    ci,mr=collect('CI_2pct'),collect('MR_CI_nominal_1pct')
    improvements=[float(100*(1-Fraction(m['worst_regret_fraction'])/Fraction(c['worst_regret_fraction'])))
                  for c,m in zip(ci,mr) if Fraction(c['worst_regret_fraction'])>0]
    for r,c,m in zip(rows,ci,mr):
        assert r['exact_global_candidate_certificate']
        assert Fraction(m['worst_regret_fraction'])<=Fraction(c['worst_regret_fraction'])
        assert Fraction(m['J_plus_fraction'])<=Fraction(r['robust_budget_fraction'])
        assert Fraction.from_float(m['J_nominal'])<=Fraction(r['nominal_budget_fraction'])
        # A candidate excluded early has a certified regret LB larger than the selected R.
        selected=Fraction(m['worst_regret_fraction'])
        for candidate in r['candidates']:
            if candidate['complete']:assert Fraction(candidate['regret_fraction'])>=selected
            else:assert Fraction(candidate['lower_bound_fraction'])>selected
        assert len(r['candidates'])==r['dual_budget_candidates']
    aggregation={}
    numeric=['worst_regret','rectangular_regret_upper','rectangular_bound_tightening_pct',
             'neutral_mean_regret_pct','neutral_max_regret_pct','smooth_mean_regret_pct',
             'robust_excess_pct','nominal_excess_vs_CI_pct','universal_grid_regret_bound']
    for method in methods:
        values=collect(method)
        aggregation[method]=dict(decisions=len(values),
            mean={k:float(np.mean([v[k] for v in values])) for k in numeric},
            maximum={k:float(max(v[k] for v in values)) for k in numeric},
            rectangular_tightening_settings=sum(Fraction(v['worst_regret_fraction'])<Fraction(v['rectangular_regret_upper_fraction']) for v in values),
            material_rectangular_tightening_settings_1e9=sum(v['rectangular_regret_upper']-v['worst_regret']>1e-9 for v in values),
            zero_regret_settings=sum(Fraction(v['worst_regret_fraction'])==0 for v in values))
    baseline_audits=[v for r in rows for v in r.get('archived_selection_audit',{}).values()]
    summary=dict(n=args.n,settings=96,cases=8,slots=3,grid='four',
        domain='All modal-feasible plans from all directed elementary routes of the first n tasks; same capped/uncapped eligible domain for every challenger and method',
        full_enumerated_plans_per_case=cases[0]['total_plans'],
        eligible_domain_min=min(r['eligible_plans'] for r in rows),eligible_domain_max=max(r['eligible_plans'] for r in rows),
        complete_global_certificates=sum(r['exact_global_candidate_certificate'] for r in rows),
        total_dual_budget_candidates=sum(r['dual_budget_candidates'] for r in rows),
        max_dual_budget_candidates=max(r['dual_budget_candidates'] for r in rows),
        case_processing_seconds=sum(c['seconds'] for c in cases),decision_processing_seconds=sum(r['seconds'] for r in rows),
        pair_queries=sum(r['evaluator_stats']['pair_queries'] for r in rows),
        primitive_pair_oracle_evaluations=sum(max(r['oracle_stats']['pairs'] for r in c['records'] if r['profile']==p) for c in cases for p in ['arithmetic','top_heavy']),
        integer_maxflow_calls=sum(max(r['oracle_stats']['flows'] for r in c['records'] if r['profile']==p) for c in cases for p in ['arithmetic','top_heavy']),
        upper_bound_pair_prunings=sum(r['evaluator_stats']['upper_pruned'] for r in rows),
        MR_vs_CI=dict(strict_R_wins=sum(Fraction(m['worst_regret_fraction'])<Fraction(c['worst_regret_fraction']) for c,m in zip(ci,mr)),
            equal_R=sum(Fraction(m['worst_regret_fraction'])==Fraction(c['worst_regret_fraction']) for c,m in zip(ci,mr)),
            material_R_wins_1e9=sum(c['worst_regret']-m['worst_regret']>1e-9 for c,m in zip(ci,mr)),
            reporting_threshold='1e-9 absolute J units (the existing v12 numeric tolerance); exact wins and material wins both reported',
            R_losses=0,changed_plan=sum(c['plan_index']!=m['plan_index'] for c,m in zip(ci,mr)),
            positive_CI_regret_settings=len(improvements),mean_relative_R_improvement_pct=float(np.mean(improvements)),max_relative_R_improvement_pct=max(improvements),
            neutral_mean_wins=sum(m['neutral_mean_regret_pct']<c['neutral_mean_regret_pct']-1e-10 for c,m in zip(ci,mr)),
            neutral_mean_losses=sum(m['neutral_mean_regret_pct']>c['neutral_mean_regret_pct']+1e-10 for c,m in zip(ci,mr)),
            nominal_budget_violations=0,robust_budget_violations=0),
        archived_baseline_decisions_audited=len(baseline_audits),archived_baseline_selections_changed=sum(not a['same'] for a in baseline_audits),
        maximum_nominal_horizon_score_change=max(r['max_nominal_horizon_score_change'] for r in rows),methods=aggregation,
        scope='Exact finite-domain regret certificates on the explicitly documented dyadic numeric model; constructed memberships and completions, not field evidence; larger n concerns the stated elementary-route domain only')
    protocol=next((p for p in [HERE.parent/'results/endpoint_regret_v13_protocol.json',HERE.parent/'protocol_v13.json'] if p.exists()),HERE.parent/'protocol_v13.json')
    combined=dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest() if protocol.exists() else None,
        part_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        summary=summary,raw=cases)
    output=HERE/('endpoint_regret_v13_n'+str(args.n)+'.json')
    output.write_text(json.dumps(combined,indent=1)+'\n')
    (HERE/('summary_n'+str(args.n)+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
