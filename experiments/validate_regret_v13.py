"""Replay exact full-domain regret certificates independently of search order.

For each reported plan: its stated worst challenger supplies an exact lower
witness; every challenger whose endpoint upper bound is larger than the
reported regret is evaluated again. All others have a direct exact upper
certificate. Candidate lower witnesses and complete candidate coverage prove
the selected dual-budget optimum without trusting the search's early exits.
"""
from pathlib import Path
from fractions import Fraction
import argparse,copy,hashlib,json,time
import numpy as np
import run_seven_stop as driver
from exact_flow_oracle import ExactFlowOracle

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=7);args=parser.parse_args()
    name='endpoint_regret_v13_n'+str(args.n)+'.json'
    source=next((p for p in [HERE/name,driver.V12/'results'/name] if p.exists()),None)
    if source is None:raise FileNotFoundError(name)
    data=json.loads(source.read_text());assert len(data['raw'])==8
    driver.decision.N=args.n;start=time.perf_counter()
    input_records={r['id']:r for r in driver.load_external_records()}
    stats=dict(settings=0,method_certificates=0,endpoint_pruned_pairs=0,flow_checked_pairs=0,
               worst_challenger_witnesses=0,candidate_witnesses=0,exact_candidate_domains=0)
    for case_data in data['raw']:
        original=driver.DecisionCase(input_records[case_data['id'].split('-')[0]])
        original.id=case_data['id'];case=copy.copy(original);case.L=float(original.L+driver.TOL)
        assert case_data['n']==args.n and case_data['modal_feasible_plans']==len(case.plans)
        linear=driver.piecewise_linear_risks(case,'four')
        for profile,weights in driver.PROFILES.items():
            oracle=ExactFlowOracle(original,weights,'four');lower,upper=oracle.bounds()
            assert np.all(np.isfinite(upper[case.plans]));maximum=upper[case.plans].max(axis=1)
            pl_owa=case.plan_owa(linear,weights)
            paircache={}
            for row in case_data['records']:
                if row['profile']!=profile:continue
                stats['settings']+=1;alpha=row['alpha']
                units=driver.ObjectiveUnits(case,weights,alpha,lower,upper)
                eligible=np.ones(len(case.plans),dtype=bool) if row['policy']=='uncapped' else maximum<=driver.CAP+driver.TOL
                domain=np.flatnonzero(eligible)
                def valid_challenger(y):
                    return 0<=y<len(case.plans) and bool(eligible[y])
                assert len(domain)==row['eligible_plans']
                methods={m['method']:m for m in row['methods']};assert len(methods)==4
                mm=min((int(i) for i in domain),key=lambda i:(units.up[i],float(case.workload[i]),float(maximum[i]),i))
                assert methods['MM']['plan_index']==mm
                robust=Fraction(102,100)*units.ratio(units.up[mm]);assert robust==Fraction(row['robust_budget_fraction'])
                pl=alpha*case.workload/case.scale+(1-alpha)*pl_owa
                assert methods['PL']['plan_index']==driver.choose(pl,case.workload,maximum,eligible)
                budget=np.array([False]*len(case.plans));budget[domain]=[units.ratio(units.up[int(i)])<=robust for i in domain]
                ci=driver.choose(pl,case.workload,maximum,budget)
                assert ci==methods['CI_2pct']['plan_index']
                nominal=Fraction(101,100)*Fraction.from_float(float(pl[ci]));assert nominal==Fraction(row['nominal_budget_fraction'])
                candidate_ids={int(i) for i in np.flatnonzero(budget) if Fraction.from_float(float(pl[i]))<=nominal}
                assert candidate_ids=={int(c['plan']) for c in row['candidates']};stats['exact_candidate_domains']+=1
                def pair_value(x,y):
                    key=(x,y)
                    if key not in paircache:paircache[key]=oracle.pair_fraction(case.plans[x],case.plans[y])
                    return Fraction(units.common[x]-units.common[y],units.D)+(1-units.alpha)*paircache[key]
                for method,record in methods.items():
                    stats['method_certificates']+=1;x=record['plan_index'];R=Fraction(record['worst_regret_fraction'])
                    assert x in domain and R>=0
                    assert record['routes']==[list(case.sequences[r]) for r in case.plans[x]]
                    y=record['certificate']['worst_challenger'];assert valid_challenger(y);assert pair_value(x,y)==R
                    stats['worst_challenger_witnesses']+=1
                    R_units=R*units.D;assert R_units.denominator==1;R_units=int(R_units)
                    for yy in domain:
                        yy=int(yy)
                        if units.up[x]-units.lo[yy]<=R_units:
                            stats['endpoint_pruned_pairs']+=1
                        else:
                            assert pair_value(x,yy)<=R;stats['flow_checked_pairs']+=1
                    assert Fraction(record['J_plus_fraction'])==units.ratio(units.up[x])
                    assert R<=units.ratio(units.up[x]-min(units.lo[int(j)] for j in domain))
                chosen=methods['MR_CI_nominal_1pct'];best=Fraction(chosen['worst_regret_fraction'])
                assert units.ratio(units.up[chosen['plan_index']])<=robust
                assert Fraction.from_float(float(pl[chosen['plan_index']]))<=nominal
                for candidate in row['candidates']:
                    x=int(candidate['plan']);y=int(candidate['worst_challenger'])
                    assert valid_challenger(y)
                    key='regret_fraction' if candidate['complete'] else 'lower_bound_fraction'
                    reported=Fraction(candidate[key]);assert pair_value(x,y)==reported
                    assert reported>=best
                    if not candidate['complete']:assert reported>best
                    stats['candidate_witnesses']+=1
                assert best<=Fraction(methods['CI_2pct']['worst_regret_fraction'])
        print(case_data['id'],'full-domain certificates replayed',flush=True)
    assert stats['settings']==96 and stats['method_certificates']==384
    output=dict(status='pass',n=args.n,seconds=time.perf_counter()-start,statistics=stats,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Exact integer/dyadic certificate replay over every eligible challenger; samples and search ordering are not used')
    (HERE/('validation_n'+str(args.n)+'.json')).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
