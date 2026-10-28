"""Globally certified seven-stop regret decisions on the full v12 domains.

All robust risk comparisons and regret values use exact integer units. Input
workloads/normalizer are the immutable v12 doubles interpreted as dyadic
rationals; alpha and OWA/grid parameters are intended decimal rationals.
Sampled completions only choose useful challengers: every lower bound actually
used for pruning is recomputed by the exact shared-leg flow oracle.
"""
from pathlib import Path
from fractions import Fraction
from math import lcm
import argparse,copy,hashlib,json,sys,time
import numpy as np

def locate_package():
    for ancestor in Path(__file__).resolve().parents:
        for candidate in (ancestor,ancestor/'3-Secondo Invio v12/code and data/ms-vrppu-fr',ancestor/'code and data/ms-vrppu-fr'):
            if (candidate/'data/amazon_v4/instances.json').is_file() and (candidate/'src/endpoint_decision_v11.py').is_file():
                return candidate
    raise FileNotFoundError('Cannot locate the accompanying ms-vrppu-fr data/source package')
V12=locate_package()
sys.dont_write_bytecode=True;sys.path.insert(0,str(V12/'src'))
from external_instances_v4 import load_external_records
import endpoint_decision_v11 as decision
from endpoint_decision_v11 import DecisionCase,PROFILES,ALPHAS,CAP,TOL,choose
from endpoint_information_v11 import piecewise_linear_risks,completion_risks
from endpoint_certified_v12 import gap_exponent_risks,neutral_exponents
from exact_flow_oracle import ExactFlowOracle

class ObjectiveUnits:
    def __init__(self,case,weights,alpha,lower,upper):
        self.alpha=Fraction(str(alpha));self.one_minus=1-self.alpha
        weights=[Fraction(str(float(w))).limit_denominator(10**6) for w in weights]
        self.wden=lcm(*(w.denominator for w in weights))
        wn=np.array([int(w*self.wden) for w in weights],dtype=np.int64)
        self.rden=20*self.wden
        self.lower_num=np.sort(np.rint(20*lower[case.plans]).astype(np.int64),axis=1)[:,::-1]@wn
        self.upper_num=np.sort(np.rint(20*upper[case.plans]).astype(np.int64),axis=1)[:,::-1]@wn
        factor=self.alpha/Fraction.from_float(float(case.scale))
        ratios=[float(w).as_integer_ratio() for w in case.workload]
        self.D=lcm(factor.denominator*max(d for _,d in ratios),self.one_minus.denominator*self.rden)
        common=[factor.numerator*n*(self.D//(factor.denominator*d)) for n,d in ratios]
        riskunit=self.one_minus*self.D/self.rden
        assert riskunit.denominator==1
        self.common=common
        self.up=[c+int(u)*int(riskunit) for c,u in zip(common,self.upper_num)]
        self.lo=[c+int(u)*int(riskunit) for c,u in zip(common,self.lower_num)]
        self.riskunit=self.one_minus*self.D
    def pair_value(self,x,y,risk):
        value=risk*self.riskunit
        assert value.denominator==1
        return self.common[x]-self.common[y]+int(value)
    def ratio(self,value):return Fraction(value,self.D)

class RegretEvaluator:
    def __init__(self,case,oracle,units,eligible,seed_challengers,pair_cache):
        self.case,self.oracle,self.units=case,oracle,units
        self.domain=[int(i) for i in np.flatnonzero(eligible)]
        self.seeds=list(dict.fromkeys(int(i) for i in seed_challengers))
        self.pair_cache=pair_cache;self.full={};self.stats=dict(pair_queries=0,pair_cache_hits=0,upper_pruned=0,cutoff_exits=0)
        # Sorting by L_y is equivalent to sorting U_x-L_y, for every x.
        self.challengers=sorted(self.domain,key=lambda y:units.lo[y])
    def pair(self,x,y):
        if x==y:return 0
        key=(int(x),int(y));self.stats['pair_queries']+=1
        if key in self.pair_cache:
            self.stats['pair_cache_hits']+=1;risk=self.pair_cache[key]
        else:
            risk=self.oracle.pair_fraction(self.case.plans[x],self.case.plans[y]);self.pair_cache[key]=risk
            if len(self.oracle.gap_cache)>150000:self.oracle.gap_cache.clear()
        return self.units.pair_value(x,y,risk)
    def evaluate(self,x,cutoff=None):
        if x in self.full:return self.full[x]
        started=time.perf_counter();best=0;challenger=x;tested=set([x]);queries_before=self.stats['pair_queries']
        for y in self.seeds:
            if y in tested:continue
            tested.add(y)
            if self.units.up[x]-self.units.lo[y]<=best:continue
            value=self.pair(x,y)
            if value>best:best,challenger=value,y
            if cutoff is not None and best>cutoff:
                self.stats['cutoff_exits']+=1
                return dict(plan=x,complete=False,lower_bound=best,worst_challenger=challenger,
                            queried=self.stats['pair_queries']-queries_before,seconds=time.perf_counter()-started)
        remaining_bound=None
        for pos,y in enumerate(self.challengers):
            bound=self.units.up[x]-self.units.lo[y]
            if bound<=best:
                self.stats['upper_pruned']+=len(self.challengers)-pos;remaining_bound=bound;break
            if y in tested:continue
            tested.add(y);value=self.pair(x,y)
            if value>best:best,challenger=value,y
            if cutoff is not None and best>cutoff:
                self.stats['cutoff_exits']+=1
                return dict(plan=x,complete=False,lower_bound=best,worst_challenger=challenger,
                            queried=self.stats['pair_queries']-queries_before,seconds=time.perf_counter()-started)
        rec=dict(plan=x,complete=True,regret=best,worst_challenger=challenger,
                 queried=self.stats['pair_queries']-queries_before,seconds=time.perf_counter()-started,
                 remaining_upper_bound=remaining_bound)
        self.full[x]=rec;return rec

def compact_record(rec,units):
    out={k:v for k,v in rec.items() if k not in ['regret','lower_bound','remaining_upper_bound']}
    for key in ['regret','lower_bound','remaining_upper_bound']:
        if rec.get(key) is not None:
            value=units.ratio(rec[key]);out[key]=float(value);out[key+'_fraction']=str(value)
    return out

def execute_case(record,args):
    start=time.perf_counter();decision.N=args.n;original_case=DecisionCase(record)
    original_case.id=record['id']+'-'+str(args.n)
    case=copy.copy(original_case);case.L=float(original_case.L+TOL)
    original_linear=piecewise_linear_risks(original_case,'four')
    linear=piecewise_linear_risks(case,'four')
    archives=json.loads((V12/'results/endpoint_certified_v12.json').read_text())['raw']
    archive=next((r for r in archives if r['id']==case.id),None)
    samples=[completion_risks(case,p) for p in (.45,1.,2.2,.1,10.)]
    frozen=json.loads((V12/'results/endpoint_certified_v12_protocol.json').read_text())
    rng=np.random.default_rng(frozen['neutral_seed']+int(record['id'][1:])-1)
    exponents=neutral_exponents(case,rng,frozen['neutral_draws'],*frozen['neutral_exponent_range'])
    samples += [gap_exponent_risks(case,e) for e in exponents]
    code_files=[Path(__file__).resolve(),Path(sys.modules[ExactFlowOracle.__module__].__file__).resolve()]
    result=dict(id=case.id,n=args.n,slots=3,
                code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
                source_sha256=hashlib.sha256((V12/'data/amazon_v4/instances.json').read_bytes()).hexdigest(),
                neutral_protocol_sha256=hashlib.sha256((V12/'results/endpoint_certified_v12_protocol.json').read_bytes()).hexdigest(),total_plans=case.total_plans,modal_feasible_plans=len(case.plans),records=[],
                reference_horizon=original_case.L,effective_horizon=case.L,horizon_rule='exact dyadic float(v12 L + 1e-9), preserving the archived endpoint-comparison RHS',
                objective_inputs='v12 precomputed workload and scale floats as exact dyadic numbers; intended decimal alpha/profile/grid rational parameters')
    for profile,weights in PROFILES.items():
        oracle=ExactFlowOracle(original_case,weights,'four');lower,upper=oracle.bounds()
        v12lo,v12up=original_case.route_bounds('four')
        assert np.all(np.isfinite(upper[case.plans]))
        assert np.array_equal(lower[case.modal_ok],v12lo[case.modal_ok])
        assert np.array_equal(upper[case.modal_ok],v12up[case.modal_ok])
        maximum=upper[case.plans].max(axis=1)
        up_owa=case.plan_owa(upper,weights);pl_owa=case.plan_owa(linear,weights)
        original_pl_owa=case.plan_owa(original_linear,weights)
        sample_owas=np.array([case.plan_owa(r,weights) for r in samples])
        pair_cache={}
        for policy in ('uncapped','capped'):
            eligible=np.ones(len(case.plans),bool) if policy=='uncapped' else maximum<=CAP+TOL
            domain=np.flatnonzero(eligible)
            for alpha in ALPHAS:
                if args.max_settings and len(result['records'])>=args.max_settings:
                    result['seconds']=time.perf_counter()-start;return result
                settingstart=time.perf_counter();units=ObjectiveUnits(case,weights,alpha,lower,upper)
                common=alpha*case.workload/case.scale;up=common+(1-alpha)*up_owa;pl=common+(1-alpha)*pl_owa
                # Exact robust minimizer, with the v12 deterministic tie break.
                key=lambda i:(units.up[int(i)],float(case.workload[i]),float(maximum[i]),int(i))
                mm=int(min(domain,key=key));PL=choose(pl,case.workload,maximum,eligible)
                robust_budget=Fraction(102,100)*units.up[mm]
                budget=np.array([bool(eligible[i] and units.up[i]<=robust_budget) for i in range(len(case.plans))])
                ci=choose(pl,case.workload,maximum,budget)
                nominal_budget=Fraction(101,100)*Fraction.from_float(float(pl[ci]))
                candidates=[int(i) for i in np.flatnonzero(budget)
                            if Fraction.from_float(float(pl[i]))<=nominal_budget]
                assert ci in candidates
                sample_obj=common[None,:]+(1-alpha)*sample_owas
                seed_ids=[mm,PL,ci]
                seed_ids += [int(domain[np.argmin(row[domain])]) for row in sample_obj]
                evaluator=RegretEvaluator(case,oracle,units,eligible,seed_ids,pair_cache)
                selected={};evals={};rectangular_min=min(units.lo[int(i)] for i in domain)
                grid_delta=max(oracle.widths)
                standard_bound=(1-units.alpha)*grid_delta+Fraction(2,100)*units.ratio(units.up[mm])
                for method,ix in [('CI_2pct',ci),('MM',mm),('PL',PL)]:
                    rec=evaluator.evaluate(ix);selected[method]=ix;evals[method]=rec
                bestplan=ci;bestregret=evals['CI_2pct']['regret'];candidate_audits=[]
                # Sample bounds are used solely to order candidates; tested pair lower bounds are exact.
                samples_min=sample_obj[:,domain].min(axis=1)
                sample_lb=np.max(sample_obj[:,candidates]-samples_min[:,None],axis=0)
                order=sorted(zip(candidates,sample_lb),key=lambda t:(t[1],float(case.workload[t[0]]),t[0]))
                for ix,_ in order:
                    rec=evaluator.evaluate(ix,cutoff=bestregret)
                    candidate_audits.append(compact_record(rec,units))
                    if rec['complete']:
                        tie=lambda i:(float(case.workload[i]),float(maximum[i]),int(i))
                        if rec['regret']<bestregret or (rec['regret']==bestregret and tie(ix)<tie(bestplan)):
                            bestplan,bestregret=ix,rec['regret']
                    else:assert rec['lower_bound']>bestregret
                selected['MR_CI_nominal_1pct']=bestplan;evals['MR_CI_nominal_1pct']=evaluator.evaluate(bestplan)
                rows=[]
                for method,ix in selected.items():
                    rec=evals[method]
                    assert rec['complete']
                    assert rec['regret']<=units.up[ix]-rectangular_min
                    assert units.ratio(rec['regret'])<=(1-units.alpha)*grid_delta+units.ratio(units.up[ix]-units.up[mm])
                    tests=sample_obj[:,ix]-samples_min
                    rows.append(dict(method=method,plan_index=ix,routes=[list(case.sequences[r]) for r in case.plans[ix]],
                        worst_regret=float(units.ratio(rec['regret'])),worst_regret_fraction=str(units.ratio(rec['regret'])),
                        J_plus=float(units.ratio(units.up[ix])),J_plus_fraction=str(units.ratio(units.up[ix])),J_nominal=float(pl[ix]),
                        rectangular_regret_upper=float(units.ratio(units.up[ix]-rectangular_min)),
                        rectangular_regret_upper_fraction=str(units.ratio(units.up[ix]-rectangular_min)),
                        rectangular_bound_tightening_pct=float(100*(1-Fraction(rec['regret'],units.up[ix]-rectangular_min))) if units.up[ix]>rectangular_min else 0.,
                        v12_CI_grid_regret_bound=float(standard_bound) if units.up[ix]<=robust_budget else None,
                        v12_CI_grid_regret_bound_fraction=str(standard_bound) if units.up[ix]<=robust_budget else None,
                        universal_grid_regret_bound=float((1-units.alpha)*grid_delta+units.ratio(units.up[ix]-units.up[mm])),
                        robust_excess_pct=float(100*(Fraction(units.up[ix],units.up[mm])-1)),
                        nominal_excess_vs_CI_pct=float(100*(pl[ix]/pl[ci]-1)),
                        smooth_mean_regret_pct=float(np.mean(100*tests[:3]/samples_min[:3])),
                        neutral_mean_regret_pct=float(np.mean(100*tests[5:]/samples_min[5:])),
                        neutral_max_regret_pct=float(np.max(100*tests[5:]/samples_min[5:])),
                        certificate=compact_record(rec,units)))
                assert evals['MR_CI_nominal_1pct']['regret']<=evals['CI_2pct']['regret']
                recordout=dict(profile=profile,policy=policy,alpha=alpha,eligible_plans=int(eligible.sum()),
                    robust_budget_candidates=int(budget.sum()),dual_budget_candidates=len(candidates),
                    robust_budget_fraction=str(units.ratio(robust_budget)),nominal_budget_fraction=str(nominal_budget),
                    exact_global_candidate_certificate=True,all_challengers_domain='all eligible v12 plans',
                    max_nominal_horizon_score_change=float((1-alpha)*np.max(np.abs(pl_owa-original_pl_owa))),
                    archived_selection_audit={method:dict(archived_plan=int(next(r['plan_index'] for r in archive['records'] if r['grid']=='four' and r['profile']==profile and r['policy']==policy and r['alpha']==alpha and r['method']==archived_method)),
                        current_plan=int(selected[method]),same=bool(selected[method]==next(r['plan_index'] for r in archive['records'] if r['grid']=='four' and r['profile']==profile and r['policy']==policy and r['alpha']==alpha and r['method']==archived_method))) for method,archived_method in [('MM','minimax'),('PL','piecewise_linear'),('CI_2pct','ci_0.02')]} if archive is not None else {},
                    candidates=candidate_audits,methods=rows,seconds=time.perf_counter()-settingstart,
                    evaluator_stats=evaluator.stats.copy(),oracle_stats=oracle.stats.copy(),pair_cache_size=len(pair_cache))
                result['records'].append(recordout)
                print(json.dumps(dict(case=case.id,profile=profile,policy=policy,alpha=alpha,
                    candidates=len(candidates),seconds=recordout['seconds'],queries=evaluator.stats['pair_queries'],
                    R_CI=rows[0]['worst_regret'],R_MR=rows[-1]['worst_regret'],oracle_stats=oracle.stats)),flush=True)
                dest=Path(__file__).with_name('parts' if args.n==7 else 'parts_n'+str(args.n));dest.mkdir(exist_ok=True)
                result['seconds']=time.perf_counter()-start
                (dest/(record['id']+'.partial.json')).write_text(json.dumps(result,indent=1)+'\n')
    result['seconds']=time.perf_counter()-start;return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,choices=[7,8,9],default=7);ap.add_argument('--cases',nargs='*',default=[]);ap.add_argument('--max-settings',type=int,default=0);ap.add_argument('--force',action='store_true')
    args=ap.parse_args();records=[r for r in load_external_records() if not args.cases or r['id'] in args.cases]
    parts=Path(__file__).with_name('parts' if args.n==7 else 'parts_n'+str(args.n));parts.mkdir(exist_ok=True)
    for record in records:
        done=parts/(record['id']+'.json')
        if done.exists() and not args.max_settings and not args.force:
            print(record['id'],'already complete',flush=True);continue
        out=execute_case(record,args)
        target=parts/(record['id']+('.pilot.json' if args.max_settings else '.json'))
        target.write_text(json.dumps(out,indent=1)+'\n')
        print(json.dumps(dict(case=record['id'],settings=len(out['records']),seconds=out['seconds'],output=str(target))),flush=True)
if __name__=='__main__':main()
