"""Exact J_alpha and an oracle ceiling sweep, plus direct CC search.

Frozen design: ten canonical seeds in S10, S12 and F12; 7-stop restrictions
of all eight released Amazon scaffolds, first seven canonical task indices,
K=1,P=3. Restrictions retain original heterogeneous endpoint rules before
rescaling the horizon from three contiguous chunks. No outcome-based sampling.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time, platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from exact_fff import homothetic_columns, heterogeneous_columns, direct_reference, audit_plan
from instances_io import CanonicalInstances
from solver import RALNS, _State
from aggregators import FFF_REPORT
from cc_lns import CCLNS
from external_instances_v4 import ExternalMatrixInstance, load_external_records
from external_solver_v4 import EndpointCCLNS

ALPHAS=(.1,.3,.7)

def restrict(rec,n=7):
    inst=ExternalMatrixInstance(rec)
    inst.n,inst.K,inst.P=n,1,3
    chunks=[list(range(1,n+1))[lo:hi] for lo,hi in [(0,3),(3,5),(5,7)]]
    scale=sum(inst.route_endpoints(c)[1] for c in chunks)
    inst.L=scale/(.85*3)
    return inst,scale

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    cases=[];ci=CanonicalInstances()
    for family in ('S10','S12','F12'):
        for seed in (10,20,30,40,50,60,70,80,90,100):
            inst=ci.build(family,seed)
            init=RALNS(inst,FFF_REPORT,seed=0).initial()
            scale=_State(inst,init).makespan
            cases.append((family+'-'+str(seed),'homothetic',inst,scale))
    for rec in load_external_records():
        inst,scale=restrict(rec)
        cases.append((rec['id']+'-7','heterogeneous',inst,scale))
    if a.smoke:cases=[cases[0],cases[-1]]
    payload=dict(protocol=dict(date='2026-09-14',release='10.0.0',alphas=ALPHAS,
        search_seeds=[10,20,30],iterations=300,
        python=sys.version.split()[0],platform=platform.platform(),
        exact='All canonical unlabelled partitions and nondominated route columns',
        ceiling_reference='Every exact risk breakpoint; minimum workload; FFF tie break',
        timing='Reference certification has no common search deadline; search uses fixed iterations; wall times reported'),raw=[])
    path=ROOT/'results'/('direct_fff_smoke.json' if a.smoke else 'direct_fff.json')
    for idx,(cid,kind,inst,scale) in enumerate(cases):
        t=time.perf_counter()
        if kind=='homothetic':columns=homothetic_columns(inst);route_count=None
        else:columns,route_count=heterogeneous_columns(inst)
        enumeration_seconds=time.perf_counter()-t
        ref=direct_reference(columns,inst.n,inst.K,inst.P,scale,ALPHAS)
        rec=dict(id=cid,kind=kind,n=inst.n,K=inst.K,P=inst.P,L=inst.L,scale=scale,
                 enumeration_seconds=enumeration_seconds,routes_enumerated=route_count,reference=ref,runs=[])
        for alpha in ALPHAS:
            for label in ('direct','oracle_sweep'):audit_plan(inst,ref[label][str(alpha)],alpha,scale)
            for seed in (10,20,30):
                for full in (False,True):
                    started=time.perf_counter()
                    if kind=='homothetic':
                        solver=CCLNS(inst,FFF_REPORT,alpha=alpha,seed=seed,iterations=300,
                            workload_scale=scale,use_controller=False,
                            positions_per_cell=(inst.n+1 if full else 1))
                    else:
                        solver=EndpointCCLNS(inst,alpha=alpha,seed=seed,workload_scale=scale,
                            deadline=float('inf'),all_positions=full,max_iterations=300)
                    out=solver.run()
                    result=dict(workload=out['makespan'],fff=out['fff'] if 'fff' in out else FFF_REPORT(out['R']),
                                max_risk=out['max_risk'],solution=out['solution'])
                    result['J']=alpha*result['workload']/scale+(1-alpha)*result['fff']
                    audit_plan(inst,result,alpha,scale)
                    optimum=ref['direct'][str(alpha)]['J']
                    assert result['J']>=optimum-1e-8
                    rec['runs'].append(dict(result,method='CC-full' if full else 'CC-modal',alpha=alpha,
                         seed=seed,gap_pct=100*(result['J']/optimum-1),seconds=time.perf_counter()-started))
        payload['raw'].append(rec);path.write_text(json.dumps(payload,indent=1)+'\n')
        print(f'{idx+1}/{len(cases)} {cid}: {ref["evaluated_plans"]} plans, {enumeration_seconds+ref["seconds"]:.2f}s',flush=True)

if __name__=='__main__':main()
