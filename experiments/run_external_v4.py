"""Pre-specified 8-route, 3-alpha, 3-repetition wall-clock experiment.

Each method has 4 seconds INCLUDING preprocessing, model construction, seed
search and scalarized search, evaluated with the same exact fuzzy objective.
Two workers are used, each runs one method at a time; method order rotates.
OR-Tools sweep is an epsilon-constraint reference, not a direct FFF solver:
four predetermined ceilings, .25,.40,.60,.80, equal cumulative budget checkpoints; choose the
best fully served possibilistically feasible plan by the common J_alpha.
Hybrid gives half its total allowance to the same sweep, then optimizes FFF.
Three fixed seeds govern CC-LNS. OR-Tools routing uses its deterministic
defaults; its repetitions measure time-limited search variability.
Run: python3 experiments/run_external_v4.py --workers 2
The protocol is saved BEFORE jobs run; every raw route is saved.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aggregators import FFF_REPORT,max_risk
from crisp_equivalent import CredibilisticCeiling,PlanningDurationMatrix
from external_instances_v4 import ExternalMatrixInstance,load_external_records
from external_solver_v4 import EndpointCCLNS
from problem import Solution
from solver import _State

ALPHAS=(.1,.3,.7)
SEEDS=(10,20,30)
CEILINGS=(.25,.4,.6,.8)
METHODS=('OR-sweep','CC-all','CC-modal','OR+CC')
SECONDS=4.0

def verify(inst,out):
    sol=Solution(inst.K,inst.P); sol.cells=out['solution']
    assert sorted(sol.all_tasks())==list(range(1,inst.n+1))
    st=_State(inst,sol)
    assert abs(st.makespan-out['makespan'])<1e-7*max(1.,st.makespan)
    assert abs(FFF_REPORT(st.R)-out['fff'])<1e-9
    assert abs(max_risk(st.R)-out['max_risk'])<1e-9
    assert (st.violation<=1e-9)==out['feasible']
    return st

def ortools_one(inst,ceiling,deadline):
    from ortools.constraint_solver import pywrapcp,routing_enums_pb2
    start=time.perf_counter(); q=PlanningDurationMatrix(inst,CredibilisticCeiling(ceiling))
    man=pywrapcp.RoutingIndexManager(inst.n+1,inst.K*inst.P,0)
    model=pywrapcp.RoutingModel(man); scale=1000.
    def cost(i,j):
        return int(round(scale*inst.modal_travel(man.IndexToNode(i),man.IndexToNode(j))))
    def dur(i,j):
        a,b=man.IndexToNode(i),man.IndexToNode(j)
        return int(math.ceil(scale*(q.travel[a][b]+q.service[a])))
    model.SetArcCostEvaluatorOfAllVehicles(model.RegisterTransitCallback(cost))
    model.AddDimension(model.RegisterTransitCallback(dur),0,int(math.floor(scale*inst.L)),True,'duration')
    for n in range(1,inst.n+1): model.AddDisjunction([man.NodeToIndex(n)],10**9)
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(max(1,int(1000*(deadline-time.perf_counter()))))
    sol=model.SolveWithParameters(params)
    out=dict(ceiling=ceiling,elapsed_s=time.perf_counter()-start,found=False)
    if sol is None: return out
    dropped=sum(sol.Value(model.NextVar(man.NodeToIndex(n)))==man.NodeToIndex(n) for n in range(1,inst.n+1))
    if dropped: return dict(out,dropped=dropped)
    cells=[]
    for v in range(inst.K*inst.P):
        seq=[]; i=sol.Value(model.NextVar(model.Start(v)))
        while not model.IsEnd(i):
            seq.append(man.IndexToNode(i)); i=sol.Value(model.NextVar(i))
        cells.append(seq)
    s=Solution(inst.K,inst.P); s.cells=[cells[k*inst.P:(k+1)*inst.P] for k in range(inst.K)]
    st=_State(inst,s)
    out.update(found=True,solution=s.cells,makespan=st.makespan,R=st.R,
               fff=FFF_REPORT(st.R),max_risk=max_risk(st.R),feasible=st.violation<=1e-9,
               ceiling_met=st.violation<=1e-9 and max_risk(st.R)<=ceiling+1e-9)
    verify(inst,out)
    assert out['ceiling_met']
    return out

def sweep(inst,alpha,scale,deadline,start):
    records=[]; candidates=[]
    duration=deadline-start
    for i,c in enumerate(CEILINGS):
        if time.perf_counter()>=deadline: break
        out=ortools_one(inst,c,start+duration*(i+1)/len(CEILINGS))
        records.append(out)
        if out.get('found') and out['feasible']:
            out['J']=alpha*out['makespan']/scale+(1-alpha)*out['fff']
            candidates.append(out)
    return (min(candidates,key=lambda x:x['J']) if candidates else None),records

def job(args):
    idx,rec,alpha,seed=args
    results=[]
    order=METHODS[idx%4:]+METHODS[:idx%4]
    for method in order:
        start=time.perf_counter(); deadline=start+SECONDS
        inst=ExternalMatrixInstance(rec); stages=[]; seed_out=None
        if method in ('OR-sweep','OR+CC'):
            seed_end=deadline if method=='OR-sweep' else start+SECONDS/2
            seed_out,stages=sweep(inst,alpha,rec['workload_scale'],seed_end,start)
        if method=='OR-sweep':
            out=dict(seed_out) if seed_out else dict(found=False,feasible=False)
        else:
            solver=EndpointCCLNS(inst,alpha=alpha,seed=seed,workload_scale=rec['workload_scale'],
                                 deadline=deadline,all_positions=method!='CC-modal',
                                 initial_cells=seed_out['solution'] if seed_out else None)
            out=solver.run(); out['found']=True
        elapsed=time.perf_counter()-start
        if out.get('solution'):
            verify(inst,out)
            out['J']=alpha*out['makespan']/rec['workload_scale']+(1-alpha)*out['fff'] if out['feasible'] else None
        if seed_out and method=='OR+CC' and out['feasible']:
            assert out['J']<=seed_out['J']+1e-9
        # Validation and JSON serialization are outside every search allowance.
        out.update(instance=rec['id'],source_route_id=rec['source_route_id'],n=rec['n'],
                   method=method,alpha=alpha,seed=seed,budget_s=SECONDS,wall_clock_s=elapsed,
                   workload_scale=rec['workload_scale'],L=rec['L'],
                   OR_stages=stages,seed_J=seed_out['J'] if seed_out else None)
        results.append(out)
    return results

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    records=load_external_records()
    protocol=dict(created_client_date='2026-09-10',selection='8 first lexicographic route IDs; all dropoffs',
                  alphas=ALPHAS,seeds=SEEDS,ceilings=CEILINGS,methods=METHODS,
                  seconds_per_method=SECONDS,hybrid_OR_fraction=.5,workers=args.workers,
                  python=sys.version,platform=platform.platform(),cpu_count=os.cpu_count(),
                  all_position_repair='Exact lower/modal/upper directed insertion deltas; symmetric modal 2-opt disabled',
                  objective='Feasible-first, J=alpha*total_modal_workload/historical_split_workload+(1-alpha)*FFF',
                  time_includes='External instance TFN creation, OR model creation/search, OR seed generation, LNS initialization/search. Final independent verification is outside all allowances.',
                  OR_seed_note='Default deterministic Routing engine; CC-LNS random seeds 10,20,30. These are three time-limited repetitions, not three independent new operational datasets.',
                  data_sha256=hashlib.sha256((ROOT/'data/amazon_v4/instances.json').read_bytes()).hexdigest(),
                  caveats='Operational modal scaffold; constructed multi-shift slots, horizon and heterogeneous TFNs; no measured overtime frequencies or field deployment.')
    target=ROOT/'results'; target.mkdir(exist_ok=True)
    name='external_v4_smoke' if args.smoke else 'external_v4'
    (target/f'{name}_protocol.json').write_text(json.dumps(protocol,indent=2))
    jobs=[(i,r,a,s) for i,(r,a,s) in enumerate((r,a,s) for r in records for a in ALPHAS for s in SEEDS)]
    if args.smoke: jobs=jobs[:1]
    all_results=[]; started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i,out in enumerate(ex.map(job,jobs,chunksize=1),1):
            all_results.extend(out)
            (target/f'{name}.json').write_text(json.dumps(dict(protocol=protocol,raw=all_results),indent=1,allow_nan=False))
            print(f'{i}/{len(jobs)} groups; {len(all_results)} plans; {time.perf_counter()-started:.1f}s',flush=True)

if __name__=='__main__': main()
