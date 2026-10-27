"""Independent standard-library arithmetic audit of the R1 expert median-spread replay."""
from pathlib import Path
import json,csv,statistics as st,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    panel_source=ROOT/'data/reliability20/responses_numeric.csv'
    rows=[r for r in csv.DictReader(panel_source.open()) if r['session']=='R1']
    def tri(s):
        out=[]
        for r in rows:
            v=[r[s+'_'+k] for k in ('a','b','q080','q060','q030','q010','c')]
            if '' in (v[0],v[1],v[-1]):continue
            chain=[float(x) for x in v if x!='']
            if chain==sorted(chain):out.append((float(v[0]),float(v[1]),float(v[-1])))
        return out
    spreads={s:(st.median(1-a/b for a,b,c in tri(s)),st.median(c/b-1 for a,b,c in tri(s))) for s in ('s1','s3')}
    source=json.loads((ROOT/'data/amazon_v4/instances.json').read_text())['records']
    instances={r['id']:r for r in source}
    payload=json.loads((ROOT/'results/external_extended_shapes.json').read_text())
    run=[r for r in payload['raw'] if r['scenario']=='elicited']
    errors=[];checks=0;maximum=0.;plans=0
    def equal(k,a,b):
        nonlocal checks,maximum
        checks+=1
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            maximum=max(maximum,abs(a-b));ok=abs(a-b)<=1e-7
        else:ok=a==b
        if not ok:errors.append(dict(check=k,actual=a,expected=b))
    for x in run:
        r=instances[x['instance']]
        cells=x.get('solution')
        if cells is None:
            equal('unfound run has no objective',x['J'],None);continue
        plans+=1
        equal('coverage',sorted(i for row in cells for seq in row for i in seq),list(range(1,r['n']+1)))
        equal('vehicle shape',len(cells),r['K']);equal('shift shape',all(len(row)==r['P'] for row in cells),True)
        riskrows=[];workload=0.;feasible=True
        for row in cells:
            risks=[]
            for seq in row:
                path=[0]+seq+[0]
                T=sum(r['travel_minutes'][i][j] for i,j in zip(path,path[1:]))
                S=sum(r['service_minutes'][i] for i in seq)
                a=T*(1-spreads['s1'][0])+S*(1-spreads['s3'][0])
                b=T+S;c=T*(1+spreads['s1'][1])+S*(1+spreads['s3'][1]);L=r['L']
                pos=0. if L<a else 1. if L>=b else (L-a)/(b-a)
                complement=0. if L>=c else 1. if L<b else (c-L)/(c-b)
                risks.append(1.-.5*(pos+1.-complement));workload+=b
                feasible=feasible and a<=L+1e-7
            riskrows.append(risks)
        flat=sorted((v for row in riskrows for v in row),reverse=True);n=len(flat)
        w=[2*(n-i)/(n*(n+1)) for i in range(n)]
        fff=sum(k*v for k,v in zip(w,flat))
        equal('workload',workload,x['makespan']);equal('FFF',fff,x['fff'])
        equal('maximum credibility',max(v for row in riskrows for v in row),x['max_risk'])
        equal('optimistic feasibility',feasible,x['feasible'])
        equal('objective',x['alpha']*workload/r['workload_scale']+(1-x['alpha'])*fff,x['J'])
    protocol=payload['protocol'].get('executions',{}).get('elicited',payload['protocol'])
    digest=hashlib.sha256(panel_source.read_bytes()).hexdigest()
    equal('survey provenance',protocol['elicitation']['data_sha256'],digest)
    equal('run count',len(run),144)
    report=dict(status='passed' if not errors else 'failed',independent_of_model_solver_and_fuzzy_implementation=True,
                method_runs=len(run),returned_plans_checked=plans,checks=checks,failures=len(errors),errors=errors,
                max_absolute_numeric_difference=maximum,survey_sha256=digest,
                result_sha256=hashlib.sha256((ROOT/'results/external_extended_shapes.json').read_bytes()).hexdigest(),
                scope='Recorded final routes and declared memberships; not field calibration or an optimality certificate')
    (ROOT/'results/external_elicited_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='errors'},indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
