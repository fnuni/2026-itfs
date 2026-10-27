"""Independent standard-library audit of the first-session expert panel and recorded routes.

No model, solver, collector or fuzzy-library implementation is imported.
Optimality is established by the recorded exact solver, not recertified here.
"""
from pathlib import Path
import csv,json,statistics as st,math,hashlib

ROOT=Path(__file__).resolve().parents[1]
TOL=1e-7
errors=[];checks=0;maximum=0.
def equal(label,a,b):
    global checks,maximum
    checks+=1
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):
        err=abs(a-b);maximum=max(maximum,err)
        ok=err<=TOL
    else:ok=a==b
    if not ok:errors.append(dict(check=label,actual=a,expected=b))
def num(v):return float(v.replace(',','.')) if v else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'data/reliability20/responses_numeric.csv'
    rows=[r for r in csv.DictReader(source.open(encoding='utf-8-sig')) if r['session']=='R1']
    p=json.loads((ROOT/'results/elicited_summary.json').read_text())
    equal('source checksum',p['meta']['data_sha256'],sha(source))
    equal('unique codes',len(set(r['code'] for r in rows)),len(rows))
    equal('participants',p['summary']['n'],len(rows))
    equal('first session',p['summary']['session'],'R1')
    equal('final comprehension',sum(r['checks_passed']=='1' for r in rows),p['summary']['checks']['both_passed'])
    equal('ceiling item',sum(r['ceiling_check_passed']=='1' for r in rows),p['summary']['checks']['ceiling_item'])
    scenarios={}
    keys=('a','b','c','q080','q060','q030','q010')
    for s in ('s1','s2','s3','s4'):
        vals=[{k:num(r[s+'_'+k]) for k in keys} for r in rows]
        def ok(v):
            chain=[v[k] for k in ('a','b','q080','q060','q030','q010','c') if v[k] is not None]
            return v['b'] is not None and v['q060'] is not None and v['b']>0 and chain==sorted(chain) and min(chain)>=0
        good=[v for v in vals if ok(v)]
        equal(s+' coherent direct',len(good),p['summary']['scenarios'][s]['n_direct'])
        q=[v['q060'] for v in good]
        tri=[v for v in good if v['a'] is not None and v['c'] is not None]
        qt=[v['b']+.4*(v['c']-v['b']) for v in tri]
        v=p['summary']['scenarios'][s]
        equal(s+' triangles',len(tri),v['n_triangle'])
        equal(s+' unbounded c',sum(r[s+'_c_unbounded']=='1' for r in rows),v['unbounded_c'])
        equal(s+' median q',st.median(q),v['q']['median'])
        equal(s+' maximum q',max(q),v['q']['maximum'])
        equal(s+' triangular maximum',max(qt),v['q_tri']['maximum'])
        equal(s+' absdelta',st.median(abs(x['q060']-y) for x,y in zip(tri,qt)),v['abs_delta']['median'])
        for k,level in (('q080',.8),('q030',.3),('q010',.1)):
            res=[(x[k]-x['b'])-(1-level)/.4*(x['q060']-x['b']) for x in good if x[k] is not None]
            equal(s+' residuals '+k,len(res),len(v['residual_values'][k]))
            equal(s+' residual median '+k,st.median(res),v['residual'][k]['median'])
        ref=st.median(x['b'] for x in good)
        med=tuple(st.median(x[k] for x in tri) for k in ('a','b','c'))
        scenarios[s]=dict(reference=ref,consensus_direct=st.median(q)/ref,
                          consensus_triangular=(med[1]+.4*(med[2]-med[1]))/ref,
                          envelope_direct=max(q)/ref,envelope_pass=max(q)/ref,
                          envelope_triangular=max(qt)/ref)
        for basis in ('consensus_direct','consensus_triangular','envelope_direct','envelope_triangular'):
            leg='travel' if s in ('s1','s2') else 'service'
            ctx=next(c for c in p['summary']['contexts'] if c.lower().split('+')[0 if leg=='travel' else 1]==s)
            equal(s+' '+basis,scenarios[s][basis],p['summary']['contexts'][ctx][basis][leg])
    source_cells=0
    # Direct Euclidean travel and per-task service, independent of Instance.
    families=json.loads((ROOT/'data/instances_small.json').read_text())['families']
    def quantities(rec,solution):
        fam=families[rec['family']];inst=next(i for i in fam['instances'] if i['seed']==rec['seed'])
        xy={0:inst['depot']};sv={0:0.}
        for task in inst['tasks']:xy[task['id']]=[task['x'],task['y']];sv[task['id']]=task['service_modal']
        flat=[i for row in solution for cell in row for i in cell]
        equal('coverage '+str((rec['family'],rec['seed'],rec['context'])),sorted(flat),list(range(1,fam['n']+1)))
        equal('vehicle shape',len(solution),fam['K']);equal('shift shape',all(len(r)==fam['P'] for r in solution),True)
        out=[]
        for row in solution:
            for seq in row:
                path=[0]+seq+[0]
                travel=sum(math.hypot(xy[i][0]-xy[j][0],xy[i][1]-xy[j][1])/inst['speed'] for i,j in zip(path,path[1:]))
                out.append((travel,sum(sv[i] for i in seq)))
        return out
    def durations(rec,q,basis):
        ts,ss=rec['context'].lower().split('+')
        return [T*scenarios[ts][basis]+S*scenarios[ss][basis] for T,S in q]
    small=json.loads((ROOT/'results/elicited_small.json').read_text())
    ext=json.loads((ROOT/'results/elicited_extension.json').read_text())
    equal('small source',small['meta']['data_sha256'],sha(source));equal('extension source',ext['meta']['data_sha256'],sha(source))
    plans=0
    for r in small['raw']:
        for basis in ('consensus_direct','consensus_triangular','envelope_direct','envelope_pass','envelope_triangular'):
            o=r[basis]
            tests=[(o,r['limit'])] if o['attainable'] else []
            tests.append((o['at_min_shift'],o['min_shift_ratio']*r['limit']))
            for result,L in tests:
                q=quantities(r,result['solution']);plans+=1
                equal('modal workload',sum(t+s for t,s in q),result['workload'])
                equal('own planning feasibility',max(durations(r,q,basis))<=L+1e-6,True)
                for env in ('envelope_direct','envelope_pass','consensus_direct'):
                    equal('saved overrun',max(durations(r,q,env))/L-1.,result['overrun_'+env])
    for r in ext['raw']:
        q=quantities(r,r['paired_triangular_solution']);plans+=1
        L=r['min_shift']['envelope_paired_triangular']*r['limit']
        equal('paired triangle feasibility',max(durations(r,q,'envelope_triangular'))<=L+1e-6,True)
        equal('paired direct overrun',max(durations(r,q,'envelope_direct'))/L-1,r['paired_triangular_plan']['overrun_envelope_direct'])
    report=dict(status='passed' if not errors else 'failed',independent_of_analysis_collector_and_model=True,
                survey_records=len(rows),session='R1',recorded_plans_checked=plans,
                checks=checks,max_absolute_numeric_difference=maximum,failures=len(errors),errors=errors,
                data_sha256=sha(source),scope='Independent transcription, descriptive arithmetic, coverage, workload and planning-duration feasibility; exact optimality is not independently recertified')
    (ROOT/'results/elicited_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='errors'},indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
