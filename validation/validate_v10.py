"""Independent exhaustive six-task check and agreement/collector unit tests."""
import sys,json,itertools,math,importlib.util,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from exact_fff import heterogeneous_columns,direct_reference
from external_instances_v4 import load_external_records
spec=importlib.util.spec_from_file_location('v10driver',ROOT/'experiments/run_direct_fff.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

def independent_risk(a,b,c,L):
 if L>=c:return 0.
 if L<b:return 1. if L<=a else 1.-(L-a)/(2*(b-a))
 return (c-L)/(2*(c-b))

def brute(inst,scale,alphas):
 minima={a:float('inf') for a in alphas};bymax={};count=0
 for seq in itertools.permutations(range(1,inst.n+1)):
  for i in range(inst.n+1):
   for j in range(i,inst.n+1):
    routes=[seq[:i],seq[i:j],seq[j:]];risks=[];workload=0.;valid=True
    for route in routes:
     a=b=c=0.;prev=0
     for node in list(route)+([0] if route else []):
      x=inst.travel(prev,node)
      a+=x.a;b+=x.b;c+=x.c
      if node:
       x=inst.service(node);a+=x.a;b+=x.b;c+=x.c
      prev=node
     if a>inst.L+1e-9:valid=False;break
     workload+=b;risks.append(independent_risk(a,b,c,inst.L) if route else 0.)
    if not valid:continue
    count+=1;risks.sort(reverse=True);fff=sum(w*x for w,x in zip([.5,1/3,1/6],risks))
    for alpha in alphas:minima[alpha]=min(minima[alpha],alpha*workload/scale+(1-alpha)*fff)
    mx=risks[0];prev=bymax.get(mx)
    if prev is None or workload < prev[0]-1e-9 or (abs(workload-prev[0])<=1e-9 and fff<prev[1]):bymax[mx]=(workload,fff)
 frontier=[];inc=None
 for mx,rec in sorted(bymax.items()):
  if inc is None or rec[0]<inc[0]-1e-9 or (abs(rec[0]-inc[0])<=1e-9 and rec[1]<inc[1]):inc=rec;frontier.append(rec)
 sweep={a:min(a*w/scale+(1-a)*f for w,f in frontier) for a in alphas}
 return minima,sweep,count

def main():
 checks=[]
 for rec in load_external_records()[:3]:
  inst,scale=driver.restrict(rec,7);inst.n=6
  columns,_=heterogeneous_columns(inst);ref=direct_reference(columns,6,1,3,scale)
  direct,sweep,count=brute(inst,scale,[.1,.3,.7])
  for a in direct:
   assert abs(direct[a]-ref['direct'][str(a)]['J'])<1e-9
   assert abs(sweep[a]-ref['oracle_sweep'][str(a)]['J'])<1e-9
  checks.append(dict(id=rec['id'],unfiltered_feasible_labelled_plans=count,status='passed'))
 spec=importlib.util.spec_from_file_location('reliability',ROOT/'elicitation/reliability20/raccogli_e_analizza.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 import numpy as np
 est=m.AgreementEstimator(replicates=200)
 assert abs(est.icc_a1(np.array([(x,x) for x in range(1,21)],float))-1)<1e-12
 assert est.icc_a1(np.array([(x,x+5) for x in range(1,21)],float))<1
 assert math.isnan(est.icc_a1(np.array([(10,10)]*20,float)))
 for text,value in [('29 min',29.),('26,5',26.5),('circa 70',None),('non so',None),('nessuno',None),('',None),('20-25',None)]:
  assert m.numeric(text)==value,(text,m.numeric(text))
 for text in ['18/09/2026','18 09 2026','18-09-2026','18.09.2026']:assert str(m.date(text))=='2026-09-18'
 payload=dict(status='passed',independent_direct_enumeration=checks,icc_identical_and_systematic_shift_checks='passed',
   collector_parsing_checks='passed',
   scope='Direct optima and full ceiling oracle checked against unfiltered labelled tours; no human reliability result is inferred')
 (ROOT/'results/v10_validation.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
