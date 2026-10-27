"""Endpoint-only certificates and joint-vs-Cartesian protection on the R1 expert panel.

Uses the first-session (R1) answers of the twenty-expert campaign. The elicited
grid is the four direct thresholds (ceilings 0.05, 0.15, 0.30, 0.40), the
upper support bound c (ceiling 0) and the representative duration b (sentinel
1/2). All membership completions below are labelled mathematical sensitivity
constructions, not new human data. Test-retest agreement is analyzed
separately in run_reliability20.py.
"""
import sys,json,csv,statistics as st,math,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from instances_io import CanonicalInstances
from exact_small import SubsetTSP
from exact_fff import RouteColumn,partitions,direct_reference,cell_owa
from endpoint_bounds import risk_interval
from solver import RALNS,_State
from aggregators import FFF_REPORT
from elicitation import ElicitationPanel

LEVELS=(0.,.05,.15,.30,.40,.50)
KNOTS=('c','q010','qprime','q','q080','b')
def completion_endpoint(r,knots,power):
 if r>=.5:
  return knots[-1]-(2*r-1)*(knots[-1]-.7)
 for i in range(len(LEVELS)-1):
  if r<=LEVELS[i+1]:
   x=(r-LEVELS[i])/(LEVELS[i+1]-LEVELS[i])
   return knots[i]+(knots[i+1]-knots[i])*x**power
 return knots[-1]

def completion_risk(T,V,L,kt,ks,p):
 # All candidate cells in this study have representative duration <= L.
 if T*kt[0]+V*ks[0]<=L:return 0.
 lo,hi=0.,.5
 for _ in range(60):
  mid=(lo+hi)/2
  if T*completion_endpoint(mid,kt,p)+V*completion_endpoint(mid,ks,p)<=L:hi=mid
  else:lo=mid
 return hi

def main():
 panel=ElicitationPanel.from_files()
 def chain(j):
  # complete coherent threshold chain; an unbounded upper support counts as +inf
  vals=[getattr(j,k) for k in KNOTS[1:]]
  if None in vals or not j.coherent:return None
  return [math.inf if j.c is None else j.c]+vals
 scenarios={};used={}
 for s in ['s1','s2','s3','s4']:
  rows=[x for x in (chain(j) for j in panel.judgments(s)) if x is not None]
  med=[float(np.median(np.array([r[i] for r in rows]))) for i in range(len(KNOTS))]
  assert all(math.isfinite(v) for v in med) and med==sorted(med,reverse=True)
  scenarios[s]=dict(zip(KNOTS,med));used[s]=len(rows)
 data=[dict(code=r.rid,**{s+'_q060':r.judgments[s].q for s in ['s1','s2','s3','s4']}) for r in panel.respondents]
 ci=CanonicalInstances();ident=[];joint=[]
 for seed in [10,20,30,40,50,60,70,80,90,100]:
  inst=ci.build('S10',seed);tsp=SubsetTSP(np.array([[inst.modal_travel(i,j) for j in range(11)] for i in range(11)]))
  T=tsp.best;V=np.array([sum(inst.service_modal[i+1] for i in range(10) if m>>i&1) for m in range(1<<10)])
  groups=list(partitions((1<<10)-1,2))
  modal_min=min(max(T[m]+V[m] for m in g) for g in groups)
  L=1.25*modal_min
  scale=_State(inst,RALNS(inst,FFF_REPORT,seed=0).initial()).makespan
  for ts,ss in [('s1','s3'),('s1','s4'),('s2','s3'),('s2','s4')]:
   context=ts+'+'+ss
   kt=[scenarios[ts][k]/scenarios[ts]['b'] for k in KNOTS]
   ks=[scenarios[ss][k]/scenarios[ss]['b'] for k in KNOTS]
   upp={};low={};intervals={}
   for mask in range(1,1<<10):
    if T[mask]+V[mask]>L:continue
    q=[(r,float(T[mask]*a+V[mask]*b)) for r,a,b in zip(LEVELS,kt,ks)]
    lo,hi=risk_interval(q,L);intervals[mask]=(lo,hi)
    args=(mask,tuple(tsp.tour(mask)),float(T[mask]+V[mask]))
    upp[mask]=[RouteColumn(*args,hi,.7*args[2])]
    low[mask]=[RouteColumn(*args,lo,.7*args[2])]
   upper=direct_reference(upp,10,1,2,scale)
   lower=direct_reference(low,10,1,2,scale)
   delta=max(hi-lo for lo,hi in intervals.values())
   for power in [.5,1.,2.]:
    truth={mask:[RouteColumn(mask,col[0].sequence,col[0].workload,
         completion_risk(T[mask],V[mask],L,kt,ks,power),col[0].optimistic)] for mask,col in upp.items()}
    for m,col in truth.items():
     assert intervals[m][0]-1e-9<=col[0].risk<=intervals[m][1]+1e-9
    opt=direct_reference(truth,10,1,2,scale)
    for alpha in [.1,.3,.7]:
     chosen=upper['direct'][str(alpha)];masks=[sum(1<<(i-1) for i in c) for row in chosen['solution'] for c in row if c]
     actual=alpha*chosen['workload']/scale+(1-alpha)*cell_owa([truth[m][0].risk for m in masks],2)
     gap=actual-opt['direct'][str(alpha)]['J'];assert gap>=-1e-9 and gap<=(1-alpha)*delta+1e-9
     ident.append(dict(seed=seed,context=context,power=power,alpha=alpha,L=L,grid=list(LEVELS),max_cell_width=delta,
       optimum=opt['direct'][str(alpha)]['J'],upper_plan_actual_J=actual,regret=gap,
       uniform_bound=(1-alpha)*delta,lower_optimum=lower['direct'][str(alpha)]['J'],
       upper_optimum=chosen['J'],upper_solution=chosen['solution']))
   # Joint scenario of each respondent, with the same full-panel reference.
   ft=[float(r[ts+'_q060'])/scenarios[ts]['b'] for r in data]
   fs=[float(r[ss+'_q060'])/scenarios[ss]['b'] for r in data]
   cons_t,cons_s=st.median(ft),st.median(fs)
   def score(g,rule):
    if rule=='consensus':return max(T[m]*cons_t+V[m]*cons_s for m in g)
    if rule=='cartesian':return max(T[m]*max(ft)+V[m]*max(fs) for m in g)
    return max(T[m]*a+V[m]*b for m in g for a,b in zip(ft,fs))
   minima={rule:min(score(g,rule) for g in groups) for rule in ['consensus','joint','cartesian']}
   assert minima['consensus']<=minima['joint']+1e-7 and minima['joint']<=minima['cartesian']+1e-7
   joint.append(dict(seed=seed,context=context,min_shifts={k:float(v) for k,v in minima.items()},
     joint_price_pct=100*(minima['joint']/minima['consensus']-1),
     cartesian_price_pct=100*(minima['cartesian']/minima['consensus']-1),
     cartesian_over_joint_pct=100*(minima['cartesian']/minima['joint']-1)))
  print('S10',seed,'endpoint and joint protection certified',flush=True)
 (ROOT/'results/endpoint_identification.json').write_text(json.dumps(dict(protocol=dict(
   source='twenty-expert campaign, first session (R1)',knots=dict(levels=list(LEVELS),quantities=list(KNOTS),
   scenario_medians=scenarios,respondents_with_complete_chain=used,unbounded_c='counted as larger than every finite bound'),study='constructed compatible completions; not measured membership truth',
   cases=40,alphas=[.1,.3,.7],powers=[.5,1.,2.],modal_feasible_routes=True,
   horizon='1.25 times certified minimum modal shift',completion_left='common nonnegative linear convention; not used for right-branch risks'),raw=ident),indent=1)+'\n')
 (ROOT/'results/joint_protection.json').write_text(json.dumps(dict(protocol=dict(source='twenty-expert campaign, first session (R1)',
   cases=40,family='S10',ceiling=.30,exact='All shortest-subset-tour two-route partitions',
   reference='median representative duration of the complete-chain respondents, shared by joint and Cartesian rules'),raw=joint),indent=1)+'\n')
if __name__=='__main__':main()
