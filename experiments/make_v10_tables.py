import json,statistics as st,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tables';R=ROOT/'results'
def table(name,caption,label,head,rows,spec):
 text='\\begin{table}[tb]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}'+spec+'@{}}\n\\toprule\n'+head+'\\\\\\midrule\n'
 text+='\n'.join(' & '.join(str(v) for v in row)+'\\\\' for row in rows)
 text+='\n\\bottomrule\n\\end{tabular}\n\\end{table}\n';(T/name).write_text(text)

def minimax_summary(path,prefix,macros):
 d=json.loads(path.read_text());allr=[(rec,a,x) for c in d['raw'] for rec in c['records'] for a,x in rec['alphas'].items()]
 gaps=[x['sweep_gap_pct'] for _,_,x in allr];lost=[x for _,_,x in allr if x['sweep_gap_pct']>1e-6]
 macros[prefix+'cases']=len(allr);macros[prefix+'lost']=len(lost);macros[prefix+'gap.max']=f'{max(gaps):.1f}'
 macros[prefix+'gap.meanlost']=f"{st.mean(x['sweep_gap_pct'] for x in lost):.1f}" if lost else '0.0'
 macros[prefix+'ratio.max']=f"{max((x['sweep_gap']/x['bound'] if x['bound']>0 else 0.) for _,_,x in allr):.2f}"
 macros[prefix+'instances']=sum(len(c['records']) for c in d['raw'])
 macros[prefix+'over']=sum(x['counts'][-1]>0 for _,_,x in allr)
 for rec,a,x in allr:
  if 'concentrates' not in x:
   feas=[v for v in rec['sweep'].values() if v['status']=='Optimal']
   sp=min(feas,key=lambda v:float(a)*v['workload']/rec['scale']+(1-float(a))*v['fff_upper'])
   x['concentrates']=bool(x['max_upper']>=sp['max_upper']-1e-12 and x['counts'][0]<=sp['counts'][0])
   x['sweep_gap']=x['sweep_gap'];x['workload_saving_pct']=100*(x['workload']/sp['workload']-1)
 if lost:
  macros[prefix+'conc']=sum(x['concentrates'] for x in lost)
  macros[prefix+'abs.max']=f"{max(x['sweep_gap'] for x in lost):.3f}"
  macros[prefix+'extra.med']=f"{st.median(x['workload_saving_pct'] for x in lost):.1f}"
 return d,allr

def minimax_table(macros):
 path=R/'endpoint_minimax.json'
 if not path.exists():return
 d,allr=minimax_summary(path,'mm.',macros)
 macros['mm.enum']=sum('enumeration_J' in x for _,_,x in allr)
 macros['mm.secmax']=f"{max(x['seconds'] for _,_,x in allr):.0f}"
 macros['mm.tiebreak.cut']=sum(v.get('tiebreak') not in (None,'optimal') for c in d['raw'] for r in c['records'] for v in r['sweep'].values())
 opt=R/'endpoint_minimax_optimistic.json'
 if opt.exists():minimax_summary(opt,'mmo.',macros)
 rows=[];fams=d['protocol']['families']
 for fam in fams:
  rr=[x for rec,_,x in allr if rec['family']==fam]
  if not rr:continue
  lost=[x for x in rr if x['sweep_gap_pct']>1e-6]
  ratio=max((x['sweep_gap']/x['bound'] if x['bound']>0 else 0.) for x in rr)
  f=fams[fam]
  rows.append([fam,f['n'],f['K']*f['P'],f'{len(lost)}/{len(rr)}',f"{st.mean(x['sweep_gap_pct'] for x in rr):.2f}",f"{max(x['sweep_gap_pct'] for x in rr):.1f}",
   f"{max(x['sweep_gap'] for x in rr):.3f}",f'{ratio:.2f}',f"{sum(x['concentrates'] for x in lost)}",f"{st.median(x['seconds'] for x in rr):.1f}"])
 table('tab_minimax.tex','Exact minimax optima (least-favourable completion, R1 median endpoints, routes with modal duration at most $L$) against the exact sweep over all grid ceilings, over two contexts, two horizons and three $\\alpha$: sweep losses, mean and maximum relative gap of the sweep (\\%), maximum absolute gap in $J^+$, largest gap as a fraction of the bound of Theorem~\\ref{thm:sweep}, losses where the minimax plan keeps a worst-cell risk at least as large on no more exposed cells (Conc.), median MILP time (s).','tab:minimax','Family & $n$ & $n_c$ & Lost & Mean & Max. & Abs. & Gap/bd. & Conc. & s',rows,'lrrrrrrrrr')

def main():
 direct=json.loads((R/'direct_fff.json').read_text());rows=[];summary={}
 for kind in ['homothetic','heterogeneous']:
  cases=[x for x in direct['raw'] if x['kind']==kind];summary[kind]={}
  for method in ['oracle_sweep','CC-modal','CC-full']:
   if method=='oracle_sweep':gaps=[100*(r['reference'][method][str(a)]['J']/r['reference']['direct'][str(a)]['J']-1) for r in cases for a in [.1,.3,.7]]
   else:gaps=[x['gap_pct'] for r in cases for x in r['runs'] if x['method']==method]
   values=dict(n=len(gaps),mean=st.mean(gaps),median=st.median(gaps),maximum=max(gaps),optima=sum(x<=1e-6 for x in gaps))
   summary[kind][method]=values
   rows.append(['Scaled (30)' if kind=='homothetic' else 'Directed (8)', 'Exact sweep' if method=='oracle_sweep' else method,
    f'{values["mean"]:.2f}',f'{values["median"]:.2f}',f'{values["maximum"]:.2f}',f'{values["optima"]}/{values["n"]}'])
 table('tab_direct_fff.tex','Direct $J_\\alpha$ optima on 30 canonical scaled-copy instances and eight seven-stop directed Amazon restrictions. Relative gaps (\\%) over three $\\alpha$. The exact sweep solves every maximum-risk breakpoint optimally (90/24 comparisons); each direct-search rule uses 300 iterations and three seeds (270/72).','tab:direct-fff','Instances & Reference & Mean & Median & Max. & Opt. ',rows,'llrrrr')
 joint=json.loads((R/'joint_protection.json').read_text())['raw'];rows=[]
 for context in ['s1+s3','s1+s4','s2+s3','s2+s4']:
  rr=[x for x in joint if x['context']==context]
  rows.append([context.upper(),f'{st.median(x["joint_price_pct"] for x in rr):.1f}',f'{st.median(x["cartesian_price_pct"] for x in rr):.1f}',f'{st.median(x["cartesian_over_joint_pct"] for x in rr):.1f}'])
 table('tab_joint_v10.tex','R1 answers of the twenty experts on ten S10 instances per context at ceiling 0.30: certified minimum-shift premiums over consensus when each expert is protected jointly or all leg-wise combinations are protected (Cartesian); last column: extra Cartesian premium.','tab:joint-v10','Context & Joint (\\%) & Cartesian (\\%) & Extra (\\%)',rows,'lrrr')
 ident=json.loads((R/'endpoint_identification.json').read_text())['raw'];rows=[]
 for alpha in [.1,.3,.7]:
  rr=[x for x in ident if x['alpha']==alpha]
  rows.append([f'{alpha:.1f}',len(rr),f'{st.median(x["regret"] for x in rr):.4f}',f'{max(x["regret"] for x in rr):.4f}',f'{max(x["uniform_bound"] for x in rr):.3f}'])
 table('tab_endpoint_v10.tex','Regret of the upper-objective plan under three constructed completions of the R1 median endpoints (ceilings $0,0.05,0.15,0.30,0.40$, sentinel $0.5$) on 40 S10 context cases; last column: bound of Corollary~\\ref{cor:regret}.','tab:endpoint-v10','$\\alpha$ & Cases & Median regret & Max. regret & Bound',rows,'rrrrr')
 source=ROOT/'data/reliability20/responses_numeric.csv'
 panel=[r for r in csv.DictReader(source.open()) if r['session']=='R1'];n=len(panel);rows=[]
 for rank in [1,2,3,4,5]:
  th={s:sorted([float(r[s+'_q060']) for r in panel],reverse=True)[rank-1] for s in ['s1','s2','s3','s4']}
  counts=[]
  for ts,ss in [('s1','s3'),('s1','s4'),('s2','s3'),('s2','s4')]:counts.append(sum(float(r[ts+'_q060'])<=th[ts] and float(r[ss+'_q060'])<=th[ss] for r in panel))
  rows.append([rank,f'{n+1-rank}/{n}']+[f'{c}/{n}' for c in counts])
 table('tab_coverage_v10.tex',f'Planning every leg at the $k$-th largest R1 answer of its scenario: guaranteed marginal share and experts covered in both legs of each context ({n} experts).','tab:coverage-v10','$k$ & Marginal min. & S1+S3 & S1+S4 & S2+S3 & S2+S4',rows,'rrrrrr')
 source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
 summary['panel_sha256']=source_sha;summary['endpoint_cases']=len(ident);summary['joint_cases']=len(joint)
 summary['human_data']='40 returned forms, 20 paired experts; R1 enters planning studies, R1/R2 agreement in reliability20.json'
 (R/'v10_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 macros={'vten.oracle.scaled.max':summary['homothetic']['oracle_sweep']['maximum'],
  'vten.oracle.directed.max':summary['heterogeneous']['oracle_sweep']['maximum'],
  'vten.oracle.lost':sum(100*(r['reference']['oracle_sweep'][str(a)]['J']/r['reference']['direct'][str(a)]['J']-1)>1e-6 for r in direct['raw'] for a in [.1,.3,.7]),
  'vten.direct.cases':len(direct['raw']),'vten.direct.pairs':3*len(direct['raw'])}
 jp=json.loads((R/'joint_protection.json').read_text())['raw']
 ctxmed=[st.median(x['cartesian_over_joint_pct'] for x in jp if x['context']==c) for c in ['s1+s3','s1+s4','s2+s3','s2+s4']]
 macros['vten.joint.extra.min']=f'{min(ctxmed):.1f}';macros['vten.joint.extra.max']=f'{max(ctxmed):.1f}'
 jm=[st.median(x['joint_price_pct'] for x in jp if x['context']==c) for c in ['s1+s3','s1+s4','s2+s3','s2+s4']]
 macros['vten.joint.price.min']=f'{min(jm):.1f}';macros['vten.joint.price.max']=f'{max(jm):.1f}'
 for a,tag in [(.1,'a'),(.3,'b'),(.7,'c')]:
  rr=[x for x in ident if x['alpha']==a]
  macros['vten.regret.max.'+tag]=f"{max(x['regret'] for x in rr):.4f}";macros['vten.regret.bound.'+tag]=f"{max(x['uniform_bound'] for x in rr):.3f}"
 macros['vten.regret.width']=f"{max(x['max_cell_width'] for x in ident):.2f}"
 macros['vten.endpoint.records']=len(ident)
 minimax_table(macros)
 (T/'numbers_v10.tex').write_text('% Generated from the direct-objective and endpoint records\n'+''.join('\\expandafter\\def\\csname res@'+k+'\\endcsname{'+(f'{v:.2f}' if isinstance(v,float) else str(v))+'}\n' for k,v in macros.items()))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
