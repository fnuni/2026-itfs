"""Route-level paired summaries; repeated solver seeds are not independent n."""
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean,median

ROOT=Path(__file__).resolve().parents[1]
payload=json.loads((ROOT/'results/external_v4.json').read_text()); raw=payload['raw']
assert len(raw)==288,'Run the full benchmark first'
methods=('OR-sweep','CC-all','CC-modal','OR+CC'); alphas=(.1,.3,.7)
by=defaultdict(list)
for row in raw: by[row['instance'],row['alpha'],row['method']].append(row)
summary=[]; paired=[]
for a in alphas:
    for method in methods:
        rows=[r for r in raw if r['alpha']==a and r['method']==method]
        good=[r for r in rows if r.get('J') is not None]
        summary.append(dict(alpha=a,method=method,feasible=sum(r['feasible'] for r in rows),runs=len(rows),
                            mean_J=mean(r['J'] for r in good) if good else None,
                            median_J=median(r['J'] for r in good) if good else None,
                            mean_FFF=mean(r['fff'] for r in good) if good else None,
                            median_FFF=median(r['fff'] for r in good) if good else None,
                            mean_normalized_workload=mean(r['makespan']/r['workload_scale'] for r in good) if good else None,
                            median_normalized_workload=median(r['makespan']/r['workload_scale'] for r in good) if good else None,
                            mean_max_risk=mean(r['max_risk'] for r in good) if good else None,
                            mean_wall_clock_s=mean(r['wall_clock_s'] for r in rows),
                            min_wall_clock_s=min(r['wall_clock_s'] for r in rows),
                            max_wall_clock_s=max(r['wall_clock_s'] for r in rows)))
    for method in ('CC-all','CC-modal','OR+CC'):
        differences=[]; per_route=[]
        for instance in sorted({r['instance'] for r in raw}):
            base=[r for r in by[instance,a,'OR-sweep'] if r.get('J') is not None]
            comp=[r for r in by[instance,a,method] if r.get('J') is not None]
            if len(base)!=3 or len(comp)!=3: continue
            b,c=mean(r['J'] for r in base),mean(r['J'] for r in comp)
            pct=100*(c-b)/b; differences.append(pct)
            per_route.append(dict(instance=instance,reference_J=b,method_J=c,delta_percent=pct))
        rng=random.Random(20261010)
        boot=sorted(mean(rng.choices(differences,k=len(differences))) for _ in range(10000)) if differences else []
        paired.append(dict(alpha=a,method=method,reference='OR-sweep',route_count=len(differences),
                           mean_delta_percent=mean(differences) if differences else None,
                           bootstrap_95_percentile=[boot[249],boot[9749]] if boot else None,
                           improved_routes=sum(d<-1e-9 for d in differences),
                           worsened_routes=sum(d>1e-9 for d in differences),per_route=per_route))
hybrid_seed=[]
for a in alphas:
    rows=[r for r in raw if r['alpha']==a and r['method']=='OR+CC' and r.get('seed_J') is not None and r.get('J') is not None]
    hybrid_seed.append(dict(alpha=a,seeded_runs=len(rows),
                            improved_seed_runs=sum(r['J']<r['seed_J']-1e-9 for r in rows),
                            mean_seed_improvement_percent=mean(100*(r['seed_J']-r['J'])/r['seed_J'] for r in rows) if rows else None,
                            note='Within-hybrid seed has 2 s OR allowance; this is not a comparison with the separate 4 s OR-sweep method.'))
out=dict(summary=summary,paired=paired,hybrid_seed=hybrid_seed,
         uncertainty='95% percentile bootstrap of route-level mean percentage difference; 10,000 resamples, seed 20261010. n=8 source routes; three solver repetitions are averaged within route. Separate alpha scenarios are dependent. Intervals are exploratory: common-station/common-day correlations are not modeled and the sample is not representative.',
         limitations='Small fixed public application sample; scenario TFNs and multi-shifts; short computational study. Direct scalarized global optima unknown.')
(ROOT/'results/external_v4_summary.json').write_text(json.dumps(out,indent=2))
tables=ROOT/'tables'; tables.mkdir(exist_ok=True)
lookup={(r['alpha'],r['method']):r for r in summary}
lines=[r'\begin{table}[t]',r'\centering',r'\caption{Eight complete Amazon route scaffolds (132--192 stops); median metrics over 24 runs per $\alpha$, with 4 s per method including preprocessing and seeding. OR-sweep uses four fixed ceilings; full evaluates every insertion position; modal is a restricted-position ablation.}',r'\label{tab:external-v4}',r'\footnotesize',r'\setlength{\tabcolsep}{2.7pt}',r'\begin{tabular}{clcccc}',r'\toprule',r'$\alpha$ & Method & Feas. & $J_\alpha$ & FFF & $W/W_0$ \\',r'\midrule']
for a in alphas:
    for k,m in enumerate(('OR-sweep','CC-modal','CC-all','OR+CC')):
        row=lookup[a,m]; label='CC-full' if m=='CC-all' else m
        av=f'{a:.1f}' if k==0 else ''
        lines.append(f"{av} & {label} & {row['feasible']}/{row['runs']} & {row['median_J']:.3f} & {row['median_FFF']:.3f} & {row['median_normalized_workload']:.3f}"+r' \\')
    if a!=alphas[-1]: lines.append(r'\addlinespace[2pt]')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(tables/'tab_external_v4.tex').write_text('\n'.join(lines)+'\n')
detail=[r'\begin{table*}[t]',r'\centering',r'\caption{Public-data replay: feasibility and common evaluation metrics. The normalized workload uses the same historical-split reference for all methods. These scalarized searches impose no fixed cellwise credibility ceiling: a smaller $J_\alpha$ need not reduce the maximum cell risk.}',r'\label{tab:external-v4-detail}',r'\footnotesize',r'\begin{tabular}{clcccccc}',r'\toprule',r'$\alpha$ & Method & Feasible & $J_\alpha$ & FFF & $W/W_0$ & Max risk & Mean time (s) \\',r'\midrule']
for row in summary:
    detail.append(f"{row['alpha']:.1f} & {row['method']} & {row['feasible']}/{row['runs']} & {row['mean_J']:.4f} & {row['mean_FFF']:.4f} & {row['mean_normalized_workload']:.4f} & {row['mean_max_risk']:.4f} & {row['mean_wall_clock_s']:.3f}"+r' \\')
detail += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(tables/'tab_external_v4_detail.tex').write_text('\n'.join(detail)+'\n')
macros=[]
for a in alphas:
    key={.1:'Low',.3:'Middle',.7:'High'}[a]
    pr=next(r for r in paired if r['alpha']==a and r['method']=='OR+CC')
    macros += [f"\\newcommand{{\\ExternalHybrid{key}Delta}}{{{pr['mean_delta_percent']:.2f}}}",
               f"\\newcommand{{\\ExternalHybrid{key}CILower}}{{{pr['bootstrap_95_percentile'][0]:.2f}}}",
               f"\\newcommand{{\\ExternalHybrid{key}CIUpper}}{{{pr['bootstrap_95_percentile'][1]:.2f}}}",
               f"\\newcommand{{\\ExternalHybrid{key}Wins}}{{{pr['improved_routes']}}}"]
macros += [f"\\newcommand{{\\ExternalFeasibleRuns}}{{{sum(r['feasible'] for r in raw)}}}",
           f"\\newcommand{{\\ExternalTotalRuns}}{{{len(raw)}}}"]
(tables/'numbers_external_v4.tex').write_text('\n'.join(macros)+'\n')
records=json.loads((ROOT/'data/amazon_v4/instances.json').read_text())['records']
route_lines=[r'\begin{table*}[t]',r'\centering',r'\caption{Frozen source routes and source-level mean $J_{0.3}$ across three repetitions. Directed transit asymmetry is the mean $|t_{ij}-t_{ji}|$ divided by mean $t_{ij}$ across distinct stop pairs; service CV uses published planned service totals. Shift length $L$ is a constructed scenario parameter.}',r'\label{tab:external-v4-routes}',r'\footnotesize',r'\begin{tabular}{clrrrrrrr}',r'\toprule',r'Route & Station & Stops & $L$ (min) & Asymm. & Serv. CV & OR-sweep & CC-full & OR+CC \\',r'\midrule']
for rec in records:
    t=rec['travel_minutes']; n=rec['n']; transit=[t[i][j] for i in range(n+1) for j in range(n+1) if i!=j]
    asym=mean(abs(t[i][j]-t[j][i]) for i in range(n+1) for j in range(n+1) if i!=j)/mean(transit)
    serv=rec['service_minutes'][1:]; sm=mean(serv); cv=(mean((s-sm)**2 for s in serv)**.5)/sm
    js=[mean(r['J'] for r in by[rec['id'],.3,m] if r.get('J') is not None) for m in ('OR-sweep','CC-all','OR+CC')]
    route_lines.append(f"{rec['id']} & {rec['source_station']} & {n} & {rec['L']:.1f} & {asym:.3f} & {cv:.3f} & "+' & '.join(f'{v:.4f}' for v in js)+r' \\')
route_lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(tables/'tab_external_v4_routes.tex').write_text('\n'.join(route_lines)+'\n')
(tables/'tab_external_details_v4.tex').write_text('\n'.join(detail)+'\n\n'+'\n'.join(route_lines)+'\n')
for r in summary: print(r)
for r in paired: print({k:v for k,v in r.items() if k!='per_route'})
