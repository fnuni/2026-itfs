"""Tables and quoted numbers of the twenty-expert test-retest study."""
from pathlib import Path
import json, statistics as st

ROOT = Path(__file__).resolve().parents[1]; T = ROOT / 'tables'
S = ['s1', 's2', 's3', 's4']
CTX = ['s1+s3', 's1+s4', 's2+s3', 's2+s4']


def table(name, caption, label, head, rows, spec):
    text = ('\\begin{table}[tb]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n'
            '\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}' + spec + '@{}}\n\\toprule\n' + head + '\\\\\\midrule\n')
    text += '\n'.join(' & '.join(str(v) for v in row) + '\\\\' for row in rows)
    (T / name).write_text(text + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


def interval(v, dec=2):
    return f'[{v[0]:.{dec}f}, {v[1]:.{dec}f}]'


def main():
    r = json.loads((ROOT / 'results/reliability20.json').read_text())
    p = json.loads((ROOT / 'results/reliability20_planning.json').read_text())
    E = r['estimates']
    rows = []
    for s in S:
        v = E['all'][s + '_q060']
        rows.append([s.upper(), v['n'], f'{v["icc_A1"]:.3f}', interval(v['icc_bootstrap_95'], 3),
                     f'{v["mean_difference_R2_minus_R1"]:.2f}', interval(v['limits_agreement_95'])])
    table('tab_reliability20.tex', 'Test--retest agreement of $q_{060}$ (20 experts). ICC(A,1) with paired-expert '
          'bootstrap 95\\% interval; mean R2$-$R1 difference and 95\\% limits of agreement (LoA), in minutes.',
          'tab:reliability20', 'Scenario & $n$ & ICC & 95\\% interval & Diff. & LoA', rows, 'lrrlrl')
    rows = []
    for cohort, name in [('coherent', 'Coherent'), ('no_recall_or_new_information', 'No recall/info')]:
        for s in S:
            v = E[cohort][s + '_q060']
            rows.append([name, s.upper(), v['n'], f'{v["icc_A1"]:.3f}', interval(v['icc_bootstrap_95'], 3),
                         f'{v["mean_difference_R2_minus_R1"]:.2f}'])
    table('tab_reliability20_sensitivity.tex', 'Sensitivity of the $q_{060}$ agreement. Coherent: monotone complete '
          'thresholds in both sessions for that scenario; no recall/info: experts reporting neither recall of earlier '
          'numerical answers nor new information in R2.',
          'tab:reliability20-sensitivity', 'Cohort & Scenario & $n$ & ICC & 95\\% interval & Diff.', rows, 'llrrlr')
    rows = []
    for s in S:
        for e in ['b', 'q080', 'q030', 'q010']:
            v = E['all'][s + '_' + e]
            rows.append([s.upper(), ('$b$' if e == 'b' else '$q_{' + e[1:] + '}$'), v['n'], f'{v["icc_A1"]:.3f}',
                         interval(v['icc_bootstrap_95'], 3), f'{v["mean_difference_R2_minus_R1"]:.2f}',
                         interval(v['limits_agreement_95'])])
    table('tab_reliability20_secondary.tex', 'Agreement of the other elicited quantities; missing values excluded '
          'item-wise. Differences and LoA in minutes.',
          'tab:reliability20-secondary', 'Scenario & Item & $n$ & ICC & 95\\% interval & Diff. & LoA', rows, 'llrrlrl')
    rows = []
    for s in S:
        v = E['all'][s + '_q060']; m = v['multiplicative_agreement']
        rows.append([s.upper(), f'{m["geometric_ratio_R2_R1"]:.3f}', interval(m['limits_95'], 3),
                     interval(v['mean_difference_bootstrap_95']), interval(v['limits_agreement_bootstrap_95'][0]),
                     interval(v['limits_agreement_bootstrap_95'][1])])
    table('tab_reliability20_agreement.tex', '$q_{060}$ agreement on the ratio scale (geometric R2/R1 ratio and '
          'log-ratio LoA) and bootstrap 95\\% intervals of the mean difference and of each minute-scale LoA.',
          'tab:reliability20-agreement', 'Scenario & Ratio & Ratio LoA & Diff. CI & Lower LoA CI & Upper LoA CI', rows, 'lrllll')
    rows, summaries = [], {}
    for c in CTX:
        rr = [x for x in p['raw'] if x['context'] == c]
        v = dict(n=rr[0]['n'], change=st.median(x['joint_R2_change_pct'] for x in rr),
                 over=st.median(x['R1_joint_plan_R2_excess_pct'] for x in rr),
                 violated=sum(x['R1_joint_plan_R2_excess_minutes'] > 1e-7 for x in rr),
                 union=st.median(x['union_over_R1_joint_pct'] for x in rr),
                 extra=st.median(x['cartesian_over_joint_union_pct'] for x in rr))
        summaries[c] = v
        rows.append([c.upper(), v['n'], f'{v["change"]:.1f}', f'{v["over"]:.1f}', f'{v["violated"]}/10',
                     f'{v["union"]:.1f}', f'{v["extra"]:.1f}'])
    table('tab_reliability20_planning.tex', 'R1 versus R2 endpoints on ten S10 instances per context at ceiling 0.30 '
          '(certified optima). Medians (\\%): change of the minimum joint-protected shift (Change), overrun of the R1 '
          'plan under R2 (Over; Viol.: instances overrun), premium of protecting both sessions (Union) and extra '
          'Cartesian premium (Extra).',
          'tab:reliability20-planning', 'Context & $n$ & Change & Over & Viol. & Union & Extra', rows, 'lrrrrrr')
    rows = [[x['context'].upper(), x['session'], x['n'], f'{x["marginal_counts"][0]}/{x["n"]}',
             f'{x["marginal_counts"][1]}/{x["n"]}', f'{x["joint_count"]}/{x["n"]}'] for x in p['coverage']]
    table('tab_reliability20_coverage.tex', 'Marginal 90\\% $q_{060}$ thresholds per scenario and the experts '
          'jointly covered in both legs of each context.',
          'tab:reliability20-coverage', 'Context & Session & $n$ & Travel & Service & Joint', rows, 'llrrrr')
    allv = [x for x in p['raw']]
    macros = {'retest.n': 20, 'retest.forms': 40, 'retest.norecall': r['cohort_sizes']['no_recall_or_new_information'],
              'retest.recall': r['r2_recall'], 'retest.newinfo': r['r2_new_information'],
              'retest.ceilingpass': r['ceiling_checks_passed'],
              'retest.s4.loa': E['all']['s4_q060']['limits_agreement_95'][1],
              'retest.s4.loa.low': E['all']['s4_q060']['limits_agreement_95'][0],
              'retest.s3.icc.low': E['all']['s3_q060']['icc_bootstrap_95'][0],
              'retest.s3.icc.high': E['all']['s3_q060']['icc_bootstrap_95'][1],
              'retest.plan.violations': sum(x['R1_joint_plan_R2_excess_minutes'] > 1e-7 for x in allv),
              'retest.plan.union.max': max(x['union_over_R1_joint_pct'] for x in allv),
              'retest.plan.excess.max': max(x['R1_joint_plan_R2_excess_pct'] for x in allv)}
    for s in S:
        macros['retest.' + s + '.icc'] = E['all'][s + '_q060']['icc_A1']
        macros['retest.' + s + '.coherent.icc'] = E['coherent'][s + '_q060']['icc_A1']
    (T / 'numbers_reliability20.tex').write_text('% Generated from the returned forms\n' + ''.join(
        '\\expandafter\\def\\csname res@' + k + '\\endcsname{' +
        (f'{v:.3f}' if 'icc' in k else f'{v:.2f}' if isinstance(v, float) else str(v)) + '}\n' for k, v in macros.items()))
    (ROOT / 'results/reliability20_planning_summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print(json.dumps(macros, indent=1))


if __name__ == '__main__':
    main()
