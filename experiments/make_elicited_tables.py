"""Survey tables and quoted numbers from the first-session (R1) expert panel.

python3 experiments/make_elicited_tables.py
"""
from pathlib import Path
import json, statistics as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'; TAB = ROOT / 'tables'
CTX = ('S1+S3', 'S1+S4', 'S2+S3', 'S2+S4')
HEADER = '% Generated from the R1 session of the twenty-expert campaign; do not edit.\n'
# Rounding each answer by at most 0.5 min bounds the straight-branch residual
# (x - b) - k (q060 - b) by 0.5 (1 + |1 - k| + k): k = 0.5, 1.75, 2.25.
TOL = {'q080': 1.0, 'q030': 1.75, 'q010': 2.25}
NUM = {}


def fmt(x, d=1):
    s = f'{x:.{d}f}'
    return '0.' + '0' * d if float(s) == 0 and d else s


def put(k, x, d=1):
    NUM[k] = fmt(x, d) if isinstance(x, float) else str(x)


def table(name, caption, label, cols, heads, rows):
    (TAB / name).write_text(
        HEADER + '\\begin{table}[tb]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n'
        '\\scriptsize\\setlength{\\tabcolsep}{2.4pt}\n\\begin{tabular}{@{}' + cols + '@{}}\n\\toprule\n'
        + heads + '\\\\\\midrule\n' + '\n'.join(' & '.join(str(v) for v in row) + '\\\\' for row in rows)
        + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


class SurveyTables:
    def __init__(self):
        self.summary = json.loads((RES / 'elicited_summary.json').read_text())
        self.p = self.summary['summary']
        self.small = json.loads((RES / 'elicited_small.json').read_text())
        self.ext = json.loads((RES / 'elicited_extension.json').read_text())
        sha = self.summary['meta']['data_sha256']
        assert self.small['meta']['data_sha256'] == self.ext['meta']['data_sha256'] == sha
        assert self.p['session'] == 'R1'

    def descriptive(self):
        p, sc = self.p, self.p['scenarios']
        n = p['n']
        put('survey.n', n); put('survey.answers', 4 * n)
        put('survey.minutes', p['minutes']['median']); put('survey.practicality', p['practicality']['median'], 0)
        put('survey.both', p['checks']['both_passed']); put('survey.ceiling', p['checks']['ceiling_item'])
        put('survey.flagged', sum(v['n_flagged'] for v in sc.values()))
        put('survey.unbounded', sum(v['unbounded_c'] for v in sc.values()))
        put('survey.unbounded.s4', sc['s4']['unbounded_c'])
        put('survey.missing.a', sum(v['missing']['a'] for v in sc.values()))
        put('survey.triangles', sum(v['n_triangle'] for v in sc.values()))
        put('survey.qwithoutc', sum(v['q_without_c'] for v in sc.values()))
        for k, tol in TOL.items():
            vals = [e for v in sc.values() for e in v['residual_values'][k]]
            put(f'survey.res.{k}.n', len(vals))
            put(f'survey.res.{k}.beyond', sum(abs(e) > tol + 1e-9 for e in vals))
            put(f'survey.res.{k}.neg', sum(e < -tol - 1e-9 for e in vals))
            put(f'survey.res.{k}.pos', sum(e > tol + 1e-9 for e in vals))
            put(f'survey.res.{k}.tol', fmt(tol, 2))
        put('survey.res.q030.beyond.s4', sum(abs(e) > TOL['q030'] + 1e-9 for e in sc['s4']['residual_values']['q030']))
        put('survey.own.min', min(v['own_multiplier']['median'] for v in sc.values()), 2)
        put('survey.own.max', max(v['own_multiplier']['median'] for v in sc.values()), 2)
        for s, v in sc.items():
            for key, x in [('absdelta', v['abs_delta']['median']), ('qmax', v['q']['maximum']),
                           ('qtrimax', v['q_tri']['maximum']), ('ntri', v['n_triangle']),
                           ('left', 100 * v['left_spread']['median']), ('right', 100 * v['right_spread']['median'])]:
                put(f'survey.{s}.{key}', x)
            put(f'survey.{s}.binding', ', '.join(v['binding_direct']))
        labels = {'s1': 'S1 urban', 's2': 'S2 extra-urban', 's3': 'S3 access ready', 's4': 'S4 queueing'}
        rows = []
        for s, v in sc.items():
            q, th = v['q'], v['thresholds']
            rows.append([labels[s], fmt(v['reference_mode']), fmt(th['q080']['median']),
                         f"{fmt(q['median'])} ({fmt(q['q1'])}--{fmt(q['q3'])})",
                         fmt(th['q030']['median']), fmt(th['q010']['median']),
                         f"{fmt(v['abs_delta']['median'])} ({v['n_triangle']})",
                         f"{fmt(q['maximum'])}/{fmt(v['q_tri']['maximum'])}"])
        table('tab_r1.tex',
              f'First-session (R1) answers of the {n} experts, all passing both final checks; durations in minutes. '
              r'$\tilde b$: median representative duration; $q_{p}$: direct threshold at exceedance possibility $p$ '
              r'(credibility ceiling $p/2$ on the right branch). $|\Delta|=|q_{060}-(b+0.4(c-b))|$, median over the '
              r'complete finite triangles (count); last column: largest direct/triangular $q_{060}$.',
              'tab:r1', 'lccccccc',
              r'Scenario & $\tilde b$ & $q_{080}$ & $q_{060}$ med. (IQR) & $q_{030}$ & $q_{010}$ & $|\Delta|$ (n) & Max. dir./tri.',
              rows)
        rows = [['Both final checks (keys B, A)', f"{p['checks']['both_passed']}/{n}"],
                ['Ceiling item (key no)', f"{p['checks']['ceiling_item']}/{n}"],
                ['Scenario rotations (forms each)', '5/5/5/5'],
                ['Unknown lower supports $a$ (judgments)', f"{sum(v['missing']['a'] for v in sc.values())}/{4 * n}"],
                ['No finite upper support $c$ (judgments)', f"{sum(v['unbounded_c'] for v in sc.values())}/{4 * n}"],
                ['Questionnaire practicality (1--5)',
                 f"{fmt(p['practicality']['median'], 0)} ({fmt(p['practicality']['minimum'], 0)}--{fmt(p['practicality']['maximum'], 0)})"],
                ['Self-reported completion, minutes',
                 f"{fmt(p['minutes']['median'])} ({fmt(p['minutes']['minimum'], 0)}--{fmt(p['minutes']['maximum'], 0)})"]]
        table('tab_r1_items.tex', 'First-session comprehension, support answers and burden; medians (ranges) for practicality and completion time.',
              'tab:r1-items', r'p{0.70\columnwidth}l', 'Item & Observed result', rows)
        rows = []
        for s, v in sc.items():
            res = v['residual_values']
            rows.append([s.upper(), f"{v['n_triangle']}/{n}",
                         f"{fmt(v['left_spread']['median'], 2)}/{fmt(v['right_spread']['median'], 2)}",
                         fmt(v['own_multiplier']['median'], 2)]
                        + [f"{fmt(st.median(res[k]), 2)}; {sum(abs(e) > TOL[k] + 1e-9 for e in res[k])}" for k in ('q080', 'q030', 'q010')]
                        + [fmt(v['confidence']['median'])])
        table('tab_r1_shapes.tex',
              r'First-session shapes: complete finite triangles; median relative left/right spreads; median $q_{060}/b$; '
              r'straight-branch residuals $E_p=(q_p-b)-\frac{1-p}{0.4}(q_{060}-b)$, which vanish for a triangular (linear) '
              r'right branch: median and number of judgments with $|E_p|$ above the largest residual produced by rounding to '
              r'whole minutes (1.00, 1.75, 2.25 for $p=0.80,0.30,0.10$); median confidence (1--5).',
              'tab:r1-shapes', 'lccccccc',
              r'& Tri. & Left/right & $q_{060}/b$ & $E_{080}$ & $E_{030}$ & $E_{010}$ & Conf.', rows)

    def planning(self):
        p, raw, er = self.p, self.small['raw'], self.ext['raw']
        by = {c: [r for r in er if r['context'] == c] for c in CTX}

        def ratio(c, a, b):
            return st.median(r['min_shift'][a] / r['min_shift'][b] for r in by[c])
        full = {c: 100 * (ratio(c, 'envelope_direct', 'consensus_direct') - 1) for c in CTX}
        loo = {c: {j: 100 * (st.median(r['leave_one_out'][j]['envelope'] / r['leave_one_out'][j]['consensus'] for r in by[c]) - 1)
                   for j in by[c][0]['leave_one_out']} for c in CTX}
        tri = {c: 100 * (ratio(c, 'envelope_paired_triangular', 'envelope_paired_direct') - 1) for c in CTX}
        put('elic.env.min', min(full.values())); put('elic.env.max', max(full.values()))
        all_loo = [v for d in loo.values() for v in d.values()]
        put('elic.loo.min', min(all_loo)); put('elic.loo.max', max(all_loo))
        put('elic.same.tri.min', min(tri.values())); put('elic.same.tri.max', max(tri.values()))
        put('elic.same.viol', sum(r['paired_triangular_plan']['overrun_envelope_paired_direct'] > 1e-9 for r in er))
        put('elic.same.cases', len(er))
        put('elic.cov.viol', sum(r['paired_triangular_plan']['overrun_envelope_direct'] > 1e-9 for r in er))
        put('elic.cov.cases', len(er))
        put('elic.instances', len({(r['family'], r['seed']) for r in raw}))
        put('elic.benchL.cons', sum(bool(r['consensus_direct']['attainable_at_benchmark_L']) for r in raw))
        put('elic.benchL.cases', len(raw))
        att = [r for r in raw if r['consensus_direct']['attainable']]
        put('elic.cons.met', len(att))
        put('elic.cons.viol', sum(r['consensus_direct']['overrun_envelope_direct'] > 1e-9 for r in att))
        put('elic.cons.overrun', 100 * st.median(r['consensus_direct']['overrun_envelope_direct'] for r in att))
        rows = []
        for c in CTX:
            rs = [r for r in att if r['context'] == c]
            rows.append([c, fmt(full[c]), f'{fmt(min(loo[c].values()))}--{fmt(max(loo[c].values()))}', fmt(tri[c]),
                         f"{sum(r['paired_triangular_plan']['overrun_envelope_paired_direct'] > 1e-9 for r in by[c])}/{len(by[c])}",
                         f"{sum(r['consensus_direct']['overrun_envelope_direct'] > 1e-9 for r in rs)}/{len(rs)}"])
        table('tab_elicited.tex',
              r'First-session expert answers on 40 certified synthetic instances at ceiling $0.30$. Price: median '
              r'$100(L^\star_{\rm env}/L^\star_{\rm cons}-1)$; LOO: range of context medians after each respondent is removed. '
              r'Tri.: median excess of the triangular over the direct envelope minimum shift on identical respondents (complete '
              r'finite triangles); violations of that direct envelope at the triangular minimum. Cons.: consensus plans violating '
              r'the direct envelope at $L^{\prime}=\kappa L$.',
              'tab:elicited', 'lccccc', r'Legs & Price (\%) & LOO (\%) & Tri. (\%) & Tri. viol. & Cons. viol.', rows)
        qd = json.loads((RES / 'elicited_quantile.json').read_text())
        assert qd['meta']['data_sha256'] == self.small['meta']['data_sha256']
        qr = qd['raw']; K = sorted({int(k) for r in qr for k in r['price']})
        med = {c: {k: 100 * st.median(r['price'][str(k)] for r in qr if r['context'] == c) for k in K} for c in CTX}
        for k in (1, 2, 3):
            put(f'elic.q{k}.min', min(med[c][k] for c in CTX)); put(f'elic.q{k}.max', max(med[c][k] for c in CTX))
        kmed = (len(K) + 1) // 2
        put('elic.qmed.k', kmed); put('elic.qmed.min', min(med[c][kmed] for c in CTX)); put('elic.qmed.max', max(med[c][kmed] for c in CTX))
        rows = [[f"{k} ({100 * (len(K) - k + 1) // len(K)}\\%)"] + [fmt(med[c][k]) for c in CTX] for k in K if k <= kmed + 1]
        table('tab_elicited_quantile.tex',
              r'Price of protection against the number of protected respondents (certified small instances, ceiling $0.30$): '
              r'median $100(L^\star_k/L^\star_{\rm cons}-1)$ when every leg is planned at the $k$-th largest direct answer of '
              r'its scenario, i.e.\ protecting the minimum marginal share of scenario answers in parentheses; $k=1$ is the envelope of all answers.',
              'tab:elicited-quantile', 'lcccc', '$k$ (marginal min.) & ' + ' & '.join(CTX), rows)
        (RES / 'elicited_analysis.json').write_text(json.dumps(dict(
            data_sha256=self.small['meta']['data_sha256'], session='R1', protection_price_median_percent=full,
            leave_one_out_price_median_percent=loo, paired_triangular_price_median_percent=tri,
            triangular_direct_violations=int(NUM['elic.same.viol']),
            consensus_direct_violations=int(NUM['elic.cons.viol'])), indent=2) + '\n')

    def write_numbers(self):
        lines = [HEADER.strip(), r'\makeatletter',
                 r'\providecommand{\res}[1]{\@ifundefined{res@#1}{\textbf{??}}{\csname res@#1\endcsname}}']
        lines += ['\\expandafter\\def\\csname res@' + k + '\\endcsname{' + v + '}' for k, v in sorted(NUM.items())]
        lines.append(r'\makeatother')
        (TAB / 'numbers_elicited.tex').write_text('\n'.join(lines) + '\n')


def main():
    t = SurveyTables(); t.descriptive(); t.planning(); t.write_numbers()
    print('Survey tables and', len(NUM), 'quoted numbers generated from the R1 panel')


if __name__ == '__main__':
    main()
