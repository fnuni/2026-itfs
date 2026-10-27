"""Describe dependence and separate certified optimistic sweep comparisons.

Reads the existing raw results without replacing them or running a solver.
Every retained optimistic configuration must have a certified second-stage
OWA tie-break at every feasible grid ceiling; selection is by solver status,
not by the sign or size of its resulting objective gap.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(name):
    data = json.loads((ROOT / 'results' / name).read_text())
    return data, [r for case in data['raw'] for r in case['records']]


def certified(record):
    return all(s.get('tiebreak') == 'optimal' for s in record['sweep'].values()
               if s['status'] == 'Optimal')


def identity(r):
    return dict(family=r['family'], seed=r['seed'], context=r['context'], theta=r['theta'])


def summary(rows, alpha=None):
    vals = [r for row in rows for a, r in row['alphas'].items()
            if alpha is None or a == alpha]
    lost = [r for r in vals if r['sweep_gap'] > 1e-7]
    gaps = [r['sweep_gap_pct'] for r in vals]
    return dict(configurations=len(rows), comparisons=len(vals), losses=len(lost),
                mean_gap_pct=st.mean(gaps), median_gap_pct=st.median(gaps), max_gap_pct=max(gaps),
                mean_gap_among_losses_pct=st.mean(r['sweep_gap_pct'] for r in lost) if lost else 0.0,
                max_absolute_gap=max(r['sweep_gap'] for r in vals),
                optimum_modal_overruns=sum(r['counts'][-1] > 0 for r in vals))


def geometry_audit(protocol):
    families = {}
    for name in ('instances.json', 'instances_small.json'):
        families.update(json.loads((ROOT / 'data' / name).read_text())['families'])
    groups = {}
    for family, spec in protocol['families'].items():
        for record in families[spec['base']]['instances']:
            if record['seed'] not in protocol['seeds']:
                continue
            coords = [record['depot']] + [[t['x'], t['y']] for t in record['tasks'][:spec['n']]]
            fingerprint = hashlib.sha256(json.dumps(coords, separators=(',', ':')).encode()).hexdigest()
            groups.setdefault(fingerprint, []).append(dict(family=family, seed=record['seed']))
    return dict(unique_coordinate_geometries=len(groups), family_seed_combinations=sum(map(len, groups.values())),
                coordinate_definition='depot and ordered first n task coordinates, without duration scale or K/P',
                geometry_groups=groups,
                interpretation='Context, horizon and alpha reuse instances; S12/F12 and C18/C18w share coordinates. No independence claim or confidence interval is inferred.')


def main():
    principal, mainrows = records('endpoint_minimax.json')
    optimistic, optrows = records('endpoint_minimax_optimistic.json')
    assert all(certified(r) for r in mainrows)
    keep = [r for r in optrows if certified(r)]
    omit = [r for r in optrows if not certified(r)]
    alphas = [str(a) for a in principal['protocol']['alphas']]
    result = dict(
        source_sha256={name: hashlib.sha256((ROOT / 'results' / name).read_bytes()).hexdigest()
                       for name in ('endpoint_minimax.json', 'endpoint_minimax_optimistic.json')},
        selection='All feasible ceilings must have optimal OWA tie-break status; retain or exclude all alpha comparisons of that configuration together.',
        geometry=geometry_audit(principal['protocol']),
        principal=dict(all=summary(mainrows), by_alpha={a: summary(mainrows, a) for a in alphas}),
        optimistic=dict(
            certified=summary(keep), exploratory=summary(omit), all_optima=summary(optrows),
            certified_by_alpha={a: summary(keep, a) for a in alphas},
            certified_ids=[identity(r) for r in keep], exploratory_ids=[identity(r) for r in omit],
            time_limited_ceiling_tiebreaks=sum(s.get('tiebreak', '').startswith('time limit')
                                              for r in optrows for s in r['sweep'].values()),
            uncertified_ceiling_tiebreaks=sum(s['status'] == 'Optimal' and s.get('tiebreak') != 'optimal'
                                              for r in optrows for s in r['sweep'].values()),
            confirmed_secondary_time_limits=sum(s.get('secondary_status') == 'Time limit reached'
                                                for r in optrows for s in r['sweep'].values()),
            legacy_fallbacks_without_secondary_status=sum(s.get('tiebreak', '').startswith('time limit')
                                                          and 'secondary_status' not in s
                                                          for r in optrows for s in r['sweep'].values()),
            legacy_field_note='time_limited_ceiling_tiebreaks counts the legacy fallback label. The legacy driver used that label for every nonoptimal secondary status; it does not prove that these were actual timeouts.',
            interpretation='All direct minimax optima are certified. Only the retained subset has fully certified oracle sweep tie-breaks. Exploratory gaps can include tie-breaking search error. The certified subset is selected by computational tractability and is not a representative sample.'))
    assert result['principal']['all']['comparisons'] == 600
    assert result['optimistic']['certified']['comparisons'] + result['optimistic']['exploratory']['comparisons'] == 600
    (ROOT / 'results/minimax_audit_v11.json').write_text(json.dumps(result, indent=2) + '\n')

    macros = {}
    for label, obj in [('main', result['principal']['all']), ('optimistic.cert', result['optimistic']['certified']),
                       ('optimistic.exploratory', result['optimistic']['exploratory'])]:
        for key in ('configurations', 'comparisons', 'losses', 'optimum_modal_overruns'):
            macros['audit.' + label + '.' + key] = str(obj[key])
        for key in ('mean_gap_pct', 'median_gap_pct', 'max_gap_pct'):
            macros['audit.' + label + '.' + key] = f'{obj[key]:.2f}'
    macros['audit.geometries'] = str(result['geometry']['unique_coordinate_geometries'])
    macros['audit.familyseeds'] = str(result['geometry']['family_seed_combinations'])
    macros['audit.optimistic.timeouts'] = str(result['optimistic']['time_limited_ceiling_tiebreaks'])
    macros['audit.optimistic.uncertified'] = str(result['optimistic']['uncertified_ceiling_tiebreaks'])
    for a, obj in result['principal']['by_alpha'].items():
        for key in ('mean_gap_pct', 'median_gap_pct', 'max_gap_pct'):
            macros['audit.alpha.' + a + '.' + key] = f'{obj[key]:.2f}'
    macrotext = '% Generated by experiments/audit_minimax_v11.py; descriptive values, not independent replications.\n'
    macrotext += '\n'.join('\\expandafter\\def\\csname res@' + k + '\\endcsname{' + v + '}' for k, v in macros.items()) + '\n'
    (ROOT / 'tables/numbers_minimax_audit_v11.tex').write_text(macrotext)

    lines = [r'\begin{table}[tb]', r'\centering',
             r'\caption{Descriptive sweep gaps in the principal study (all tie-breaks certified). Contexts, horizons and preferences reuse 30 coordinate geometries; the 600 comparisons are not independent replications. Mean and median include zero gaps.}',
             r'\label{tab:minimax-audit-v11}', r'\scriptsize\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
             r'$\alpha$ & Comparisons & Lost & Mean (\%) & Median (\%) & Max. (\%)\\\midrule']
    for a in alphas + ['All']:
        obj = result['principal']['all'] if a == 'All' else result['principal']['by_alpha'][a]
        lines.append(f"{a} & {obj['comparisons']} & {obj['losses']} & {obj['mean_gap_pct']:.2f} & {obj['median_gap_pct']:.2f} & {obj['max_gap_pct']:.2f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (ROOT / 'tables/tab_minimax_audit_v11.tex').write_text('\n'.join(lines))
    print(json.dumps(dict(principal=result['principal'], geometry_counts={k: v for k, v in result['geometry'].items() if k != 'geometry_groups'},
                          optimistic={k: result['optimistic'][k] for k in ('certified', 'exploratory', 'time_limited_ceiling_tiebreaks')}), indent=2))


if __name__ == '__main__':
    main()
