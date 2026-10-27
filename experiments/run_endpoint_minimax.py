"""Exact minimax (least-favourable completion) optima versus the exact grid ceiling sweep.

Endpoints: medians of the first-session (R1) answers of the twenty-expert
campaign (results/endpoint_identification.json, protocol.knots), levels
0, .05, .15, .30, .40 and the representative sentinel 1/2. Travel legs follow
a travel scenario, service legs a service scenario, scaled by modal durations.
Horizon: theta times the certified minimum bottleneck of modal route durations.
Workload normalized by the certified minimum workload under optimistic feasibility.
"""
from __future__ import annotations
import argparse, json, sys, time, platform
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from instances_io import CanonicalInstances
from exact_small import SubsetTSP
from exact_fff import RouteColumn, direct_reference
from endpoint_minimax import (EndpointGrid, OWAProfile, ColumnBuilder, CountMILP,
                              GridSweep, Bottleneck)

LEVELS = (0.0, .05, .15, .30, .40, .50)
KNOTS = ('c', 'q010', 'qprime', 'q', 'q080', 'b')
ALPHAS = (.1, .3, .7)
THETAS = (1.2, 1.4)
CONTEXTS = (('s1', 's3'), ('s2', 's4'))
FAMILIES = {  # name: (canonical family, tasks kept, K, P)
    'S12': ('S12', 12, 1, 3), 'F12': ('F12', 12, 2, 2), 'F16': ('F16', 16, 2, 2),
    'C18': ('C1', 18, 2, 3), 'C18w': ('C1', 18, 3, 3)}
SEEDS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
TIEBREAK = {'representative': 1800.0, 'optimistic': 60.0}
PARTS = ROOT / 'results' / 'endpoint_minimax_parts'


def scenario_knots():
    proto = json.loads((ROOT / 'results/endpoint_identification.json').read_text())['protocol']
    med = proto['knots']['scenario_medians']
    return {s: [med[s][k] / med[s]['b'] for k in KNOTS] for s in med}


class Case:
    def __init__(self, fam, seed):
        base, n, K, P = FAMILIES[fam]
        inst = CanonicalInstances().build(base, seed)
        D = np.array([[inst.modal_travel(i, j) for j in range(n + 1)] for i in range(n + 1)])
        t0 = time.perf_counter()
        self.tsp = SubsetTSP(D)
        V = np.zeros(1 << n)
        sv = np.array(inst.service_modal[1:n + 1], float)
        for m in range(1, 1 << n):
            low = m & -m
            V[m] = V[m ^ low] + sv[low.bit_length() - 1]
        self.T, self.V, self.n, self.slots = self.tsp.best, V, n, K * P
        self.fam, self.seed, self.K, self.P = fam, seed, K, P
        self.tsp_seconds = time.perf_counter() - t0


def run_case(args):
    fam, seed, mode = args
    case = Case(fam, seed)
    knots = scenario_knots()
    grid = EndpointGrid(LEVELS)
    prof = OWAProfile(case.slots)
    masks = np.arange(1, 1 << case.n)
    modal = case.T[masks] + case.V[masks]
    t0 = time.perf_counter()
    B = Bottleneck(masks, modal, case.n, case.slots).value()
    bott_s = time.perf_counter() - t0
    out = []
    for theta in THETAS:
        L = theta * B
        for ts, ss in CONTEXTS:
            cols = ColumnBuilder(case.T, case.V, grid, knots[ts], knots[ss], L,
                                 representative=(mode == 'representative')).build(case.n)
            milp = CountMILP(cols, case.slots, grid, prof)
            base = milp.solve(1.0, 1.0)                       # minimum workload
            scale = base['workload']
            sweep = GridSweep(milp, tiebreak_limit=TIEBREAK[mode]).plans(scale)
            rec = dict(family=fam, seed=seed, n=case.n, K=case.K, P=case.P, slots=case.slots,
                       theta=theta, L=L, bottleneck=B, context=ts + '+' + ss, columns=int(cols.masks.size),
                       scale=scale, w1=float(prof.w[0]), gap_grid=grid.gap(),
                       sweep={str(k): {x: v[x] for x in ('status', 'workload', 'fff_upper', 'max_upper', 'counts', 'seconds', 'tiebreak', 'secondary_status', 'secondary_seconds')
                                       if x in v} for k, v in sweep.items()},
                       alphas={})
            enum = None
            if case.n <= 12:
                upp = {int(m): [RouteColumn(int(m), tuple(case.tsp.tour(int(m))), float(w), float(u), .7 * float(w))]
                       for m, w, u in zip(cols.masks, cols.workload, cols.upper)}
                enum = direct_reference(upp, case.n, case.K, case.P, scale, ALPHAS)
            for a in ALPHAS:
                opt = milp.solve(a, scale)
                assert opt['status'] == 'Optimal', opt
                sp, sj = GridSweep.best(sweep, a, scale)
                assert sj >= opt['J'] - 1e-9
                bound = (1 - a) * (1 - prof.w[0]) * opt['max_upper']
                assert sj - opt['J'] <= bound + 1e-9
                r = dict(J=opt['J'], workload=opt['workload'], fff_upper=opt['fff_upper'],
                         fff_lower=opt['fff_lower'], max_upper=opt['max_upper'], counts=opt['counts'],
                         masks=opt['masks'], seconds=opt['seconds'], sweep_J=sj, sweep_counts=sp['counts'],
                         sweep_gap=sj - opt['J'], sweep_gap_pct=100 * (sj / opt['J'] - 1), bound=bound,
                         sweep_workload=sp['workload'], sweep_fff_upper=sp['fff_upper'],
                         sweep_max_upper=sp['max_upper'], sweep_tiebreak=sp.get('tiebreak'),
                         sweep_secondary_status=sp.get('secondary_status'),
                         workload_saving_pct=100 * (opt['workload'] / sp['workload'] - 1),
                         owa_change=opt['fff_upper'] - sp['fff_upper'],
                         concentrates=bool(opt['max_upper'] >= sp['max_upper'] - 1e-12
                                           and opt['counts'][0] <= sp['counts'][0]))
                if enum is not None:
                    r['enumeration_J'] = enum['direct'][str(a)]['J']
                    r['enumeration_sweep_J'] = enum['oracle_sweep'][str(a)]['J']
                    assert abs(r['enumeration_J'] - opt['J']) <= 1e-7, (r['enumeration_J'], opt['J'])
                    assert abs(r['enumeration_sweep_J'] - sj) <= 1e-7, (r['enumeration_sweep_J'], sj)
                rec['alphas'][str(a)] = r
            out.append(rec)
    print(fam, seed, f'B={B:.2f} tsp={case.tsp_seconds:.1f}s bottleneck={bott_s:.1f}s', flush=True)
    res = dict(family=fam, seed=seed, tsp_seconds=case.tsp_seconds, bottleneck_seconds=bott_s, records=out)
    PARTS.mkdir(exist_ok=True)
    (PARTS / f'{mode}_{fam}_{seed}.json').write_text(json.dumps(res) + '\n')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', nargs='*', default=list(FAMILIES))
    ap.add_argument('--seeds', nargs='*', type=int, default=list(SEEDS))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--feasibility', choices=['representative', 'optimistic'], default='representative')
    a = ap.parse_args()
    out = 'endpoint_minimax.json' if a.feasibility == 'representative' else 'endpoint_minimax_optimistic.json'
    jobs = [(f, s, a.feasibility) for f in a.families for s in a.seeds]
    part = lambda f, s: PARTS / f'{a.feasibility}_{f}_{s}.json'
    todo = [j for j in jobs if not part(j[0], j[1]).exists()]
    with Pool(a.workers) as pool:
        pool.map(run_case, todo, chunksize=1)
    res = [json.loads(part(f, s).read_text()) for f, s, _ in jobs]
    payload = dict(protocol=dict(levels=LEVELS, knots=KNOTS, alphas=ALPHAS, thetas=THETAS,
                                 contexts=['+'.join(c) for c in CONTEXTS],
                                 families={k: dict(base=v[0], n=v[1], K=v[2], P=v[3]) for k, v in FAMILIES.items()},
                                 seeds=SEEDS, owa='arithmetic profile',
                                 endpoints='R1 scenario medians normalized by the representative duration',
                                 feasibility=a.feasibility,
                                 route_rule=('modal route duration at most L (upper risk at most 1/2)' if a.feasibility == 'representative'
                                             else '0.7 times the modal route duration at most L'),
                                 horizon='theta times the certified minimum bottleneck of modal route durations',
                                 scale='certified minimum workload', solver='HiGHS set partitioning, zero gap',
                                 sweep=f'minimum workload at every grid ceiling; ties by minimum upper OWA within {TIEBREAK[a.feasibility]:.0f} s, otherwise the first minimum-workload plan',
                                 python=sys.version.split()[0], platform=platform.platform()),
                   raw=res)
    (ROOT / 'results' / out).write_text(json.dumps(payload, indent=1) + '\n')
    for f, s, _ in jobs:
        part(f, s).unlink()
    if not any(PARTS.iterdir()):
        PARTS.rmdir()

if __name__ == '__main__':
    main()
