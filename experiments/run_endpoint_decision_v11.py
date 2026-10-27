"""Run the deterministic heterogeneous finite-information decision experiment."""
from pathlib import Path
import argparse, hashlib, json, platform, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from external_instances_v4 import load_external_records
from endpoint_decision_v11 import (DecisionCase, LEVELS, GRIDS, PROFILES, ALPHAS,
                                   TEST_POWERS, THETA, CAP, TOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    source = ROOT/'data/amazon_v4/instances.json'
    payload = dict(protocol=dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sampling='All eight released Amazon scaffolds, first seven canonical task indices; no outcome-based selection',
        domain='All directed routes; all unlabelled partitions into at most three cells; every modal route duration <= L',
        horizon=f'L={THETA} times the exact minimum modal bottleneck, before endpoint objectives',
        scale='Exact minimum modal workload on the common modal-feasible domain',
        grids={name: LEVELS[ix].tolist() for name, ix in GRIDS.items()},
        interior_endpoint_formula='q_e(rho)=b_e+(c_e-b_e)*(1-2*rho)^gamma_e; travel gamma=(.5,1,2,3)[(3i+5j)%4], service gamma=(.5,1,2,3)[(3i+1)%4]',
        common_support='Every completion has the same original lower support a, mode b and upper support c; the common left branch is linear from (a,0) to (b,1). Only the right branch is used on modal-feasible routes.',
        uncertainty='Constructed stress scenarios; only modal inputs are operational; no new human data or uncertainty calibration',
        triangular_baseline='Linear interpolation from b,c, without fitting interior endpoints; deliberate representation convention',
        hidden_test_completions=dict(powers=TEST_POWERS,
            formula='Between each pair of adjacent full-grid endpoints use q_lo-(q_lo-q_hi)*z^[p*(1+.12*(leg_id%3-1))]',
            separation='All full-grid values are shared; interior interpolations are used only after optimization to evaluate every method against an exact oracle on the same domain'),
        profiles={k: v.tolist() for k, v in PROFILES.items()}, alphas=ALPHAS, cap=CAP,
        cap_policy='Apply the same endpoint-certified worst-cell cap to triangular, minimax and exact-sweep methods within each grid; retain infeasibility',
        regret='Test objective minus exact test optimum on the identical grid/policy feasible set; cap cost separately versus uncapped',
        sweep='Every attainable upper-risk ceiling, exact minimum workload, then minimum upper OWA',
        inference='Descriptive eight-route experiment; repeated alpha/grid/weight/completions are not independent samples',
        tolerance=TOL, python=sys.version.split()[0], platform=platform.platform(),
        timing='No time-limited or comparative timing experiment'), raw=[])
    records = load_external_records()
    if args.smoke:
        records = records[:1]
    path = ROOT/'results'/('endpoint_decision_v11_smoke.json' if args.smoke else 'endpoint_decision_v11.json')
    for rec in records:
        result = DecisionCase(rec).run()
        payload['raw'].append(result)
        path.write_text(json.dumps(payload, indent=1)+'\n')
        print(result['id'], 'plans', result['modal_feasible_plans'], 'of', result['total_plans'], flush=True)
    payload['protocol']['code_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        for p in ['src/endpoint_decision_v11.py', 'experiments/run_endpoint_decision_v11.py']}
    path.write_text(json.dumps(payload, indent=1)+'\n')


if __name__ == '__main__':
    main()
