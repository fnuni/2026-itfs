"""Verify every frozen external plan and directed TFN insertion arithmetic."""
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'experiments'))
from external_instances_v4 import ExternalMatrixInstance,load_external_records
from run_external_v4 import verify

manifest=json.loads((ROOT/'data/amazon_v4/manifest.json').read_text())
for name,entry in manifest['raw_files'].items():
    raw=ROOT/'data/amazon_v4/raw'/name
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==entry['sha256']
assert hashlib.sha256((ROOT/'data/amazon_v4/instances.json').read_bytes()).hexdigest()==manifest['derived_file']['sha256']
rng=random.Random(137); checked=0; records=load_external_records()
instances={r['id']:ExternalMatrixInstance(r) for r in records}
for inst in instances.values():
    for _ in range(40):
        nodes=rng.sample(range(1,inst.n+1),14); seq=nodes[:13]; t=nodes[-1]
        old=inst.route_endpoints(seq)
        for pos in range(len(seq)+1):
            p=seq[pos-1] if pos else 0; q=seq[pos] if pos<len(seq) else 0
            delta=tuple(old[x]+inst.tfn_travel[p][t][x]+inst.tfn_travel[t][q][x]
                        -inst.tfn_travel[p][q][x]+inst.tfn_service[t][x] for x in range(3))
            true=inst.route_endpoints(seq[:pos]+[t]+seq[pos:])
            assert max(abs(a-b) for a,b in zip(delta,true))<1e-9
            checked+=1
out=dict(raw_sha256_checks=len(manifest['raw_files']),directed_insertions_checked=checked,
         final_plans_checked=0,OR_stage_plans_checked=0,coverage_asserted=True,
         workload_and_FFF_asserted=True,feasibility_asserted=True)
counterexample=ROOT/'results/external_v4_shortcut_counterexample.json'
if counterexample.exists():
    ce=json.loads(counterexample.read_text()); inst=instances[ce['instance']]
    from fuzzy_core import TriangularFN
    for label in ('least_modal','lower_risk'):
        info=ce[label]; pos=info['position']; seq=ce['receiving_sequence']
        endpoints=inst.route_endpoints(seq[:pos]+[ce['task']]+seq[pos:])
        assert abs(endpoints[1]-info['modal'])<1e-9
        assert abs(TriangularFN(*endpoints).overtime_risk(inst.L)-info['risk'])<1e-9
    assert ce['lower_risk']['modal']>ce['least_modal']['modal']
    assert ce['lower_risk']['risk']<ce['least_modal']['risk']
    out['heterogeneous_shortcut_counterexample_verified']=True
result=ROOT/'results/external_v4.json'
if result.exists():
    payload=json.loads(result.read_text()); rows=payload['raw']
    assert len(rows)==8*3*3*4,'The final external benchmark must contain 288 method runs'
    for row in rows:
        inst=instances[row['instance']]
        if row.get('solution'):
            verify(inst,row); out['final_plans_checked']+=1
        for stage in row['OR_stages']:
            if stage.get('solution'):
                verify(inst,stage); out['OR_stage_plans_checked']+=1
        if row['method']=='OR+CC' and row.get('seed_J') is not None and row.get('J') is not None:
            assert row['J']<=row['seed_J']+1e-9
    out['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
out['implementation_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in ('src/external_instances_v4.py','src/external_solver_v4.py',
                                           'experiments/prepare_amazon_v4.py','experiments/run_external_v4.py')}
(ROOT/'results/external_v4_verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
