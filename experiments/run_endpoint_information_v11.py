"""Run the frozen same-information and fixed-stress extension of the v11 study."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from external_instances_v4 import load_external_records
from endpoint_decision_v11 import DecisionCase
from endpoint_information_v11 import extend_case


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    frozen_path = ROOT/'results/endpoint_information_v11_protocol.json'
    frozen = json.loads(frozen_path.read_text())
    original_path = ROOT/'results/endpoint_decision_v11.json'
    original = json.loads(original_path.read_text())
    assert sha(original_path) == frozen['original_record_sha256']
    assert sha(ROOT/'data/amazon_v4/instances.json') == frozen['source_sha256']
    assert frozen['smooth_powers'] == [.45, 1., 2.2]
    assert frozen['stress_powers'] == [.1, 10.]
    originals = {r['id']: r for r in original['raw']}
    protocol = dict(original['protocol'], extension=frozen,
        frozen_protocol_sha256=sha(frozen_path), original_record_sha256=sha(original_path),
        smooth_test_field='tests', stress_test_field='stress_tests',
        common_adversary_field='common_adversary',
        python=sys.version.split()[0], platform=platform.platform(),
        extension_code_sha256={p: sha(ROOT/p) for p in
            ['src/endpoint_information_v11.py', 'experiments/run_endpoint_information_v11.py']})
    output = dict(protocol=protocol, raw=[])
    sources = load_external_records()
    if args.smoke:
        sources = sources[:1]
    for source in sources:
        case = DecisionCase(source)
        output['raw'].append(extend_case(case, originals[case.id], frozen))
        print(case.id, 'anchored original decisions, PL and two fixed stresses complete', flush=True)
    name = 'endpoint_information_v11_smoke.json' if args.smoke else 'endpoint_information_v11.json'
    destination = ROOT/'results'/name
    destination.write_text(json.dumps(output, indent=1)+'\n')
    print(destination)


if __name__ == '__main__':
    main()
