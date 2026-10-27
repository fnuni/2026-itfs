"""Run the v12 certified-interpolation and neutral-completion evaluation."""
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
from endpoint_certified_v12 import evaluate_case


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    frozen_path = ROOT/'results/endpoint_certified_v12_protocol.json'
    frozen = json.loads(frozen_path.read_text())
    original = json.loads((ROOT/'results/endpoint_decision_v11.json').read_text())
    information = json.loads((ROOT/'results/endpoint_information_v11.json').read_text())
    originals = {r['id']: r for r in original['raw']}
    infos = {r['id']: r for r in information['raw']}
    output = dict(protocol=dict(frozen, frozen_protocol_sha256=sha(frozen_path),
        original_record_sha256=sha(ROOT/'results/endpoint_decision_v11.json'),
        information_record_sha256=sha(ROOT/'results/endpoint_information_v11.json'),
        python=sys.version.split()[0], numpy=__import__('numpy').__version__,
        platform=platform.platform(),
        code_sha256={p: sha(ROOT/p) for p in
            ['src/endpoint_certified_v12.py', 'experiments/run_endpoint_certified_v12.py']}), raw=[])
    sources = load_external_records()
    if args.smoke:
        sources = sources[:1]
    for k, source in enumerate(sources):
        case = DecisionCase(source)
        output['raw'].append(evaluate_case(case, originals[case.id], infos[case.id], frozen, k))
        print(case.id, 'certified interpolation and neutral completions complete', flush=True)
    name = 'endpoint_certified_v12_smoke.json' if args.smoke else 'endpoint_certified_v12.json'
    destination = ROOT/'results'/name
    destination.write_text(json.dumps(output, indent=1)+'\n')
    print(destination)


if __name__ == '__main__':
    main()
