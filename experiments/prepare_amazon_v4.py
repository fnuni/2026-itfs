"""Freeze the first eight complete routes in the official 13-route apply sample.

Data are Copyright Amazon.com, Inc. or affiliates, CC BY-NC 4.0. This is an
operational-data scaffold with scenario multi-shifts/memberships, not an
observed fuzzy multi-shift deployment. No solver outcome enters extraction.
Run from the repository root: python3 experiments/prepare_amazon_v4.py
Use --download to obtain raw public files (urllib, no AWS credentials).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'amazon_v4'
BASE = ('https://amazon-last-mile-challenges.s3.us-west-2.amazonaws.com/'
        'almrrc2021/almrrc2021-data-training/')
URLS = {name: BASE + folder + '/' + name for folder, name in (
    ('model_apply_inputs', 'new_route_data.json'),
    ('model_apply_inputs', 'new_package_data.json'),
    ('model_apply_inputs', 'new_travel_times.json'),
    ('model_score_inputs', 'new_actual_sequences.json'))}
URLS['License.txt'] = ('https://amazon-last-mile-challenges.s3.us-west-2.amazonaws.com/'
                        'almrrc2021/License.txt')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(download=False):
    raw = OUT / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if download:
            urlretrieve(url, raw / name)
        if not (raw / name).exists():
            raise FileNotFoundError(f'{name}: run with --download')
    r, p, t, a = [json.loads((raw / name).read_text()) for name in (
        'new_route_data.json', 'new_package_data.json',
        'new_travel_times.json', 'new_actual_sequences.json')]
    records = []
    for idx, rid in enumerate(sorted(r)[:8], 1):
        rr = r[rid]
        depots = [s for s, d in rr['stops'].items() if d['type'] == 'Station']
        assert len(depots) == 1
        depot = depots[0]
        drops = sorted(s for s, d in rr['stops'].items() if d['type'] == 'Dropoff')
        assert len(drops) + 1 == len(rr['stops'])
        stops = [depot] + drops
        travel = [[float(t[rid][i][j]) / 60.0 for j in stops] for i in stops]
        service = [0.0] + [sum(float(v['planned_service_time_seconds'])
                                  for v in p[rid].get(s, {}).values()) / 60.0
                            for s in drops]
        assert all(math.isfinite(x) and x >= 0 for row in travel for x in row)
        assert all(math.isfinite(x) and x >= 0 for x in service)
        ids = {s: i for i, s in enumerate(stops)}
        order = [ids[s] for s in sorted(drops, key=lambda s: a[rid]['actual'][s])]
        n, slots = len(drops), 6
        chunks = [order[q*n//slots:(q+1)*n//slots] for q in range(slots)]
        def duration(seq):
            arcs = [0] + seq + [0]
            return sum(travel[i][j] for i, j in zip(arcs, arcs[1:])) + sum(service[i] for i in seq)
        w0 = sum(duration(seq) for seq in chunks)
        records.append(dict(id=f'A{idx}', source_route_id=rid,
                            source_station=rr['station_code'], source_date=rr['date_YYYY_MM_DD'],
                            n=n, K=2, P=3, rho=0.90, L=w0/(slots*0.90), workload_scale=w0,
                            stop_ids=stops,
                            zones=[rr['stops'][s].get('zone_id') if isinstance(rr['stops'][s].get('zone_id'),str) else None for s in stops],
                            package_counts=[0] + [len(p[rid].get(s, {})) for s in drops],
                            travel_minutes=travel, service_minutes=service,
                            historical_split=[chunks[:3], chunks[3:]],
                            coordinates=[[rr['stops'][s]['lat'], rr['stops'][s]['lng']] for s in stops]))
    output = OUT / 'instances.json'
    output.write_text(json.dumps(dict(format='ms-vrppu-fr/amazon-v4', records=records), indent=1,
                                     allow_nan=False))
    manifest = dict(accessed='2026-09-10', registry='https://registry.opendata.aws/amazon-last-mile-challenges/',
                    documentation='https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md',
                    dataset_doi='10.1287/trsc.2022.1173', license='CC BY-NC 4.0',
                    raw_files={name: dict(url=url, bytes=(raw/name).stat().st_size,
                                         sha256=digest(raw/name)) for name, url in URLS.items()},
                    derived_file=dict(name='instances.json', sha256=digest(output)),
                    selection='First 8 route IDs in lexicographic order of the complete 13-route public apply sample; retain every dropoff.',
                    translation='Travel seconds / 60; sum package planned_service_time_seconds at each stop / 60; original depot; asymmetric matrix retained.',
                    scenario='K=2,P=3; historical order split into six contiguous equal-count chunks with depot returns; L = total split modal duration / (6 * 0.90). Fuzzy spreads are scenario assumptions, not estimated from observations.',
                    membership_formula='Travel TFN: left=.10+.05*((i+2*j)%3), right=.20+.10*((3*i+j)%4)+.10*(different nonmissing zones). Service TFN: left=.10+.025*(package_count%3), right=.20+.05*min(package_count,5).',
                    omitted_constraints='Package time windows, van volume capacity, executor identity, departure time and driver preferences are not modeled. No claim to reproduce historical dispatch.',
                    routes=[dict(id=x['id'],source_route_id=x['source_route_id'],n=x['n'],L=x['L']) for x in records])
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['routes'], indent=1))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--download', action='store_true')
    prepare(ap.parse_args().download)
