# Public operational routes: replay protocol

Eight complete routes of the public model-application sample of the 2021 Amazon
Last Mile Routing Research Challenge (first eight route identifiers in lexical
order, chosen before any solver run) keep their depot, all 132-192 delivery
stops, directed travel-time matrices and package-level planned service times.
Source: https://registry.opendata.aws/amazon-last-mile-challenges/ ; dataset
description: Merchán et al., Transportation Science, DOI 10.1287/trsc.2022.1173.
Copyright Amazon.com, Inc. or affiliates; CC BY-NC 4.0. Source URLs, sizes,
SHA-256 checksums and the access date (10 September 2026) are in
`data/amazon_v4/manifest.json`; the licence and the downloaded files are in
`data/amazon_v4/raw/`.

Published travel and service values are used as modal durations of triangular
memberships with fixed heterogeneous spread rules. Two vehicles and three shifts
are imposed; the published sequence is split into six contiguous chunks whose
depot-return workloads sum to the reference workload, and the shift limit is
that workload divided by 6 x 0.90. Time windows and capacities are not used.

Fixed protocol (`results/external_v4_protocol.json`, 10 September 2026):
alpha in {0.1, 0.3, 0.7}; CC-LNS seeds 10, 20, 30; four seconds per method,
including instance and model construction, seeding and search; two worker
processes with rotated method order; final verification outside the allowance.
OR-sweep solves the ceilings 0.25, 0.40, 0.60, 0.80 with OR-Tools and keeps the
fully served feasible plan of lowest J. CC-full evaluates every insertion
position with exact directed endpoint deltas; CC-modal only the least
modal-cost position. OR+CC gives two seconds to the sweep and two to CC-LNS.
Symmetric 2-opt is disabled on directed routes.

```
python3 experiments/prepare_amazon_v4.py
python3 experiments/run_external_v4.py --workers 2
python3 experiments/summarize_external_v4.py
python3 validation/verify_external_v4.py
python3 validation/validate_external_v4_independent.py
```

The source route is the replication unit: repetitions are averaged within each
route and alpha before paired route-level comparisons; bootstrap intervals over
eight routes are descriptive.
