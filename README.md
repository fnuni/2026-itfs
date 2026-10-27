# MS-VRPPU-FR — reproducibility package

Release **12.0.0**, **2026-10-27**, accompanies manuscript version 12,
**Finite-Endpoint Credibility Models for Fleet Routing: Sharp Risk Bounds and
Certified Ordered-Risk Decisions**. This package contains code, canonical
instances, anonymous numerical questionnaire answers, result records and
generated experimental tables. `PACKAGE_MANIFEST.json` records the size and
SHA-256 of every included file except the manifest itself.

## Repository and citation

The anonymous review package is available at
<https://github.com/fnuni/2026-itfs>. The tag `v12.0.0` identifies the exact
submitted code-and-data version. Author metadata and Git commit authorship
are anonymous; the repository remains under the GitHub account in this URL.
No DOI has been created or assigned. Cite the repository URL and version
12.0.0, as in the accompanying manuscript. The submission ZIP and this tagged
tree contain the same manifest-covered payload files.

## Release 12 additions

Release 12 adds, without modifying any v11 record:

* `results/endpoint_certified_v12.json` — certified interpolation CI-tau
  (tau = 0, 1, 2, 5, 10 %): the piecewise-linear (PL) objective is minimized
  among plans whose certified worst-case objective J+ is within (1+tau) of
  the minimax value; a neutral completion family (30 draws per input,
  exponents log-uniform on [0.1, 10] per leg and endpoint interval) is
  evaluated together with the original smooth and stress families. The
  protocol `endpoint_certified_v12_protocol.json` was recorded before the
  runs, after inspection of the v11 results; it is not preregistration.
* `data/reliability20/strata.csv` and `results/reliability20_strata_v12.json`
  — coarse self-reported area and experience band per pseudonymous code and
  the stratified residual counts and q060 agreement.
* Independent validators `validation/validate_endpoint_certified_v12.py` and
  `validation/validate_reliability20_strata_v12.py`, and the table generator
  `experiments/make_v12_tables.py`.

```sh
python3 experiments/run_endpoint_certified_v12.py
python3 experiments/run_reliability20_strata_v12.py
python3 validation/validate_endpoint_certified_v12.py
python3 validation/validate_reliability20_strata_v12.py
python3 experiments/make_v12_tables.py
```

The certified-interpolation driver anchors the archived v11 decisions and
asserts that PL selections and smooth-test regrets are reproduced.

## New and inherited evidence

Release 11 corrects the finite-endpoint constructions: the lower membership
has singleton core `{b}` and preserves the representative sentinel. Sharpness
concerns a fixed finite domain of representatively feasible plans and a fixed
horizon. With a positive first ceiling, an upper completion needs a finite
tail chosen for that horizon. New tests check complete memberships, strong
endpoints, sentinel equality and both possibility and necessity directly.
The stored principal minimax results depend on endpoint values and are unchanged.

`endpoint_decision_v11.json` is a new exhaustive experiment using all eight
released Amazon scaffolds restricted to seven stops. It compares two/four
interior endpoints, two OWA profiles, a common worst-cell cap, direct minimax,
exact ceiling sweeps and a triangular representation convention. Interior
endpoints and test completions are **constructed stress scenarios**, not
observations or additional elicitation. The triangle is deliberately
misspecified relative to these nontriangular endpoints; the comparison does
not establish universal superiority over triangular models. Methods share the
same feasible domain within each policy, and regret uses an exact oracle on
that domain. Repeated preferences, grids, weights and completions are not
independent observations.

Patch 11.0.1 leaves the original `endpoint_decision_v11.json`, its core
implementation and its driver unchanged. The additive
`endpoint_information_v11.json` supplies a piecewise-linear point-completion
benchmark using the same available endpoints as minimax on each grid. It
optimizes that completion on the same domain, with the same cap, weights and
workload scale. The original three smooth tests use powers **0.45, 1, 2.2**.
Two additional stress tests use **0.1 and 10**, and are reported separately,
not averaged with the three original tests as an empirical distribution.

`endpoint_information_v11_protocol.json` records extension choices after the
original results were inspected and before the added stress evaluations; it
is **not preregistration**. The original eight inputs and all grid, policy,
weight and preference settings are retained. For each grid, the common
least-favourable completion yields each plan's `upper_J`; this is a robust
certificate under a common adversary, not another independent empirical win.
Point interpolation may perform better on the selected smooth completions;
the comparison does not establish universal superiority of either method.
`RELEASE_NOTES.md` summarizes the patch and its provenance.

The synthetic benchmarks, routing-engine comparisons, direct-objective
references, original minimax runs, twenty-person questionnaire and full-route
replay are inherited records. They were not all rerun for release 11; their
saved protocols identify the original environments and settings. New
validation/table-generation runs are distinct from new solver runs.

`minimax_audit_v11.json` distinguishes 30 coordinate geometries, 50 family/seed
combinations, 200 context/horizon configurations and 600 preference
comparisons. Every principal-study sweep tie-break is certified. In the
inherited optimistic-feasibility variant, **52 ceiling tie-breaks are
uncertified**. The legacy driver recorded these as time-limit fallbacks for
any nonoptimal second-stage status, so the raw records do not prove that all
52 actually timed out. Only 444 comparisons have fully certified sweeps;
they contain 287 losses. The remaining 156 comparisons are exploratory and
their 109 observed losses may include tie-breaking search error. Selection
uses solver status, not gap size, but the certified subset is not a random
sample. All 600 direct minimax optima are certified. Raw records are preserved;
future runs record the actual secondary solver status.

## Layout

```
src/            fuzzy memberships, credibility, routing, OWA and exact solvers
  endpoint_minimax.py       full staircases, count MILP and ceiling sweep
  endpoint_decision_v11.py  exhaustive heterogeneous finite-information study
  endpoint_information_v11.py  additive point-completion and stress comparison
  endpoint_certified_v12.py    certified interpolation and neutral completions
experiments/    experiment drivers, audits and table generators
validation/     independent membership, arithmetic, domain and result checks
elicitation/    questionnaire reader and agreement statistics
data/           canonical instances, public Amazon inputs, numerical answers
results/        results and validation reports with protocol metadata
tables/         generated LaTeX experimental tables and number macros
```

## Dependencies

Use Python 3.10 or newer. `requirements.txt` lists all imported third-party
packages. New release-11 checks were tested with Python 3.10, NumPy 1.26.4 and
HiGHS/highspy 1.15.1. SciPy 1.15.3 was present but is not imported by these
sources. Minimum requirements are not a lockfile or a claim that future
versions produce identical output.

For the new experiment, audit and membership validation alone:

```sh
python3 -m pip install numpy==1.26.4 highspy==1.15.1
```

For all workflows:

```sh
python3 -m pip install -r requirements.txt
```

OR-Tools and `hygese` are required only to rerun inherited routing engines;
their original versions/platforms are recorded with their results. Run
engines and HiGHS exact solvers in separate processes. `pypdf`, or the external
Poppler `pdftotext` command, is used only for original PDF-form import, not
analysis of the released CSV.

## Questionnaire provenance and privacy

Twenty anonymous adults reporting relevant experience answered the same
questionnaire twice, 7–14 days apart. Eligibility and experience are
self-reported, not independently credential-verified. Scenarios are
hypothetical; agreement does not establish calibration against observed
overtime. This is the inherited campaign, not a new release-11 collection.

`data/reliability20/responses_numeric.csv` contains pseudonymous numerical
answers and quality indicators. Approximate/unknown answers are missing
numerically; nothing is imputed or repaired. The public import audit records
the source-bundle checksum and numerical anomalies. R1 supplies planning
inputs; paired R1/R2 responses supply agreement statistics.

**Original completed PDF forms and private import records are not included.**
Public reproduction starts from the CSV. The optional
`run_reliability20.py --import-forms` branch requires original forms outside
this package and is not a self-contained public reproduction step. The numeric
CSV releases no participant names, contacts or individual session dates.

## New experiments and validation

Run commands from the extracted `ms-vrppu-fr` directory. The following checks
validate the released records, including full in-memory reproduction of the
decision experiments, and regenerate the new tables. No private forms or
downloads are needed:

```sh
python3 validation/validate_endpoint_minimax.py
python3 experiments/audit_minimax_v11.py
python3 validation/validate_endpoint_decision_v11.py
python3 validation/validate_endpoint_information_v11.py
python3 experiments/make_v11_tables.py
```

To write new experiment records, run either driver in its own fresh extraction:

```sh
python3 experiments/run_endpoint_decision_v11.py
```

```sh
python3 experiments/run_endpoint_information_v11.py
```

Drivers replace their named output JSON files. The additive driver requires
the unchanged released original JSON as its anchor and checks its hash; do
not run it after overwriting that original with a fresh driver run. Environment
metadata can change a rerun's file hash even when its numerical results agree.
The released archive preserves the original records and protocol.

The membership validator checks normality, upper semicontinuity, compact
positive cuts, strong endpoints, sentinel equality, direct Pos/Nec events,
count identities, sharpness constructions and inherited result consistency.
The decision validator independently enumerates the domain by permutations
and cuts, audits scalar endpoints and OWA, and reproduces exhaustive optima.
The additive experiment records the original-input and protocol hashes and
has its own validator; it does not replace the original decisions. Numeric
protocol powers in the manuscript are generated from these JSON records and
checked when the submission archive is built.

To regenerate tables from existing records without large solver reruns:

```sh
python3 experiments/make_tables.py
python3 experiments/make_v10_tables.py
python3 experiments/make_reliability20_tables.py
python3 experiments/make_elicited_tables.py
python3 experiments/summarize_external_v4.py
python3 experiments/summarize_external_extended.py
python3 experiments/audit_minimax_v11.py
python3 experiments/make_v11_tables.py
```

Generators write `tables/`; some also update manuscript table copies when
they detect the surrounding project. The archive contains generated tables
referenced by the accompanying main text and supplement.

## Inherited analyses and solver studies

The questionnaire analysis is reproducible from the public numerical answers:

```sh
python3 experiments/run_reliability20.py
python3 validation/validate_reliability20.py
python3 experiments/run_elicited.py --part describe
python3 experiments/run_elicited.py --part small --workers 2
python3 experiments/run_elicited.py --part extension --workers 2
python3 experiments/run_elicited.py --part quantile --workers 2
python3 validation/validate_elicited.py
python3 validation/validate_external_elicited.py
```

The following rerun inherited solver studies. Full minimax runs contain many
MILPs with explicit time limits and can take substantially longer than checks:

```sh
python3 experiments/run_reference.py --seconds 1 --out ortools_1s.json
python3 experiments/run_reference.py --seconds 10
python3 experiments/run_reference.py --engine hgs --seconds 1 --out hgs_1s.json
python3 experiments/run_reference.py --engine hgs --seconds 10 --out hgs_10s.json
python3 experiments/run_ceiling.py --workers 2
python3 experiments/run_exact_small.py --workers 2
python3 experiments/run_ablation.py --workers 2
python3 experiments/run_biobjective.py aggregators
python3 experiments/run_shapes_matched.py
python3 experiments/run_direct_fff.py
python3 validation/validate_v10.py
python3 experiments/run_endpoint_identification.py
python3 experiments/run_endpoint_minimax.py --workers 2
python3 experiments/run_endpoint_minimax.py --feasibility optimistic --workers 2
python3 experiments/prepare_amazon_v4.py
python3 experiments/run_external_v4.py --workers 2
python3 validation/verify_external_v4.py
python3 validation/validate_external_v4_independent.py
python3 experiments/run_external_extended.py --block equal --workers 2
python3 experiments/run_external_extended.py --block long --workers 2
python3 experiments/run_external_extended.py --block iterations --workers 2
python3 experiments/run_external_extended.py --block shapes --workers 2
```

Canonical instances are included. Timed comparisons require an otherwise idle
machine; timings and heuristic outputs can vary by platform. The public-route
replay is an operational-data scaffold with constructed uncertainty, not field
validation. Its inherited protocol is in `EXTERNAL_V4_PROTOCOL.md`.

## Conventions and licences

`makespan` in saved records means total representative workload. The main
minimax and new decision studies require representative feasibility; older
studies include optimistic-feasibility variants. Each protocol specifies its
domain. The in-house allowance counts fuzzy cell evaluations of a reference
120-iteration search.

Code: MIT (`LICENSE`). Synthetic instances, numerical questionnaire answers,
records and tables: CC BY 4.0 (`LICENSE-DATA`). Public Amazon source data and
derived instances retain CC BY-NC 4.0 (`data/amazon_v4/`).
