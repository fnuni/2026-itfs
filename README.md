# MS-VRPPU-FR — reproducibility package 13.0.0

**Common-Membership Regret Certificates for Fleet Routing from Finite Endpoints**

Code, numerical data, protocols, result records and independent internal validators accompany the complete study.
The release is available at <https://github.com/fnuni/2026-itfs/releases/tag/v13.0.0>.
The anonymous submission ZIP and this tagged source tree contain the same manifest-verified payload. No DOI is assigned.

## Contribution and scope

The paper studies routing from finite fuzzy-duration endpoints, including sharp endpoint bounds, robust ordered-weighted
objectives, certified interpolation (CI) and common-membership worst-case regret. The regret representation uses classical
random-set upper expectation. The routing-specific contribution preserves elicited strong endpoints and reduces elementary-route
strict violation masks to exact bounded bipartite circulation. A primitive uncertainty is an individual stop-to-stop arc or
service duration occurring at most once in each plan. Shared latent factors and repeated road-segment variables are outside
this network theorem. Generic regret, random-set natural extension and flow integrality are credited to prior work.

`src/exact_flow_oracle.py` uses arbitrary-precision integer feasibility. Stored endpoint values and the explicit numerical
horizon `Ldagger=float(L+1e-9)` are exact dyadic inputs. OWA, grid and preference parameters use their declared rational
values; workload and normalizer doubles are fixed dyadic objective inputs. Nominal PL inversion and completion scoring
remain numerical. The nominal constraint applies to those fixed evaluated scores.

## Complete-domain regret experiments

`results/endpoint_regret_v13_n7.json`, `endpoint_regret_v13_n8.json` and `endpoint_regret_v13_n9.json` each contain 96 settings:
all eight inputs, two profiles, two policy domains and three preferences. Every eligible challenger belongs to the complete
domain, including plans outside the selected plan's robust or nominal budgets. Samples only order the search; exact endpoint
bounds and flow witnesses certify every exclusion.

The RB rule permits at most 2% extra worst-case cost relative to minimax and at most 1% extra nominal score relative to CI.
It never raises certified worst-case regret relative to CI. Material improvements (more than 1e-9 absolute objective units)
occur in 31/96, 24/96 and 35/96 settings at 7, 8 and 9 tasks. Complete nine-task domains contain 3,507,840 plans per input before
eligibility filtering. These are small exhaustive routing domains; they do not establish operational-scale routing optima.
At nine tasks, eight further rational improvements are numerical-scale differences and are excluded from the 35 material wins.

PL has the lowest mean regret on the smooth and neutral constructed completion families. These means are not operational
expectations. Stress families are reported separately. The study is exploratory, not preregistered: design choices followed
exploratory analyses; the 8/9-task protocol was fixed before those outcomes, after the 7-task results.

## Reproduce the exact certificates

Python 3.10+ and NumPy suffice for the flow certificates. `results/environment_v13.json` records the environment.
Run these commands from the package root:

```sh
python3 validation/validate_regret_math_v13.py
python3 experiments/validate_regret_v13.py --n 7
python3 experiments/validate_regret_v13.py --n 8
python3 experiments/validate_regret_v13.py --n 9
python3 experiments/make_v13_tables.py
```

The mathematical audit compares 384 rational masks with independent exact vertex enumeration, checks 72 complete membership
constructions through closed-cut sup–min arithmetic, and tests tiny strict margins, equality, empty routes, repeated-leg
rejection and the sharp scaling factor. Certificate validators replay 1,152 method certificates and 1,809 candidate witnesses,
checking every relevant challenger and the exact bounds excluding the others. They separately verify PL/MM/CI choices,
worst-challenger eligibility and both RB budgets. These are independent internal implementations, not external review.
Full nine-task validation can take several minutes and about 2 GB peak memory. Timings vary by machine.

Fresh runs preserve the canonical released records under `results/`:

```sh
python3 experiments/run_seven_stop.py --n 7
python3 experiments/make_regret_summary.py --n 7
```

Use `--n 8` or `--n 9` analogously. Fresh per-case and combined outputs are written beside the driver under `experiments/`.
The validator prefers fresh combined outputs there when present, otherwise it reads the canonical `results/` copies.
Validation reports are written beside each validator. Compare rational decisions and certificate values, not elapsed times
or complete JSON hashes. Original per-case records and hashes are under `results/endpoint_regret_v13_parts/`;
`results/validation_n7.json`, `validation_n8.json` and `validation_n9.json` audit the canonical combined records.

## Supporting studies

The package also contains finite-endpoint decision experiments, same-information PL and triangular conventions,
CI tolerance sweeps, neutral completions, synthetic benchmarks, ceiling comparisons, direct-objective references,
minimax audits, public-route replay and a twenty-person questionnaire. File hashes and protocol links identify their
inputs and implementations. Protocols specify which studies use representative or optimistic feasibility.

A descriptive launcher exposes their checks without requiring knowledge of internal file identifiers:

```sh
python3 experiments/reproduce_supporting.py list
python3 experiments/reproduce_supporting.py membership
python3 experiments/reproduce_supporting.py endpoint-decisions
python3 experiments/reproduce_supporting.py endpoint-information
python3 experiments/reproduce_supporting.py certified-interpolation
python3 experiments/reproduce_supporting.py expert-strata
python3 experiments/reproduce_supporting.py minimax-audit
python3 experiments/reproduce_supporting.py questionnaire
```

The smooth completions use powers 0.45, 1 and 2.2; separate stress powers are 0.1 and 10. The neutral family uses 30 draws
per input with independent leg/interval exponents log-uniform on [0.1, 10]. All are constructed evaluation scenarios,
not samples from an empirical uncertainty law. Repeated preferences, grids, weights and completions are not independent observations.
The triangular convention is deliberately misspecified for the nontriangular endpoints; its comparison proves no universal superiority.

The minimax audit distinguishes 30 coordinate geometries, 50 family/seed combinations, 200 context/horizon settings and
600 preference comparisons. All 600 direct minimax optima and all principal-study sweep tie-breaks are certified.
The optimistic-feasibility variant has 52 uncertified secondary solves: 444 comparisons have fully certified sweeps
(287 losses); 156 comparisons are exploratory (109 observed losses). A historical solver fallback label cannot establish
that each uncertified solve timed out. The certified subset is not a random sample.

Table regeneration and public-route checks are also available through the launcher:

```sh
python3 experiments/reproduce_supporting.py tables
python3 experiments/reproduce_supporting.py external-replay
```

Rerunning solver studies is more expensive than checking saved certificates. Entry points include
`experiments/run_endpoint_minimax.py`, `run_reference.py`, `run_ceiling.py`, `run_exact_small.py`, `run_elicited.py`,
`run_external_v4.py` and `run_external_extended.py`; consult their `--help` options and recorded protocols.
Some drivers replace their named outputs. Use a fresh extraction for solver reruns and retain the canonical records for
hash-based checks. Environment metadata can change file hashes even when numerical results agree.

## Questionnaire and privacy

Twenty anonymous adults reporting relevant experience answered the same hypothetical questionnaire twice, 7–14 days apart.
Eligibility and experience are self-reported. Agreement does not establish calibration against observed overtime.
`data/reliability20/responses_numeric.csv` contains pseudonymous numerical answers and quality indicators;
`strata.csv` supplies coarse self-reported area and experience bands. Approximate or unknown answers remain missing;
nothing is imputed or repaired. R1 supplies planning inputs; paired R1/R2 responses supply agreement statistics.

Original completed PDF forms and private import records are excluded. Public reproduction starts from the numerical CSV.
The optional form-import branch requires private originals and is not a public reproduction step. The CSV contains
no participant names, contact details or individual session dates.

## Dependencies, conventions and limitations

Install NumPy for exact-regret validation; `requirements.txt` lists all workflow dependencies. HiGHS is needed for the
MILP studies; OR-Tools and hygese are optional routing engines. Recorded environments identify tested versions.
Do not infer identical timing or heuristic output on another platform. Run different solver engines in separate processes.

`makespan` in saved records means total representative workload. The allowance counts fuzzy-cell evaluations of a
120-iteration reference search. Synthetic uncertainty and completion families, homogeneous vehicles and uncoupled shifts
limit transfer to operations. Public-route replay uses an operational-data scaffold with constructed uncertainty;
it is not field validation. Exact certificates concern the represented rational/dyadic instance, not physical uncertainty
or calibrated overtime probabilities. Eighteen-task scaled-copy studies are distinct from heterogeneous complete-domain results.

## Layout and licences

`src/`: models and solvers; `experiments/`: drivers and table generators; `validation/`: independent checks;
`data/`: canonical inputs and numerical questionnaire; `results/`: records and protocols; `tables/`: generated LaTeX;
`elicitation/`: questionnaire reader and agreement statistics. `PACKAGE_MANIFEST.json` records every payload SHA-256.
Internal record identifiers are stable so recorded hashes and cross-file provenance remain reproducible.

Code: MIT (`LICENSE`). Synthetic instances, numerical questionnaire answers, records and tables: CC BY 4.0 (`LICENSE-DATA`).
Public Amazon inputs and derived instances retain CC BY-NC 4.0 (`data/amazon_v4/`).
