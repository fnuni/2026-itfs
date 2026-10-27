# Release 12.0.0 — 27 October 2026

Publication alignment: the anonymous package is published at
https://github.com/fnuni/2026-itfs under tag `v12.0.0`, without a DOI.
Repository and citation metadata have been added; experimental source,
data, result records and tables are unchanged. The manifest is regenerated
from the final submission package.

Accompanies manuscript version 12. All v11 records, sources and drivers are
unchanged. Added: certified interpolation (CI-tau) and a neutral completion
family on the exhaustive seven-stop study (`endpoint_certified_v12*`), expert
strata by self-reported area and experience band (`strata.csv`,
`reliability20_strata_v12*`), their validators and the table generator
`make_v12_tables.py`. The neutral family is one uninformative choice of
admissible completions, not an empirical uncertainty model; its protocol was
recorded before the runs and after inspection of the v11 results.

# Release 11.0.1 — 16 October 2026

This is a patch release of the code/data artifact accompanying manuscript
version 11, not a renumbering of the article.

The original heterogeneous experiment, `endpoint_decision_v11.json`, its
core source and its driver are preserved. A separate endpoint-information
experiment adds a piecewise-linear point completion fitted to the same
available endpoint grid as minimax, keeping feasibility, workload scale,
weights, preference values and the individual-risk cap identical.

The original smooth test family uses powers 0.45, 1 and 2.2. Added stress
completions use 0.1 and 10. Results from these families remain separate;
the five completions are not treated as samples from an empirical
distribution. The common least-favourable completion is reported as a
deterministic robust certificate for every method on a given grid, not as an
independent experiment favouring minimax by construction.

The extension protocol was recorded after inspecting the original experiment
and before running the added stress evaluations. It is not preregistration.
All eight original inputs and every original setting are retained, without
outcome-based case selection. Input, protocol, source and validation hashes
connect the additive results to the unchanged original records.

The patch also corrects the manuscript's description of the original powers
and generates those numerical protocol values from JSON. Packaging checks
compare the generated macros, manuscript references and recorded protocol
before creating the anonymous ZIP.

Prior qualifications remain: uncertainty and completion families are
constructed, questionnaire experience is self-reported, original completed
forms are not publicly included, most inherited solver studies were not rerun,
and 52 optimistic-sweep secondary solves lack certified tie-breaks. The
historical fallback label alone does not establish their termination reasons.

The superseded 11.0.0 upload archive is removed only after 11.0.1 passes archive
verification and the saved pre-correction snapshot is checked to contain an
identical copy of the superseded ZIP.
