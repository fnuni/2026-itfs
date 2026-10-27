# Twenty-expert campaign collector

`reliability20/raccogli_e_analizza.py` reads the returned R1/R2 PDF forms
(E01-E20) of the anonymous expert campaign. Answers are located by the labels
printed on the form; literal text is preserved privately and converted to
numbers only when it is a plain number, optionally followed by a minute unit.
"non so", "nessuno", approximate values ("circa ...") and ranges are missing
numerically; no answer is repaired, imputed or reordered.

The module also provides the agreement statistics of the test-retest study:
ICC(A,1) from the two-way ANOVA with fixed sessions, Bland-Altman limits,
log-ratio limits and paired-expert percentile bootstrap (2,000 replicates,
seed 20261014).

Driver: `../experiments/run_reliability20.py` (`--import-private` re-reads the
PDF forms from the coordinator's private folders). The public numerical file is
`../data/reliability20/responses_numeric.csv`; its R1 rows also feed the
planning studies of `../experiments/run_elicited.py`.
