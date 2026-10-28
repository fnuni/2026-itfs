#!/usr/bin/env python3
"""Descriptive entry points for supporting checks and table regeneration."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "membership": ["validation/validate_endpoint_minimax.py"],
    "endpoint-decisions": ["validation/validate_endpoint_decision_v11.py"],
    "endpoint-information": ["validation/validate_endpoint_information_v11.py"],
    "certified-interpolation": ["validation/validate_endpoint_certified_v12.py"],
    "expert-strata": ["validation/validate_reliability20_strata_v12.py"],
    "minimax-audit": ["experiments/audit_minimax_v11.py"],
    "questionnaire": ["validation/validate_reliability20.py"],
    "external-replay": ["validation/verify_external_v4.py",
                        "validation/validate_external_v4_independent.py"],
    "tables": ["experiments/make_tables.py", "experiments/make_v10_tables.py",
               "experiments/make_reliability20_tables.py", "experiments/make_elicited_tables.py",
               "experiments/summarize_external_v4.py", "experiments/summarize_external_extended.py",
               "experiments/audit_minimax_v11.py", "experiments/make_v11_tables.py",
               "experiments/make_v12_tables.py", "experiments/make_v13_tables.py"],
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["list", *STUDIES])
    args = parser.parse_args()
    if args.study == "list":
        print("\n".join(STUDIES))
        return
    for relative in STUDIES[args.study]:
        subprocess.run([sys.executable, str(ROOT / relative)], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
