"""Audit all frozen external plans without importing any model or solver code.

Run: python3 validation/validate_external_v4_independent.py

The only inputs are the derived public modal matrices, the declared scenario
spread rules, and the saved plans. References for Pos/Nec, optimistic feasibility,
cell-anonymous OWA (FFF) and workload are implemented directly below. This verifies arithmetic
and coverage in the declared scenario; it does not validate the constructed
fuzzy memberships against operational uncertainty or certify an optimum.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/amazon_v4/instances.json"
RESULT = ROOT / "results/external_v4.json"
OUTPUT = ROOT / "results/external_v4_independent_validation.json"
TOLERANCE = 1e-9


def travel_triple(record, i, j):
    if i == j:
        return (0.0, 0.0, 0.0)
    modal = record["travel_minutes"][i][j]
    zones = record["zones"]
    left = 0.10 + 0.05 * ((i + 2 * j) % 3)
    cross = bool(zones[i] and zones[j] and zones[i] != zones[j])
    right = 0.20 + 0.10 * ((3 * i + j) % 4) + 0.10 * cross
    return modal * (1.0 - left), modal, modal * (1.0 + right)


def service_triple(record, i):
    if i == 0:
        return (0.0, 0.0, 0.0)
    modal = record["service_minutes"][i]
    count = record["package_counts"][i]
    return (modal * (1.0 - (0.10 + 0.025 * (count % 3))), modal,
            modal * (1.20 + 0.05 * min(count, 5)))


def route_triple(record, sequence):
    totals = [0.0, 0.0, 0.0]
    previous = 0
    for node in sequence:
        travel = travel_triple(record, previous, node)
        service = service_triple(record, node)
        for index in range(3):
            totals[index] += travel[index] + service[index]
        previous = node
    returning = travel_triple(record, previous, 0)
    return tuple(totals[index] + returning[index] for index in range(3))


def triangular_overtime(a, b, c, horizon):
    # Direct sup on {X <= horizon} and on {X > horizon}, including crisp cases.
    possibility = (0.0 if horizon < a else
                   1.0 if horizon >= b else (horizon - a) / (b - a))
    complement_sup = (0.0 if horizon >= c else
                      1.0 if horizon < b else (c - horizon) / (c - b))
    necessity = 1.0 - complement_sup
    return 1.0 - (possibility + necessity) / 2.0


def main():
    records = {r["id"]: r for r in json.loads(DATA.read_text())["records"]}
    payload = json.loads(RESULT.read_text())
    plans = payload["raw"]
    errors = []
    largest_error = 0.0
    largest_final_error = 0.0
    numeric_checks = 0
    stages_checked = 0

    def equal(label, actual, expected):
        nonlocal largest_error, largest_final_error, numeric_checks
        numeric_checks += 1
        error = abs(actual - expected)
        largest_error = max(largest_error, error)
        if "/OR_stage/" not in label:
            largest_final_error = max(largest_final_error, error)
        if error > TOLERANCE:
            errors.append(dict(check=label, actual=actual, expected=expected))

    def audit(record, output, label, alpha=None):
        cells = output["solution"]
        if len(cells) != record["K"] or any(len(r) != record["P"] for r in cells):
            errors.append(dict(check=label + "/shape"))
            return
        tasks = [task for row in cells for seq in row for task in seq]
        if sorted(tasks) != list(range(1, record["n"] + 1)):
            errors.append(dict(check=label + "/coverage"))
            return
        risk_rows = []
        workload = 0.0
        feasible = True
        for vehicle, row in enumerate(cells):
            risks = []
            for shift, sequence in enumerate(row):
                a, b, c = route_triple(record, sequence)
                risk = triangular_overtime(a, b, c, record["L"])
                risks.append(risk)
                workload += b
                feasible = feasible and a <= record["L"] + TOLERANCE
                equal(f"{label}/R/{vehicle}/{shift}", risk, output["R"][vehicle][shift])
            risk_rows.append(risks)
        P = record["P"]
        ordered = sorted((risk for row in risk_rows for risk in row), reverse=True)
        n_cells = len(ordered)
        weights = [2.0 * (n_cells - j) / (n_cells * (n_cells + 1)) for j in range(n_cells)]
        fff = sum(weight * risk for weight, risk in zip(weights, ordered))
        maximum = max(risk for row in risk_rows for risk in row)
        equal(label + "/total_modal_workload", workload, output["makespan"])
        equal(label + "/FFF", fff, output["fff"])
        equal(label + "/worst_cell_credibility", maximum, output["max_risk"])
        if feasible != output["feasible"]:
            errors.append(dict(check=label + "/optimistic_feasibility"))
        if "ceiling_met" in output:
            ceiling_met = feasible and maximum <= output["ceiling"] + TOLERANCE
            if ceiling_met != output["ceiling_met"]:
                errors.append(dict(check=label + "/ceiling_met"))
        if alpha is not None and output.get("J") is not None:
            objective = alpha * workload / record["workload_scale"] + (1.0 - alpha) * fff
            equal(label + "/J", objective, output["J"])

    for index, plan in enumerate(plans):
        record = records[plan["instance"]]
        label = f"final/{index}/{plan['instance']}/{plan['method']}"
        audit(record, plan, label, plan["alpha"])
        equal(label + "/L", record["L"], plan["L"])
        equal(label + "/workload_scale", record["workload_scale"], plan["workload_scale"])
        for stage_index, stage in enumerate(plan["OR_stages"]):
            if stage.get("solution") is not None:
                audit(record, stage, label + f"/OR_stage/{stage_index}", plan["alpha"])
                stages_checked += 1

    methods = Counter(plan["method"] for plan in plans)
    expected_methods = {"OR-sweep": 72, "CC-all": 72, "CC-modal": 72, "OR+CC": 72}
    if methods != expected_methods or len(plans) != 288:
        errors.append(dict(check="expected_288_balanced_plans", actual=dict(methods)))
    summary = dict(
        status="passed" if not errors else "failed",
        independent_of_model_solver_and_fuzzy_implementation=True,
        inputs="Derived public modal matrices, declared scenario spreads, saved complete plans",
        final_plans_checked=len(plans), final_plans_by_method=dict(methods),
        OR_stage_plans_checked=stages_checked, numeric_checks=numeric_checks,
        absolute_tolerance=TOLERANCE, max_absolute_numeric_difference=largest_error,
        max_absolute_final_plan_numeric_difference=largest_final_error,
        failures=len(errors), errors=errors,
        assertions=["complete_task_coverage", "K_by_P_shape", "modal_workload",
                    "Pos_Nec_credibility", "FFF_cell_OWA", "worst_cell_credibility",
                    "optimistic_feasibility", "recorded_ceiling_status", "J_alpha"],
        data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        results_sha256=hashlib.sha256(RESULT.read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
