def classify_failures(claim_evaluations, metrics=None):
    failures = []

    if not claim_evaluations:
        return ["no_claims_extracted"]

    unsupported = [c for c in claim_evaluations if c["status"] == "Unsupported"]
    contradicted = [c for c in claim_evaluations if c["status"] == "Contradicted"]
    weak = [c for c in claim_evaluations if c["status"] == "Weakly supported"]

    total = len(claim_evaluations)

    unsupported_ratio = len(unsupported) / total
    contradicted_ratio = len(contradicted) / total
    weak_ratio = len(weak) / total

    # Severe contradiction
    if contradicted_ratio > 0:
        failures.append("contradicted_claims")

    # High hallucination
    if unsupported_ratio > 0.5:
        failures.append("high_hallucination_rate")

    # Partial support pattern
    if weak_ratio > 0.3 and contradicted_ratio == 0:
        failures.append("partial_support")

    # Metric-driven failures
    if metrics:
        if metrics.get("hallucination_rate", 0) > 0.5:
            failures.append("systemic_hallucination")

        if metrics.get("faithfulness", 1.0) < 0.6:
            failures.append("low_faithfulness")

    return list(set(failures))
