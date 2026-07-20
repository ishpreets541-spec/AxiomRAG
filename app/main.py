# app/main.py
from app.rag.query_engine import answer_query
from app.rag.metrics import compute_metrics
from app.schemas.response import AuditResult, Claim, Evidence, Metrics

def main(query: str, answer: str = "", llm=None) -> AuditResult:
    # Run the pipeline (llm parameter unused, as we use query_engine with Ollama)
    result = answer_query(query, answer)
    claims_data = result.get("claims", [])
    evals = result.get("claim_evaluations", [])

    # Build list of Claim models
    claims = [Claim(id=c["id"], text=c["text"]) for c in claims_data]

    # Build list of Evidence models (mapping keys)
    evidence = []
    for ev in evals:
        evidence.append(Evidence(
            claim_id=ev["claim_id"],
            chunk_ids=ev.get("evidence", []),
            support_level=ev["status"],
            rationale=ev.get("rationale", ""),
            confidence=ev.get("confidence", 0.0)
        ))

    # Compute metrics
    metrics_dict = compute_metrics(evals)
    metrics = Metrics(**metrics_dict)

    # Determine verdict
    statuses = {e.support_level.lower() for e in evidence}
    if not evidence:
        verdict = "SAFE"
    elif all(s in ("supported", "weak") for s in statuses):
        verdict = "SAFE"
    elif any(s in ("supported", "weak") for s in statuses) and any(s in ("unsupported", "contradicted") for s in statuses):
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "UNSAFE"

    return AuditResult(
        claims=claims,
        evidence=evidence,
        metrics=metrics,
        verdict=verdict,
        sources=result.get("sources", []),
        generated_answer=result.get("generated_answer", ""),
        retrieved_chunks=result.get("retrieved_chunks", []),
    )
