# app/rag/query_engine.py
import logging
import os
import time
import re
from typing import Any, Dict, List
import requests
from requests import RequestException

from app.rag.retriever import retrieve_chunks
from app.rag.claim_extractor import ClaimExtractor
from app.rag.evidence_matching import EvidenceMatcher
from app.rag.metrics import compute_metrics

# OLLAMA CONFIG
LLM_ENDPOINT = "https://unfair-folk-knoll.ngrok-free.dev/api/generate"
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))
OLLAMA_CONNECT_TIMEOUT_SEC = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SEC", "15"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))
OLLAMA_RETRY_BACKOFF_SEC = float(os.getenv("OLLAMA_RETRY_BACKOFF_SEC", "2"))

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "20"))

logger = logging.getLogger(__name__)
claim_extractor = ClaimExtractor()
evidence_matcher = EvidenceMatcher()

def _resolve_ollama_url() -> str:
    return LLM_ENDPOINT

def call_ollama(prompt: str) -> str:
    url = _resolve_ollama_url()
    last_exc: Exception | None = None
    for attempt in range(1, OLLAMA_RETRIES + 2):
        try:
            response = requests.post(
                url,
                json={
                    "model": "qwen2.5:0.5b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_TIMEOUT_SEC)
            )
            response.raise_for_status()
            data = response.json()

            # Support common response shapes from Ollama-compatible endpoints.
            text = ""
            if isinstance(data, dict):
                if isinstance(data.get("response"), str):
                    text = data["response"]
                elif isinstance(data.get("text"), str):
                    text = data["text"]
                elif isinstance(data.get("output"), str):
                    text = data["output"]
                elif isinstance(data.get("message"), dict) and isinstance(data["message"].get("content"), str):
                    text = data["message"]["content"]
                elif isinstance(data.get("choices"), list) and data["choices"]:
                    first = data["choices"][0]
                    if isinstance(first, dict):
                        if isinstance(first.get("text"), str):
                            text = first["text"]
                        elif isinstance(first.get("message"), dict) and isinstance(first["message"].get("content"), str):
                            text = first["message"]["content"]

            text = text.strip()
            if not text:
                raise ValueError(f"LLM response was empty. Raw payload keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            return text
        except RequestException as exc:
            last_exc = exc
            if attempt <= OLLAMA_RETRIES:
                time.sleep(OLLAMA_RETRY_BACKOFF_SEC * attempt)
                continue
            break
    raise RuntimeError(f"Failed to contact LLM endpoint at {url}: {last_exc}") from last_exc

def _build_context(chunks: List[Dict[str, Any]]) -> str:
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        source = (chunk.get("metadata") or {}).get("file_name") or chunk.get("source") or "unknown"
        context_lines.append(f"Source {i} ({source}): {text}")
    return "\n\n".join(context_lines)

def generate_answer_from_documents(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context = _build_context(retrieved_chunks)
    prompt = f"""
You are a careful assistant. Answer the user's question only using the provided document context.
If the context does not contain enough information, say that you could not find enough information in the documents.
Do not mention that you are using context or retrieved documents.
Keep the answer concise and factual.
Include inline citations with chunk ids like [chunk_12] when making factual statements.

Question:
{query}

Document context:
{context}

Answer:
""".strip()
    try:
        generated = call_ollama(prompt)
        if generated and generated.strip():
            return generated.strip()
    except Exception:
        pass

    # Fallback: deterministic extractive answer so auditing can still proceed.
    top_snippets = []
    for chunk in retrieved_chunks[:3]:
        snippet = (chunk.get("text") or "").strip()
        if snippet:
            top_snippets.append(snippet[:300])
    if not top_snippets:
        return "I could not find enough information in the retrieved documents."
    return " ".join(top_snippets)


def _fallback_claims_from_answer(answer: str) -> List[Dict[str, str]]:
    # Deterministic fallback when LLM claim extraction is unavailable or empty.
    parts = re.split(r"(?<=[.!?])\s+", (answer or "").strip())
    claims = []
    for part in parts:
        text = part.strip()
        if len(text) < 20:
            continue
        claims.append({"id": f"C{len(claims) + 1}", "text": text})
        if len(claims) >= 8:
            break
    return claims

def _safe_chunk_text(chunk: Dict[str, Any]) -> str:
    chunk_id = chunk.get("chunk_id", "unknown")
    text = chunk.get("text", "")
    return f"[{chunk_id}] {text}"

def _convert_to_evidence_objects(claim_evaluations: List[Dict]) -> List[Dict]:
    status_mapping = {
        "Supported": "supported",
        "Weakly supported": "weak",
        "Unsupported": "unsupported",
        "Contradicted": "contradicted",
    }

    evidence_objects = []

    for c in claim_evaluations:
        evidence_objects.append({
            "claim_id": c["claim_id"],
            "chunk_ids": c.get("evidence", []),
            "support_level": status_mapping.get(c["status"], "unsupported"),
            "rationale": c.get("rationale", ""),
            "confidence": float(c.get("confidence", 0.0)),
        })

    return evidence_objects


def _limit_context(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    limited = []
    total = 0
    for chunk in chunks[:MAX_CHUNKS]:
        text = chunk.get("text", "")
        if not isinstance(text, str):
            continue
        if total + len(text) > MAX_CONTEXT_CHARS:
            break
        limited.append(chunk)
        total += len(text)
    return limited

def answer_query(query: str, answer: str = "") -> Dict[str, Any]:
    # 1️⃣ Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(query)

    if not retrieved_chunks:
        return {
            "query": query,
            "answer": answer,
            "generated_answer": answer,
            "claims": [],
            "claim_evaluations": [],
            "retrieved_chunks": [],
            "sources": [],
        }

    retrieved_chunks = _limit_context(retrieved_chunks)

    generated_answer = answer.strip() if answer and answer.strip() else generate_answer_from_documents(query, retrieved_chunks)

    # 2️⃣ Extract claims from the generated answer
    claim_data = claim_extractor.extract(generated_answer)
    claims = claim_data.get("claims", [])
    if not claims:
        claims = _fallback_claims_from_answer(generated_answer)

    # 3️⃣ Match evidence
    claim_evaluations = []
    for claim in claims:
        eval_res = evidence_matcher.match(claim, retrieved_chunks)
        claim_evaluations.append(eval_res)

    # 4️⃣ Compute metrics
    metrics = compute_metrics(claim_evaluations)

    hallucination_rate = metrics.get("hallucination_rate")
    faithfulness = metrics.get("faithfulness")

    evidence_objects = _convert_to_evidence_objects(claim_evaluations)

    # 5️⃣ Verdict logic
    if not claim_evaluations:
        verdict = "UNSAFE"
    elif hallucination_rate is not None and hallucination_rate > 0.5:
        verdict = "UNSAFE"
    elif faithfulness is not None and faithfulness < 0.7:
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "SAFE"

    return {
        "query": query,
        "answer": generated_answer,
        "generated_answer": generated_answer,
        "claims": claims,
        "claim_evaluations": claim_evaluations,
        "evidence": evidence_objects,
        "metrics": metrics,
        "verdict": verdict,
        "retrieved_chunks": retrieved_chunks,
        "sources": list({(c.get("metadata") or {}).get("file_name", "unknown") for c in retrieved_chunks}),
    }
