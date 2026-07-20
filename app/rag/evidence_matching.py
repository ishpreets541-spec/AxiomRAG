# app/rag/evidence_matching.py

import re
from typing import List, Dict
from transformers import pipeline

# Use a pipeline as a high-level helper
from transformers import pipeline

class EvidenceMatcher:
    def __init__(self):

        self.nli = pipeline(
            "text-classification",
            model="MoritzLaurer/deberta-v3-base-mnli-fever-anli",
            return_all_scores=True
        )

    # -----------------------
    # Deterministic Guards
    # -----------------------

    def _numeric_mismatch(self, claim_text: str, chunks: List[Dict]) -> bool:
        claim_numbers = re.findall(r"\d+", claim_text)
        if not claim_numbers:
            return False

        chunk_text_combined = " ".join(c["text"] for c in chunks)
        chunk_numbers = re.findall(r"\d+", chunk_text_combined)

        for num in claim_numbers:
            if num not in chunk_numbers:
                return True

        return False

    def _strict_verb_guard(self, claim_text: str, chunks: List[Dict]) -> bool:
        claim_lower = claim_text.lower()

        if "sell" in claim_lower:
            combined = " ".join(c["text"].lower() for c in chunks)
            if "sell" not in combined:
                return True

        return False

    def _absolute_claim_guard(self, claim_text: str, chunks: List[Dict]) -> bool:
        absolute_terms = [
            "only", "always", "forever", "permanently",
            "exclusively", "entirely", "all", "none"
        ]

        claim_lower = claim_text.lower()

        if any(term in claim_lower for term in absolute_terms):
            combined = " ".join(c["text"].lower() for c in chunks)

            for term in absolute_terms:
                if term in claim_lower and term not in combined:
                    return True

        return False

    def _negation_conflict(self, claim_text: str, chunks: List[Dict]) -> bool:
        claim_lower = claim_text.lower()

        if "never" in claim_lower or "no " in claim_lower:
            for c in chunks:
                chunk_text = c["text"].lower()
                if "collect" in chunk_text or "gather" in chunk_text:
                    return True

        return False

    # -----------------------
    # Retrieval Candidate Filter
    # -----------------------

    def _candidate_chunks(self, claim: str, chunks: List[Dict], top_k: int = 5):
        claim_tokens = set(re.findall(r"\w+", claim.lower()))
        scored = []

        for c in chunks:
            chunk_tokens = set(re.findall(r"\w+", c["text"].lower()))
            overlap = len(claim_tokens & chunk_tokens)

            if overlap > 0:
                scored.append((overlap, c))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored[:top_k]]

    # -----------------------
    # MNLI Classification
    # -----------------------

    def _mnli_scores(self, claim_text: str, chunk_text: str):

        sequence = chunk_text + " </s></s> " + claim_text

        results = self.nli(sequence)

        scores = {}

        # Case 1: Nested list (some MNLI models)
        if isinstance(results[0], list):
            scores_list = results[0]

        # Case 2: Direct list of dicts (DeBERTa style)
        elif isinstance(results[0], dict):
            scores_list = results

        else:
            raise ValueError(f"Unexpected MNLI output format: {results}")

        for item in scores_list:
            if isinstance(item, dict):
                scores[item["label"].upper()] = item["score"]

        return scores

    # -----------------------
    # Main Match Function
    # -----------------------

    def match(self, claim: Dict, retrieved_chunks: List[Dict]) -> Dict:

        candidates = self._candidate_chunks(claim["text"], retrieved_chunks)

        if not candidates:
            return {
                "claim_id": claim["id"],
                "status": "Unsupported",
                "confidence": 0.0,
                "evidence": [],
                "rationale": "No retrieved chunk addresses the claim."
            }

        # ---- Deterministic Guards ----

        if self._numeric_mismatch(claim["text"], candidates):
            return {
                "claim_id": claim["id"],
                "status": "Unsupported",
                "confidence": 0.9,
                "evidence": [],
                "rationale": "Claim contains numeric detail not found in evidence."
            }

        if self._negation_conflict(claim["text"], candidates):
            return {
                "claim_id": claim["id"],
                "status": "Contradicted",
                "confidence": 0.9,
                "evidence": [c["chunk_id"] for c in candidates],
                "rationale": "Claim negates behavior explicitly described in evidence."
            }

        if self._strict_verb_guard(claim["text"], candidates):
            return {
                "claim_id": claim["id"],
                "status": "Unsupported",
                "confidence": 0.9,
                "evidence": [],
                "rationale": "Claim uses commercial action not supported in evidence."
            }

        if self._absolute_claim_guard(claim["text"], candidates):
            return {
                "claim_id": claim["id"],
                "status": "Unsupported",
                "confidence": 0.9,
                "evidence": [],
                "rationale": "Claim contains absolute qualifier not supported in evidence."
            }

        # ---- MNLI Decision ----

        best_entailment = 0
        best_contradiction = 0
        best_chunk = None

        for c in candidates:
            scores = self._mnli_scores(claim["text"], c["text"])

            entail_score = scores.get("ENTAILMENT", 0)
            contradiction_score = scores.get("CONTRADICTION", 0)

            if contradiction_score > best_contradiction:
                best_contradiction = contradiction_score
                best_chunk = c

            if entail_score > best_entailment:
                best_entailment = entail_score
                best_chunk = c

        if best_contradiction > 0.6:
            return {
                "claim_id": claim["id"],
                "status": "Contradicted",
                "confidence": round(best_contradiction, 3),
                "evidence": [best_chunk["chunk_id"]],
                "rationale": "Evidence contradicts the claim."
            }

        if best_entailment > 0.6:
            return {
                "claim_id": claim["id"],
                "status": "Supported",
                "confidence": round(best_entailment, 3),
                "evidence": [best_chunk["chunk_id"]],
                "rationale": "Evidence entails the claim."
            }

        return {
            "claim_id": claim["id"],
            "status": "Unsupported",
            "confidence": 0.5,
            "evidence": [],
            "rationale": "Evidence does not strongly support the claim."
        }