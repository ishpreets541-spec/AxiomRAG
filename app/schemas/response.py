# app/schemas/response.py
from typing import List
from pydantic import BaseModel
from typing import Optional
from pydantic import Field

class Claim(BaseModel):
    id: str
    text: str

class Evidence(BaseModel):
    claim_id: str
    chunk_ids: List[str]
    support_level: str  # supported | weak | unsupported | contradicted
    rationale: str
    confidence: float

class Metrics(BaseModel):
    faithfulness: Optional[float] = None
    hallucination_rate: Optional[float] = None
    evidence_coverage: Optional[float] = None
    decision_consistency: Optional[float] = None

class AuditResult(BaseModel):
    claims: List[Claim]
    evidence: List[Evidence]
    metrics: Metrics
    verdict: str  # SAFE | PARTIALLY_SUPPORTED | UNSAFE
    sources: List[str] = Field(default_factory=list)
    generated_answer: str = ""
    retrieved_chunks: List[dict] = Field(default_factory=list)
