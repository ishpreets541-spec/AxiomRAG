from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.main import main as audit_main
from app.utils.llm import llm

app = FastAPI(title="RAG Reliability Auditor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request Schema ----------
class AuditRequest(BaseModel):
    query: str
    answer: str = ""

# ---------- Root ----------
@app.get("/")
def root():
    return {"message": "RAG Reliability Auditor API running"}

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Audit ----------
@app.post("/audit")
def audit(request: AuditRequest):
    result = audit_main(request.query, request.answer, llm)
    return result.dict()
