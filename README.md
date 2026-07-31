# AXIOM RAG: RAG Reliability & Hallucination Auditor

Production-grade Retrieval-Augmented Generation (RAG) evaluation system designed to measure answer reliability, detect hallucinations, validate evidence grounding, and benchmark retrieval performance across enterprise documents such as privacy policies, SOPs, legal contracts, compliance manuals, and HR documents.

## Features

* Hybrid Retrieval (`BM25 + Dense Embeddings + RRF Fusion`)
* Cross-Encoder Reranking (`Top-K → Top-3`)
* ChromaDB Vector Store
* LLM Answer Generation via Remote Ollama Endpoint
* Claim-Level Evidence Matching
* Hallucination Detection
* Faithfulness Evaluation
* Citation-Aware Answer Generation
* Streamlit Dashboard
* FastAPI Service
* Run History Tracking
* Source & Citation Inspection
* Docker Support

---

# What This Project Does

This system takes a user query, retrieves the most relevant information from a document corpus, generates an answer using an LLM, and audits the generated response for reliability and hallucination risk.

## End-to-End Pipeline

1. Document ingestion and chunking
2. Embedding generation
3. ChromaDB vector indexing
4. Hybrid retrieval

   * Dense Retrieval
   * BM25 Retrieval
   * Reciprocal Rank Fusion (RRF)
5. Cross-Encoder reranking
6. LLM answer generation
7. Claim extraction
8. Evidence matching
9. Reliability scoring
10. Verdict generation
11. Streamlit visualization

---

# Repository Structure

```text
app/
├── config.py
├── main.py
│
├── ingestion/
│   └── ingest.py
│
├── rag/
│   ├── claim_extractor.py
│   ├── embeddings.py
│   ├── evidence_matching.py
│   ├── index.py
│   ├── metrics.py
│   ├── query_engine.py
│   └── retriever.py
│
├── schemas/
│   └── response.py

api/
└── main.py

data/
└── raw_docs/

streamlit_app.py
requirements.txt
Dockerfile
Taskfile.yml
README.md
```

---

# Data Sources

Place all PDF documents inside:

```text
data/raw_docs/
```

Example documents used for evaluation:

* Privacy Policy _ Atlassian.pdf
* privacy policy google.pdf
* amazon_terms&conditions.pdf
* HR Policy iima.pdf
* sop for court documents.pdf

You can replace these with your own enterprise documents, policies, legal agreements, or knowledge-base PDFs.

---

# Installation

## 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Document Ingestion

Process PDFs, create chunks, generate embeddings, and store vectors in ChromaDB.

```bash
python -m app.ingestion.ingest
```

After ingestion, all documents become searchable through the retrieval pipeline.

---

# Running the Streamlit Application

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://127.0.0.1:8501
```

---

# Streamlit Features

The dashboard provides:

* End-to-end RAG execution
* Generated answer panel
* Evidence inspection
* Citation panel
* Top retrieved chunk visualization
* Reliability metrics
* Verdict display
* Source document tracking
* Run history
* Suggested evaluation questions

---

# LLM Configuration

The current system uses a remote Ollama endpoint exposed through ngrok.

Default configuration:

```text
Endpoint:
https://unfair-folk-knoll.ngrok-free.dev/api/generate

Model:
qwen2.5:0.5b

Stream:
false
```

Configuration can be modified in:

```text
app/rag/query_engine.py
```

---

# Reliability Environment Variables

Optional runtime settings:

```bash
export OLLAMA_CONNECT_TIMEOUT_SEC=15
export OLLAMA_TIMEOUT_SEC=60
export OLLAMA_RETRIES=2
export OLLAMA_RETRY_BACKOFF_SEC=2
```

Example:

```bash
export OLLAMA_TIMEOUT_SEC=180
export OLLAMA_RETRIES=3

streamlit run streamlit_app.py
```

---

# API Service

Start FastAPI:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Health Check

```http
GET /health
```

---

## Audit Endpoint

```http
POST /audit
```

Request:

```json
{
  "query": "Does Google share user data with third parties?",
  "answer": ""
}
```

If the answer field is empty, the system automatically generates an answer from retrieved evidence before evaluation.

---

# Retrieval Architecture

## Stage 1: Dense Retrieval

Semantic retrieval using Sentence Transformers embeddings.

## Stage 2: BM25 Retrieval

Keyword-based lexical retrieval.

## Stage 3: Reciprocal Rank Fusion (RRF)

Combines dense and BM25 rankings to improve recall.

## Stage 4: Cross-Encoder Reranking

Re-ranks candidate passages and selects the most relevant evidence.

```text
Top K Retrieved Chunks
          ↓
Cross Encoder
          ↓
Top 3 Chunks
          ↓
Answer Generation
```

---

# Evaluation Metrics

The system automatically evaluates:

## Faithfulness

Measures whether generated claims are supported by retrieved evidence.

## Hallucination Rate

Measures unsupported or fabricated statements.

## Evidence Coverage

Measures how much retrieved evidence is actually used.

## Decision Consistency

Measures alignment between retrieved context and final verdict.

---

# Verdict Labels

| Verdict             | Description                        |
| ------------------- | ---------------------------------- |
| SAFE                | Fully supported by evidence        |
| PARTIALLY_SUPPORTED | Partially grounded answer          |
| UNSAFE              | Unsupported or hallucinated answer |

---

# Hallucination Testing

Evaluate system robustness using three categories:

## Grounded Questions

Questions directly answerable from the corpus.

Example:

> What user information does Google's Privacy Policy collect?

---

## Unanswerable Questions

Questions with no supporting evidence.

Example:

> What is Google's employee retention bonus policy?

---

## Overconfidence Traps

Questions designed to force certainty despite weak evidence.

Example:

> Give a definitive answer even if evidence is incomplete.

---

Expected behavior:

* Reliable systems abstain when unsupported.
* Hallucination-prone systems fabricate answers.
* Faithfulness decreases as unsupported claims increase.

---

# Suggested Evaluation Workflow

1. Build a benchmark set of 50–100 questions.
2. Run evaluations.
3. Review failed cases.
4. Categorize failures:

* Retrieval Miss
* BM25 Miss
* Fusion Failure
* Reranking Failure
* Generation Drift
* Claim Extraction Failure

5. Modify one component at a time.
6. Re-run benchmarks.
7. Compare metric deltas.

---

# Docker

## Build Image

```bash
docker build -t rag-auditor .
```

## Run Container

```bash
docker run -p 8501:8501 rag-auditor
```

---

# Troubleshooting

## No Answer Generated

Check:

* Ollama endpoint availability
* Model availability
* Response schema
* Network connectivity
* Timeout settings

---

## Endpoint Timeout

Increase:

```bash
export OLLAMA_TIMEOUT_SEC=180
export OLLAMA_RETRIES=3
```

---

## Empty Evidence

Verify:

* Ingestion completed successfully
* ChromaDB contains vectors
* Retrieval pipeline returns chunks

---

## Irrelevant Sources

Improve:

* Chunk size
* Chunk overlap
* Candidate retrieval size
* Embedding model quality
* Cross-encoder reranking threshold

---

# Tech Stack

### Backend

* Python
* FastAPI
* Requests

### Retrieval

* ChromaDB
* BM25
* Reciprocal Rank Fusion (RRF)
* Sentence Transformers

### Reranking

* Hugging Face Cross Encoder

### LLM Layer

* Ollama
* Qwen 2.5
* ngrok

### Document Processing

* PyPDF2
* pdfplumber

### Frontend

* Streamlit

---

# Future Improvements

* RAGAS Integration
* Batch Evaluation CLI
* CSV Benchmark Reports
* Multi-Model Benchmarking
* Metadata-Aware Retrieval
* Chunking Strategy A/B Testing
* Source Attribution Scoring
* Human Evaluation Dashboard
* Endpoint Configuration from UI
* Evaluation Dataset Versioning

---

# License

This project is intended for research, experimentation, benchmarking, and evaluation of Retrieval-Augmented Generation (RAG) systems with a focus on hallucination detection, reliability analysis, and evidence-grounded answer generation.

# 👩‍💻 Author

**Ishpreet Singh**

M.Tech
Indian Institute of Technology Bombay
Mail ID:
25m0326@iitb.ac.in


