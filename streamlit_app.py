from typing import Any, Dict, List

import streamlit as st

from app.main import main as audit_main
from app.utils.llm import llm


def _verdict_label(verdict: str) -> str:
    return {
        "SAFE": "Safe",
        "PARTIALLY_SUPPORTED": "Partially supported",
        "UNSAFE": "Unsafe",
    }.get(verdict, verdict)


def _verdict_color(verdict: str) -> str:
    return {
        "SAFE": "green",
        "PARTIALLY_SUPPORTED": "orange",
        "UNSAFE": "red",
    }.get(verdict, "blue")


def _format_sources(sources: List[str]) -> str:
    if not sources:
        return "No sources retrieved."
    return "\n".join(f"- {source}" for source in sources)


st.set_page_config(
    page_title="RAG Reliability Auditor",
    layout="wide",
    page_icon="🔎",
)

st.title("Production RAG Reliability Auditor")
st.caption("Hybrid retrieval + reranking + LLM generation + evaluation metrics + citation panel")

with st.sidebar:
    st.header("Pipeline")
    st.write("Ingestion: chunked PDFs in ChromaDB")
    st.write("Retrieval: BM25 + dense vectors + RRF fusion")
    st.write("Reranker: cross-encoder top-k -> top-3")
    st.write("Generation model: qwen2.5:0.5b via ngrok endpoint")

if "history" not in st.session_state:
    st.session_state.history = []

SUGGESTED_QUESTIONS = [
    {
        "pdf": "privacy policy google.pdf",
        "question": "Under what conditions does Google say it may share personal information with third parties?",
    },
    {
        "pdf": "Privacy Policy _ Atlassian.pdf",
        "question": "What rights does Atlassian mention for users in EEA, UK, or US regional disclosures?",
    },
    {
        "pdf": "amazon_terms&conditions.pdf",
        "question": "What limitations of liability are described in Amazon's terms and conditions?",
    },
    {
        "pdf": "HR Policy iima.pdf",
        "question": "What does the HR policy say about leave types, eligibility, and approval workflow?",
    },
    {
        "pdf": "sop for court documents.pdf",
        "question": "What is the step-by-step SOP for preparing, reviewing, and filing court documents?",
    },
]

st.subheader("Suggested Questions (5 PDFs, 5 aspects)")
for idx, item in enumerate(SUGGESTED_QUESTIONS, start=1):
    st.markdown(f"{idx}. **{item['pdf']}**: {item['question']}")

default_query = "Does Google share user data with third parties?"
selected_suggestion = st.selectbox(
    "Quick pick a suggested question",
    options=["Custom question"] + [f"{q['pdf']} | {q['question']}" for q in SUGGESTED_QUESTIONS],
    index=0,
)

prefill_query = default_query
if selected_suggestion != "Custom question":
    prefill_query = selected_suggestion.split(" | ", 1)[1]

query = st.text_area("Ask a question", value=prefill_query, height=120)

run = st.button("Run Full RAG Pipeline", type="primary")

if run:
    if not query.strip():
        st.error("Please provide a query.")
    else:
        with st.spinner("Running retrieval and evaluation..."):
            try:
                result = audit_main(query.strip(), "", llm)
            except Exception as exc:
                st.error(f"LLM request failed: {exc}")
                with st.expander("Error details"):
                    st.code(str(exc), language="text")
            else:
                st.session_state.history.append(
                    {
                        "query": query.strip(),
                        "answer": result.generated_answer,
                        "verdict": result.verdict,
                    }
                )

                cols = st.columns([1, 1, 1])
                cols[0].metric("Verdict", _verdict_label(result.verdict))
                cols[1].metric("Claims", len(result.claims))
                cols[2].metric("Evidence items", len(result.evidence))

                st.markdown(
                    f"### Verdict: :{_verdict_color(result.verdict)}[{_verdict_label(result.verdict)}]"
                )

                st.subheader("Generated Answer")
                st.write(result.generated_answer or "No answer was generated.")

                left, right = st.columns([1, 1])

                with left:
                    st.subheader("Claims")
                    if not result.claims:
                        st.info("No claims were extracted from the answer.")
                    else:
                        for claim in result.claims:
                            with st.expander(f"{claim.id}: {claim.text}", expanded=False):
                                st.write(claim.text)

                with right:
                    st.subheader("Evidence")
                    if not result.evidence:
                        st.info("No evidence was matched.")
                    else:
                        for item in result.evidence:
                            with st.expander(f"Claim {item.claim_id} - {item.support_level}", expanded=False):
                                st.write(f"Support level: {item.support_level}")
                                st.write(f"Confidence: {item.confidence:.2f}")
                                if item.rationale:
                                    st.write(f"Rationale: {item.rationale}")
                                st.write(f"Chunk IDs: {', '.join(item.chunk_ids) if item.chunk_ids else 'None'}")

                st.subheader("Metrics")
                metrics_dict: Dict[str, Any] = result.metrics.dict()
                st.json(metrics_dict)

                st.subheader("Sources")
                st.code(_format_sources(list(result.sources or [])), language="text")

                st.subheader("Citation Panel (Top Retrieved Chunks)")
                retrieved = list(result.retrieved_chunks or [])
                if not retrieved:
                    st.info("No retrieved chunks available.")
                else:
                    for chunk in retrieved:
                        chunk_id = chunk.get("chunk_id", "unknown")
                        source = chunk.get("source") or (chunk.get("metadata") or {}).get("file_name") or "unknown"
                        metadata = chunk.get("metadata") or {}
                        page_no = metadata.get("page_label") or metadata.get("page") or metadata.get("page_number") or "n/a"
                        with st.expander(f"{chunk_id} | source: {source} | page: {page_no}", expanded=False):
                            st.write(chunk.get("text", ""))
                            st.caption(
                                f"fused_score={chunk.get('fused_score', 'n/a')} | "
                                f"rerank_score={chunk.get('rerank_score', 'n/a')} | "
                                f"dense_score={chunk.get('dense_score', 'n/a')} | "
                                f"bm25_score={chunk.get('bm25_score', 'n/a')}"
                            )

st.subheader("Chat History")
if not st.session_state.history:
    st.info("No runs yet.")
else:
    for i, item in enumerate(reversed(st.session_state.history), start=1):
        with st.expander(f"Run {i} | {_verdict_label(item['verdict'])}", expanded=False):
            st.write(f"Q: {item['query']}")
            st.write(f"A: {item['answer']}")
