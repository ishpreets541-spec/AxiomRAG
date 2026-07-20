import math
import re
from collections import Counter
from typing import Dict, List, Tuple

import chromadb

from app.config import CHROMA_DIR, TOP_K
from app.rag.embeddings import get_embed_model
from app.rag.index import load_index

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

_RERANK_MODEL = None
_EMBED_MODEL = None

RERANK_TOP_N = 3
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _dense_retrieve(query: str, top_k: int) -> List[Dict]:
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    results = []
    for rank, node in enumerate(nodes, start=1):
        metadata = node.metadata or {}
        source = None
        if isinstance(metadata, dict):
            source = metadata.get("file_name") or metadata.get("file_path")
        results.append(
            {
                "chunk_id": node.node_id,
                "text": node.text,
                "dense_score": float(node.score) if node.score is not None else 0.0,
                "dense_rank": rank,
                "metadata": metadata,
                "source": source,
            }
        )
    return results


def _get_bm25_corpus() -> Tuple[List[str], List[str], List[Dict]]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("rag_docs")
    payload = collection.get(include=["documents", "metadatas"])
    ids = payload.get("ids") or []
    docs = payload.get("documents") or []
    metadatas = payload.get("metadatas") or [{} for _ in ids]
    return ids, docs, metadatas


def _bm25_retrieve(query: str, top_k: int) -> List[Dict]:
    ids, docs, metadatas = _get_bm25_corpus()
    if not docs:
        return []

    tokenized_docs = [_tokenize(doc) for doc in docs]
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    n_docs = len(tokenized_docs)
    avgdl = sum(len(d) for d in tokenized_docs) / max(n_docs, 1)
    df = Counter()
    for doc_tokens in tokenized_docs:
        for token in set(doc_tokens):
            df[token] += 1

    scores = []
    for i, doc_tokens in enumerate(tokenized_docs):
        if not doc_tokens:
            continue
        freq = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for token in query_tokens:
            if token not in freq:
                continue
            term_df = df.get(token, 0)
            idf = math.log(1 + (n_docs - term_df + 0.5) / (term_df + 0.5))
            tf = freq[token]
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / max(avgdl, 1e-8))
            score += idf * ((tf * (BM25_K1 + 1)) / max(denom, 1e-8))

        if score > 0:
            metadata = metadatas[i] or {}
            source = None
            if isinstance(metadata, dict):
                source = metadata.get("file_name") or metadata.get("file_path")
            scores.append(
                {
                    "chunk_id": ids[i],
                    "text": docs[i],
                    "bm25_score": float(score),
                    "metadata": metadata,
                    "source": source,
                }
            )

    scores.sort(key=lambda x: x["bm25_score"], reverse=True)
    for rank, item in enumerate(scores, start=1):
        item["bm25_rank"] = rank
    return scores[:top_k]


def _rrf_fuse(dense: List[Dict], bm25: List[Dict]) -> List[Dict]:
    merged: Dict[str, Dict] = {}

    for item in dense:
        cid = item["chunk_id"]
        fused = 1.0 / (RRF_K + item["dense_rank"])
        merged[cid] = {**item, "fused_score": fused}

    for item in bm25:
        cid = item["chunk_id"]
        fused = 1.0 / (RRF_K + item["bm25_rank"])
        if cid in merged:
            merged[cid]["fused_score"] += fused
            if not merged[cid].get("text") and item.get("text"):
                merged[cid]["text"] = item["text"]
            if not merged[cid].get("metadata") and item.get("metadata"):
                merged[cid]["metadata"] = item["metadata"]
            if not merged[cid].get("source") and item.get("source"):
                merged[cid]["source"] = item["source"]
            merged[cid]["bm25_score"] = item.get("bm25_score", 0.0)
            merged[cid]["bm25_rank"] = item.get("bm25_rank")
        else:
            merged[cid] = {**item, "fused_score": fused}

    fused_items = list(merged.values())
    fused_items.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)
    return fused_items


def _get_reranker():
    global _RERANK_MODEL
    if _RERANK_MODEL is None and CrossEncoder is not None:
        _RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANK_MODEL


def _overlap_score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q:
        return 0.0
    return len(q & t) / len(q)


def _rerank(query: str, candidates: List[Dict], top_n: int) -> List[Dict]:
    if not candidates:
        return []
    model = _get_reranker()

    if model is not None:
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
    else:
        for c in candidates:
            c["rerank_score"] = _overlap_score(query, c.get("text", ""))

    candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return candidates[:top_n]


def retrieve_chunks(query: str):
    top_k = max(TOP_K, 10)
    dense = _dense_retrieve(query, top_k=top_k)
    bm25 = _bm25_retrieve(query, top_k=top_k)
    fused = _rrf_fuse(dense, bm25)
    reranked = _rerank(query, fused[: max(top_k, 15)], top_n=RERANK_TOP_N)

    for i, item in enumerate(reranked, start=1):
        item["final_rank"] = i

    return reranked
