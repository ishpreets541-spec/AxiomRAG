# app/rag/embeddings.py
import logging
from typing import List
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

logger = logging.getLogger(__name__)

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore

    def get_embed_model(model_name: str) -> BaseEmbedding:
        return HuggingFaceEmbedding(model_name=model_name)

except Exception:
    from sentence_transformers import SentenceTransformer

    class SentenceTransformerEmbedding(BaseEmbedding):
        def __init__(self, model_name: str):
            super().__init__(model_name=model_name)
            self._model = SentenceTransformer(model_name)

        def _get_query_embedding(self, query: str) -> Embedding:
            return self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()

        async def _aget_query_embedding(self, query: str) -> Embedding:
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text: str) -> Embedding:
            return self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()

        def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
            return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def get_embed_model(model_name: str) -> BaseEmbedding:
        logger.warning(
            "llama_index.embeddings.huggingface not available; using SentenceTransformer fallback."
        )
        return SentenceTransformerEmbedding(model_name=model_name)
