# app/rag/index.py
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from app.rag.embeddings import get_embed_model
from app.config import CHROMA_DIR, EMBED_MODEL_NAME

def load_index():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection("rag_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = get_embed_model(EMBED_MODEL_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return index
