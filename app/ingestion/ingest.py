# app/ingestion/ingest.py
import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from app.rag.embeddings import get_embed_model
from app.config import DATA_DIR, CHROMA_DIR, EMBED_MODEL_NAME

def ingest():
    # 1. Load documents
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print("Loaded documents:", len(documents))

    # 2. Chunk documents
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print("Created chunks:", len(nodes))

    # 3. Initialize embedding model
    embed_model = get_embed_model(EMBED_MODEL_NAME)

    text_nodes = []
    texts = []
    for i, node in enumerate(nodes):
        text_nodes.append(TextNode(text=node.text, id_=f"chunk_{i}", metadata=node.metadata))
        texts.append(node.text)

    # 4. Compute embeddings
    try:
        embeddings = embed_model._get_text_embeddings(texts)
    except Exception:
        embeddings = [embed_model._get_text_embedding(txt) for txt in texts]

    # 5. Attach embeddings to nodes
    for node, emb in zip(text_nodes, embeddings):
        node.embedding = emb

    # 6. Persist to Chroma
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("rag_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    vector_store.add(nodes=text_nodes)
    print("Persisted chunks to Chroma:", len(text_nodes))

if __name__ == "__main__":
    ingest()
