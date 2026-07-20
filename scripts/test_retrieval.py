# test_retrieval.py
from app.rag.retriever import retrieve_chunks

if __name__ == "__main__":
    query = "What does the privacy policy say about data sharing?"
    chunks = retrieve_chunks(query)
    print(f"Retrieved {len(chunks)} chunks\n")
    for c in chunks[:2]:
        print("CHUNK ID:", c["chunk_id"])
        print("SOURCE:", c.get("source"))
        print("TEXT:", c["text"][:300])
        print("-" * 40)
