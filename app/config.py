# app/config.py
from pathlib import Path

# Directories
DATA_DIR = Path("data/raw_docs")
CHROMA_DIR = Path("data/chroma_db")

# Model names
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # or any SentenceTransformer model
TOP_K = 10
