# app/rag/trace_logger.py
import json
import uuid
from pathlib import Path
from datetime import datetime

TRACE_DIR = Path("traces")
TRACE_DIR.mkdir(exist_ok=True)

def new_trace_id() -> str:
    return str(uuid.uuid4())

def write_trace(trace: dict):
    trace["timestamp"] = datetime.utcnow().isoformat()
    trace_id = trace["trace_id"]
    path = TRACE_DIR / f"{trace_id}.json"
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)
