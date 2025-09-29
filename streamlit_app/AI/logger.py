"""Simple JSONL logger."""
import json
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR

LOG_PATH = Path(DATA_DIR) / "app" / "app.log.jsonl"

def log_turn(payload: dict):
    payload = dict(payload)
    payload.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def log_error(session_id: str, response_id: str, code: str, message: str, details=None):
    log_turn({
        "session_id": session_id,
        "response_id": response_id,
        "error": {"code": code, "message": message, "details": details},
    })
