"""
High-level entry point: answer a question with the tool-calling loop.
"""
from typing import Dict, Any, List
from pathlib import Path
from .openai_client import run_responses_loop
from .schemas import ANSWER_SCHEMA
from .logger import log_turn
from .config import AS_OF_DATE, VERSION

SYSTEM_PROMPT = Path(__file__).with_name("SYSTEM_PROMPT.txt").read_text(encoding="utf-8")

def answer_question(prompt: str, session_id: str, use_web: bool=False) -> Dict[str, Any]:
    messages: List[Dict[str,str]] = [
        {"role":"user", "content": prompt},
        {"role":"system", "content": f"session_id={session_id} use_web={use_web}"}
    ]
    out = run_responses_loop(messages, SYSTEM_PROMPT, ANSWER_SCHEMA)
    log_turn({
        "session_id": session_id,
        "messages": messages,
        "output_json": out,
        "flags": {"used_web_search": use_web}
    })
    return out
