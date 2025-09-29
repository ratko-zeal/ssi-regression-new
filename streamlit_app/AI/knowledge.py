"""Utilities for loading and searching local knowledge documents."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

from .config import KNOWLEDGE_DIR

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover - dependency optional at runtime
    PdfReader = None  # type: ignore


@dataclass
class KnowledgeDoc:
    name: str
    path: Path
    text: str

    def to_snippet(self, query: str, window: int = 360) -> str:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        match = pattern.search(self.text)
        if not match:
            return self.text[:window].strip()
        start = max(match.start() - window // 2, 0)
        end = min(match.end() + window // 2, len(self.text))
        snippet = self.text[start:end]
        return snippet.strip()


def _read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is required to read PDF knowledge files")
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_doc(path: Path) -> KnowledgeDoc:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix in {".txt", ".md", ".markdown"}:
        text = _read_text_file(path)
    else:
        raise ValueError(f"Unsupported knowledge file type: {path.name}")
    normalized = re.sub(r"\s+", " ", text).strip()
    return KnowledgeDoc(name=path.stem, path=path, text=normalized)


@lru_cache(maxsize=1)
def load_knowledge() -> List[KnowledgeDoc]:
    base = Path(KNOWLEDGE_DIR)
    if not base.exists():
        return []
    docs: List[KnowledgeDoc] = []
    for path in sorted(base.glob("**/*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}:
            try:
                docs.append(_load_doc(path))
            except Exception:
                continue
    return docs


def _score_doc(doc: KnowledgeDoc, query_tokens: Iterable[str]) -> int:
    return sum(doc.text.lower().count(token) for token in query_tokens)


def search_knowledge(query: str, limit: int = 3) -> List[dict]:
    tokens = [tok for tok in re.findall(r"\w+", query.lower()) if tok]
    if not tokens:
        return []
    matches = []
    for doc in load_knowledge():
        score = _score_doc(doc, tokens)
        if score:
            snippet = doc.to_snippet(tokens[0])
            matches.append({
                "document": doc.name,
                "path": str(doc.path),
                "score": score,
                "snippet": snippet,
            })
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:limit]


def clear_cache():
    load_knowledge.cache_clear()
