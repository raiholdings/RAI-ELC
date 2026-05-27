"""Retrieval offline cho knowledge base dự án.

Đọc index BM25 do `scripts/ingest_projects.py` build ra, trả top-k chunk.
Không cần API key, chạy hoàn toàn local.

Cache LRU 1-slot/dự án trong RAM để khỏi load lại mỗi lần gọi.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

log = logging.getLogger(__name__)

# Repo layout (local): apps/agent-engine/app/agents/retrieval.py
# -> repo root cách đây 4 cấp.
# In container (Railway): /app/app/agents/retrieval.py -> fewer parents.
# Use env var override or safe fallback.
_file_parents = Path(__file__).resolve().parents
REPO_ROOT = Path(os.environ["REPO_ROOT"]) if os.environ.get("REPO_ROOT") else (
    _file_parents[4] if len(_file_parents) > 4 else _file_parents[-1]
)
KB_DIR = REPO_ROOT / "data" / "knowledge_base"


_TOKEN = re.compile(r"[a-z0-9]+")

def _strip_accents(s: str) -> str:
    nf = unicodedata.normalize("NFD", s)
    return "".join(c for c in nf if unicodedata.category(c) != "Mn")

def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(_strip_accents(text.lower()))


@dataclass
class RetrievedChunk:
    score: float
    text: str
    source_file: str
    group: str
    chunk_index: int

    def short(self, max_chars: int = 280) -> str:
        t = self.text.replace("\n", " ").strip()
        if len(t) > max_chars:
            t = t[: max_chars - 1] + "…"
        return t


class ProjectIndex:
    def __init__(self, project_slug: str):
        self.project_slug = project_slug
        kb = KB_DIR / project_slug
        self.bm25_path = kb / "bm25.pkl"
        self.chunks_path = kb / "chunks.jsonl"
        self._bm25 = None
        self._chunks: list[dict] = []
        self._load()

    @property
    def available(self) -> bool:
        return self._bm25 is not None and bool(self._chunks)

    def _load(self) -> None:
        if not self.bm25_path.exists() or not self.chunks_path.exists():
            log.warning("Không tìm thấy index BM25 cho '%s' tại %s",
                        self.project_slug, self.bm25_path.parent)
            return
        with self.chunks_path.open("r", encoding="utf-8") as f:
            self._chunks = [json.loads(line) for line in f if line.strip()]
        with self.bm25_path.open("rb") as f:
            obj = pickle.load(f)
        self._bm25 = obj["bm25"]
        log.info("Đã load index '%s': %d chunk", self.project_slug, len(self._chunks))

    def search(self, query: str, top_k: int = 5,
               group_filter: Optional[set[str]] = None) -> list[RetrievedChunk]:
        if not self.available:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)

        # Áp dụng filter theo group nếu có
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedChunk] = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            ch = self._chunks[idx]
            if group_filter and ch["group"] not in group_filter:
                continue
            out.append(RetrievedChunk(
                score=float(scores[idx]),
                text=ch["text"],
                source_file=ch["source_file"],
                group=ch["group"],
                chunk_index=ch["chunk_index"],
            ))
            if len(out) >= top_k:
                break
        return out


# Cache index theo project_slug — load 1 lần / process.
_cache: dict[str, ProjectIndex] = {}
_cache_lock = Lock()


def get_index(project_slug: str) -> Optional[ProjectIndex]:
    with _cache_lock:
        if project_slug not in _cache:
            _cache[project_slug] = ProjectIndex(project_slug)
        idx = _cache[project_slug]
    return idx if idx.available else None


def format_context_for_llm(chunks: list[RetrievedChunk], max_chars: int = 6000) -> str:
    """Gói chunk lại để đưa vào system prompt của LLM."""
    if not chunks:
        return ""
    lines = ["Trích từ tài liệu dự án (top match):"]
    used = 0
    for i, c in enumerate(chunks, 1):
        block = (
            f"\n--- [{i}] Nguồn: {c.source_file} (nhóm: {c.group}, "
            f"chunk #{c.chunk_index}, score={c.score:.2f})\n{c.text.strip()}\n"
        )
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "".join(lines)
