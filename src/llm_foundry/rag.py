from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from collections import Counter
from typing import Iterable


@dataclass(frozen=True)
class RAGChunk:
    path: str
    text: str
    score: float = 0.0


class LocalRetriever:
    def __init__(self, root: str | Path, extensions: Iterable[str] = (".md", ".txt", ".py", ".json")) -> None:
        self.root = Path(root).resolve()
        self.extensions = tuple(extensions)
        self.chunks = self._build_chunks()

    def _build_chunks(self) -> list[RAGChunk]:
        chunks: list[RAGChunk] = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix not in self.extensions:
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            parts = re.split(r"\n\s*\n", text)
            for part in parts:
                stripped = part.strip()
                if stripped:
                    chunks.append(RAGChunk(path=str(path.relative_to(self.root)), text=stripped))
        return chunks

    def search(self, query: str, top_k: int = 5) -> list[RAGChunk]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        scored: list[RAGChunk] = []
        for chunk in self.chunks:
            score = _score(query_terms, _tokenize(chunk.text))
            if score > 0:
                scored.append(RAGChunk(path=chunk.path, text=chunk.text, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def as_tool(self):
        def tool(query: str, top_k: int = 5) -> str:
            results = self.search(query, top_k=top_k)
            if not results:
                return "no relevant context"
            return "\n\n".join(f"{item.path}: {item.text}" for item in results)

        return tool


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score(query_terms: list[str], doc_terms: list[str]) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    q = Counter(query_terms)
    d = Counter(doc_terms)
    overlap = sum(min(q[word], d[word]) for word in q if word in d)
    norm = math.sqrt(sum(v * v for v in q.values())) * math.sqrt(sum(v * v for v in d.values()))
    return overlap / norm if norm else 0.0
