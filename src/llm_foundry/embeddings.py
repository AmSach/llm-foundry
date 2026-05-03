from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
import json
import math
import re
from typing import Iterable


@dataclass(frozen=True)
class EmbeddingDocument:
    path: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    path: str
    text: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)


class HashEmbeddingModel:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = max(32, dimension)

    def embed(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self.dimension
        counts = Counter(tokens)
        vector = [0.0] * self.dimension
        for token, count in counts.items():
            bucket = _bucket(token, self.dimension)
            vector[bucket] += count * (1.0 + math.log1p(len(token)))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class EmbeddingIndex:
    def __init__(self, model: HashEmbeddingModel | None = None) -> None:
        self.model = model or HashEmbeddingModel()
        self._documents: list[EmbeddingDocument] = []
        self._vectors: list[list[float]] = []

    @property
    def documents(self) -> list[EmbeddingDocument]:
        return list(self._documents)

    def clear(self) -> None:
        self._documents.clear()
        self._vectors.clear()

    def add(self, document: EmbeddingDocument) -> None:
        self._documents.append(document)
        self._vectors.append(self.model.embed(document.text))

    def extend(self, documents: Iterable[EmbeddingDocument]) -> None:
        for document in documents:
            self.add(document)

    def rebuild(self, documents: Iterable[EmbeddingDocument]) -> None:
        self.clear()
        self.extend(documents)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[EmbeddingResult]:
        query_vector = self.model.embed(query)
        if not any(query_vector):
            return []
        results: list[EmbeddingResult] = []
        for document, vector in zip(self._documents, self._vectors):
            score = _cosine(query_vector, vector)
            if score >= min_score:
                results.append(
                    EmbeddingResult(
                        path=document.path,
                        text=document.text,
                        score=score,
                        metadata=dict(document.metadata),
                    )
                )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def to_json(self) -> dict:
        return {
            "model": {"dimension": self.model.dimension},
            "documents": [
                {"path": doc.path, "text": doc.text, "metadata": doc.metadata}
                for doc in self._documents
            ],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False))
        return path


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bucket(token: str, dimension: int) -> int:
    return abs(hash(token)) % dimension


def _cosine(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))
