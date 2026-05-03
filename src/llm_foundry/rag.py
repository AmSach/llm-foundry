from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .embeddings import EmbeddingDocument, EmbeddingIndex, HashEmbeddingModel


@dataclass(frozen=True)
class RAGChunk:
    path: str
    text: str
    score: float = 0.0


class LocalRetriever:
    def __init__(self, root: str | Path, extensions: Iterable[str] = (".md", ".txt", ".py", ".json")) -> None:
        self.root = Path(root).resolve()
        self.extensions = tuple(extensions)
        self.index = build_embedding_index(self.root, self.extensions)

    def search(self, query: str, top_k: int = 5) -> list[RAGChunk]:
        results = self.index.search(query, top_k=top_k)
        return [RAGChunk(path=item.path, text=item.text, score=item.score) for item in results]

    def as_tool(self):
        def tool(query: str, top_k: int = 5) -> str:
            results = self.search(query, top_k=top_k)
            if not results:
                return "no relevant context"
            return "\n\n".join(f"{item.path}: {item.text}" for item in results)

        return tool


def build_embedding_index(root: str | Path, extensions: Iterable[str] = (".md", ".txt", ".py", ".json")) -> EmbeddingIndex:
    root = Path(root).resolve()
    index = EmbeddingIndex(HashEmbeddingModel())
    documents: list[EmbeddingDocument] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in extensions:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        parts = re.split(r"\n\s*\n", text)
        for part in parts:
            stripped = part.strip()
            if stripped:
                documents.append(EmbeddingDocument(path=str(path.relative_to(root)), text=stripped))
    index.rebuild(documents)
    return index
