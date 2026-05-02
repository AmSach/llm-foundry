from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import json
import math
import re
from typing import Iterable, Sequence

from .tokenizer import estimate_token_count


@dataclass
class MemoryNote:
    id: str
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = ""

    def to_markdown(self) -> str:
        metadata = {
            "id": self.id,
            "title": self.title,
            "tags": ", ".join(self.tags),
            "links": ", ".join(self.links),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
        }
        header = ["---"] + [f"{key}: {value}" for key, value in metadata.items()] + ["---", ""]
        return "\n".join(header + [self.body.strip(), ""])

    @classmethod
    def from_markdown(cls, text: str) -> "MemoryNote":
        frontmatter, body = _split_frontmatter(text)
        return cls(
            id=frontmatter.get("id", _slugify(frontmatter.get("title", "note"))),
            title=frontmatter.get("title", "Untitled note"),
            body=body.strip(),
            tags=[part.strip() for part in frontmatter.get("tags", "").split(",") if part.strip()],
            links=[part.strip() for part in frontmatter.get("links", "").split(",") if part.strip()],
            created_at=frontmatter.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=frontmatter.get("updated_at", datetime.now(timezone.utc).isoformat()),
            source=frontmatter.get("source", ""),
        )


@dataclass
class MemoryMatch:
    note: MemoryNote
    score: float
    excerpt: str


@dataclass
class CompressedContext:
    task: str
    summary: str
    salient_facts: list[str]
    open_questions: list[str]
    action_items: list[str]
    memory_pack: str
    token_estimate_before: int
    token_estimate_after: int

    def to_prompt(self) -> str:
        parts = [
            "MEMORY SUMMARY:",
            self.summary.strip(),
            "",
        ]
        if self.salient_facts:
            parts.extend(["SALIENT FACTS:"] + [f"- {fact}" for fact in self.salient_facts] + [""])
        if self.action_items:
            parts.extend(["ACTION ITEMS:"] + [f"- {item}" for item in self.action_items] + [""])
        if self.open_questions:
            parts.extend(["OPEN QUESTIONS:"] + [f"- {question}" for question in self.open_questions] + [""])
        if self.memory_pack:
            parts.extend(["RELEVANT NOTES:", self.memory_pack.strip(), ""])
        return "\n".join(parts).strip()


class ObsidianMemoryVault:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.notes_dir = self.root / "notes"
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def add_note(self, title: str, body: str, tags: Sequence[str] = (), links: Sequence[str] = (), source: str = "") -> MemoryNote:
        note_id = _slugify(title) or f"note-{len(list(self.notes_dir.glob('*.md')))}"
        note = MemoryNote(id=note_id, title=title, body=body, tags=list(tags), links=list(links), source=source)
        self.save(note)
        return note

    def save(self, note: MemoryNote) -> Path:
        path = self.notes_dir / f"{_slugify(note.title)}-{note.id}.md"
        note.updated_at = datetime.now(timezone.utc).isoformat()
        path.write_text(note.to_markdown())
        return path

    def list_notes(self) -> list[MemoryNote]:
        notes: list[MemoryNote] = []
        for path in sorted(self.notes_dir.glob("*.md")):
            try:
                notes.append(MemoryNote.from_markdown(path.read_text()))
            except OSError:
                continue
        return notes

    def search(self, query: str, top_k: int = 5) -> list[MemoryMatch]:
        terms = _tokenize(query)
        if not terms:
            return []
        results: list[MemoryMatch] = []
        for note in self.list_notes():
            score = _score_note(terms, note)
            if score <= 0:
                continue
            excerpt = _build_excerpt(note.body, terms)
            results.append(MemoryMatch(note=note, score=score, excerpt=excerpt))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def render_pack(self, query: str, top_k: int = 5, max_chars: int = 4000) -> str:
        matches = self.search(query, top_k=top_k)
        if not matches:
            return "no relevant memory notes"
        blocks: list[str] = []
        total = 0
        for match in matches:
            block = f"[{match.note.title}] ({', '.join(match.note.tags)})\n{match.excerpt}\n"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n".join(blocks).strip()


class CompressionEngine:
    def __init__(self, vault: ObsidianMemoryVault | None = None, summary_sentences: int = 6) -> None:
        self.vault = vault
        self.summary_sentences = summary_sentences

    def compress_transcript(self, task: str, transcript: Sequence[str], memory_query: str = "", target_tokens: int = 512) -> CompressedContext:
        text = "\n".join(_clean_line(line) for line in transcript if line.strip())
        before = estimate_token_count(text)
        summary_sentences = _rank_sentences(text, task, self.summary_sentences)
        salient_facts = _extract_bullets(text, ["must", "should", "need", "important", "remember", "because"])
        action_items = _extract_bullets(text, ["todo", "action", "next", "do", "run", "write"])
        open_questions = _extract_questions(text)
        memory_pack = self.vault.render_pack(memory_query or task, top_k=5, max_chars=target_tokens * 4) if self.vault else ""
        summary = " ".join(summary_sentences) if summary_sentences else _default_summary(task, text)
        after_parts = [summary] + salient_facts + action_items + open_questions + ([memory_pack] if memory_pack else [])
        after = estimate_token_count("\n".join(after_parts))
        return CompressedContext(
            task=task,
            summary=summary,
            salient_facts=salient_facts,
            open_questions=open_questions,
            action_items=action_items,
            memory_pack=memory_pack,
            token_estimate_before=before,
            token_estimate_after=after,
        )

    def save_context_note(self, context: CompressedContext, title: str | None = None) -> MemoryNote | None:
        if not self.vault:
            return None
        note_title = title or f"Context for {context.task[:40]}"
        body = context.to_prompt()
        return self.vault.add_note(note_title, body, tags=["compressed", "agent", "memory"])


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text
    try:
        end = lines[1:].index("---") + 1
    except ValueError:
        return {}, text
    frontmatter_lines = lines[1:end]
    body = "\n".join(lines[end + 1 :])
    frontmatter: dict[str, str] = {}
    for line in frontmatter_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip()
    return frontmatter, body


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "note"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_note(terms: list[str], note: MemoryNote) -> float:
    tokens = Counter(_tokenize(note.title + " " + note.body + " " + " ".join(note.tags)))
    overlap = sum(min(tokens.get(term, 0), 1) for term in terms)
    title_bonus = sum(1 for term in terms if term in note.title.lower()) * 2.0
    link_bonus = sum(0.25 for term in terms if any(term in link.lower() for link in note.links))
    return float(overlap + title_bonus + link_bonus)


def _build_excerpt(text: str, terms: list[str], max_len: int = 280) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    scored = sorted(sentences, key=lambda sentence: _sentence_score(sentence, terms), reverse=True)
    excerpt = " ".join(scored[:2]).strip() or text.strip()
    return excerpt[:max_len]


def _sentence_score(sentence: str, terms: list[str]) -> float:
    lowered = sentence.lower()
    return sum(1.0 for term in terms if term in lowered) + (0.5 if "[[" in sentence else 0.0)


def _rank_sentences(text: str, task: str, limit: int) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        return []
    task_terms = _tokenize(task)
    scored = sorted(sentences, key=lambda sentence: _sentence_score(sentence, task_terms) + len(sentence) / 300.0, reverse=True)
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in scored:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
        if len(deduped) >= limit:
            break
    return deduped


def _extract_bullets(text: str, keywords: Sequence[str]) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            bullets.append(f"{line}")
    return bullets[:8]


def _extract_questions(text: str) -> list[str]:
    questions = [sentence.strip() for sentence in re.split(r"(?<=[?])\s+", text) if "?" in sentence]
    return questions[:6]


def _clean_line(line: str) -> str:
    line = line.replace("\t", " ")
    return re.sub(r"\s+", " ", line).strip()


def _default_summary(task: str, text: str) -> str:
    if not text:
        return f"Compressed memory for task: {task}"
    return f"Compressed memory for task: {task}. The transcript was condensed to the most useful facts, actions, and open questions."
