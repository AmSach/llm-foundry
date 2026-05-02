from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
import json
from datetime import datetime, timezone

from .agent import AgentTrace


@dataclass
class TraceStepRecord:
    raw: str
    action: str
    observation: str = ""
    final: str = ""


@dataclass
class TraceRecord:
    trace_id: str
    task: str
    final: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    steps: list[TraceStepRecord] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "final": self.final,
            "created_at": self.created_at,
            "steps": [asdict(step) for step in self.steps],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TraceRecord":
        return cls(
            trace_id=data["trace_id"],
            task=data["task"],
            final=data.get("final", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            steps=[TraceStepRecord(**step) for step in data.get("steps", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SFTExample:
    prompt: str
    completion: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion, "metadata": self.metadata}


class TraceDataset:
    def __init__(self, records: Iterable[TraceRecord] | None = None) -> None:
        self.records = list(records or [])

    def append(self, record: TraceRecord) -> None:
        self.records.append(record)

    def extend(self, records: Iterable[TraceRecord]) -> None:
        self.records.extend(records)

    def to_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return path

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TraceDataset":
        records: list[TraceRecord] = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            records.append(TraceRecord.from_dict(json.loads(line)))
        return cls(records)

    def to_sft_examples(self) -> list[SFTExample]:
        examples: list[SFTExample] = []
        for record in self.records:
            prompt = record.task
            if record.steps:
                transcript = "\n".join(
                    f"{step.action}: {step.observation or step.raw}" for step in record.steps
                )
                prompt = f"TASK:\n{record.task}\n\nTRACE:\n{transcript}"
            examples.append(SFTExample(prompt=prompt, completion=record.final, metadata={"trace_id": record.trace_id}))
        return examples

    def summary(self) -> dict[str, object]:
        total_steps = sum(len(record.steps) for record in self.records)
        return {
            "traces": len(self.records),
            "steps": total_steps,
            "mean_steps": (total_steps / len(self.records)) if self.records else 0,
        }

    @classmethod
    def from_agent_trace(cls, trace: AgentTrace, trace_id: str) -> "TraceDataset":
        steps = [TraceStepRecord(raw=step.raw, action=step.action, observation=step.observation, final=step.final) for step in trace.steps]
        return cls([TraceRecord(trace_id=trace_id, task=trace.task, final=trace.final, steps=steps, metadata={"source": "agent_runtime"})])
