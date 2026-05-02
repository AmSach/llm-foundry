from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import json
from datetime import datetime, timezone

from .adapters import FixedBackend, SequenceBackend, TextBackend
from .agent import AgentRuntime, ToolRegistry
from .benchmark import BenchmarkSuite, default_benchmark_cases
from .config import BenchmarkConfig
from .memory import CompressionEngine, ObsidianMemoryVault
from .rag import LocalRetriever


@dataclass
class HarnessSummary:
    name: str
    total: int
    passed: int
    details: list[dict[str, object]]
    generated_at: str = datetime.now(timezone.utc).isoformat()

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "generated_at": self.generated_at,
            "details": self.details,
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    def write_markdown(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# {self.name}",
            "",
            f"- Total: `{self.total}`",
            f"- Passed: `{self.passed}`",
            f"- Pass rate: `{self.pass_rate:.2%}`",
            "",
            "| Case | Passed | Notes |",
            "|---|---:|---|",
        ]
        for item in self.details:
            lines.append(f"| {item.get('name', '')} | {str(item.get('passed', False)).lower()} | {item.get('notes', '')} |")
        lines.append("")
        path.write_text("\n".join(lines))
        return path


def reasoning_cases() -> list:
    return default_benchmark_cases() + [
        __import__("llm_foundry.benchmark", fromlist=["BenchmarkCase"]).BenchmarkCase(
            name="multi_step_reasoning",
            prompt="Explain in one sentence how checking your own answer can reduce errors.",
            expected_contains=("check", "answer"),
        )
    ]


def coding_cases() -> list:
    BenchmarkCase = __import__("llm_foundry.benchmark", fromlist=["BenchmarkCase"]).BenchmarkCase
    return [
        BenchmarkCase(
            name="python_add_function",
            prompt="Write a Python function called add that returns the sum of two numbers.",
            expected_contains=("def add", "return"),
        ),
        BenchmarkCase(
            name="python_sort_list",
            prompt="Write Python code to sort a list named numbers.",
            expected_contains=("sorted", "numbers"),
        ),
        BenchmarkCase(
            name="json_parse",
            prompt="Write Python code that parses a JSON string into a dictionary.",
            expected_contains=("json", "loads"),
        ),
    ]


@dataclass
class ToolUseCase:
    name: str
    task: str
    backend: TextBackend
    expected_final: str


def run_reasoning_harness(backend: TextBackend) -> HarnessSummary:
    suite = BenchmarkSuite(backend)
    report = suite.run(reasoning_cases())
    details = [{"name": result.name, "passed": result.passed, "notes": result.notes} for result in report.results]
    return HarnessSummary(name="reasoning", total=report.total, passed=report.passed, details=details)


def run_coding_harness(backend: TextBackend) -> HarnessSummary:
    suite = BenchmarkSuite(backend)
    report = suite.run(coding_cases())
    details = [{"name": result.name, "passed": result.passed, "notes": result.notes} for result in report.results]
    return HarnessSummary(name="coding", total=report.total, passed=report.passed, details=details)


def run_tool_use_harness(backend: TextBackend, workspace_root: str | Path) -> HarnessSummary:
    tools = ToolRegistry(workspace_root=workspace_root)
    runtime = AgentRuntime(backend=backend, tools=tools, max_steps=5)
    trace = runtime.run("List the repo and finish with a final answer.")
    passed = bool(trace.final)
    return HarnessSummary(
        name="tool-use",
        total=1,
        passed=1 if passed else 0,
        details=[{"name": "agent_trace", "passed": passed, "notes": trace.final[:120]}],
    )


def run_memory_harness(workspace_root: str | Path) -> HarnessSummary:
    vault = ObsidianMemoryVault(Path(workspace_root) / "memory-vault")
    vault.add_note("Project Goal", "Build a modular model framework with memory, compression, and tool use.", tags=["goal", "model"])
    vault.add_note("Memory Rule", "Always compress long conversations into a compact note before exceeding budget.", tags=["memory", "compression"])
    compressor = CompressionEngine(vault=vault)
    context = compressor.compress_transcript(
        task="How should the system remember important facts?",
        transcript=[
            "The system should keep important facts.",
            "It should compress long transcripts.",
            "It should search Obsidian-like notes before answering.",
        ],
        memory_query="memory compression",
        target_tokens=256,
    )
    passed = bool(context.memory_pack) and context.token_estimate_after <= context.token_estimate_before
    return HarnessSummary(
        name="memory",
        total=1,
        passed=1 if passed else 0,
        details=[{"name": "memory_pack", "passed": passed, "notes": context.summary[:120]}],
    )


def run_all_harnesses(backend: TextBackend, workspace_root: str | Path) -> dict[str, HarnessSummary]:
    return {
        "reasoning": run_reasoning_harness(backend),
        "coding": run_coding_harness(backend),
        "tool_use": run_tool_use_harness(backend, workspace_root),
        "memory": run_memory_harness(workspace_root),
    }
