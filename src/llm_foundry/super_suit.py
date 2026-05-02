from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

from .adapters import MultiEndpointBackend, TextBackend
from .agent import AgentRuntime, ToolPolicy, ToolRegistry
from .datasets import TraceDataset
from .memory import CompressionEngine, CompressedContext, ObsidianMemoryVault


@dataclass
class SuperSuitConfig:
    workspace_root: str | Path = "."
    memory_root: str | Path = "memory-vault"
    max_steps: int = 8
    target_tokens: int = 512
    memory_query_top_k: int = 5
    allow_web_fetch: bool = False
    allow_web_search: bool = False
    allow_github_api: bool = False
    allow_shell: bool = False
    repair_attempts: int = 1
    api_endpoints_file: str | None = None
    api_endpoints_json: str | None = None
    api_strategy: str = "failover"


@dataclass
class SuperSuitResult:
    task: str
    prepared_task: str
    compressed_context: CompressedContext
    final: str
    trace: object
    active_endpoints: list[str] = field(default_factory=list)
    notes_written: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "prepared_task": self.prepared_task,
            "compressed_context": {
                "task": self.compressed_context.task,
                "summary": self.compressed_context.summary,
                "salient_facts": self.compressed_context.salient_facts,
                "open_questions": self.compressed_context.open_questions,
                "action_items": self.compressed_context.action_items,
                "memory_pack": self.compressed_context.memory_pack,
                "token_estimate_before": self.compressed_context.token_estimate_before,
                "token_estimate_after": self.compressed_context.token_estimate_after,
                "compressed_prompt": self.compressed_context.compressed_prompt,
            },
            "final": self.final,
            "active_endpoints": self.active_endpoints,
            "notes_written": self.notes_written,
            "trace": {
                "task": getattr(self.trace, "task", ""),
                "final": getattr(self.trace, "final", ""),
                "steps": [
                    {
                        "raw": step.raw,
                        "action": step.action,
                        "observation": step.observation,
                        "final": step.final,
                    }
                    for step in getattr(self.trace, "steps", [])
                ],
            },
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path


class ModelSuperSuit:
    def __init__(self, backend: TextBackend, config: SuperSuitConfig | None = None) -> None:
        self.backend = backend
        self.config = config or SuperSuitConfig()
        self.vault = ObsidianMemoryVault(self.config.memory_root)
        self.compressor = CompressionEngine(vault=self.vault)
        self.policy = ToolPolicy(
            allow_workspace_write=True,
            allow_web_fetch=self.config.allow_web_fetch,
            allow_web_search=self.config.allow_web_search,
            allow_github_api=self.config.allow_github_api,
            allow_shell=self.config.allow_shell,
        )
        self.tools = ToolRegistry(workspace_root=self.config.workspace_root, policy=self.policy)
        self.runtime = AgentRuntime(
            backend=self.backend,
            tools=self.tools,
            max_steps=self.config.max_steps,
            repair_attempts=self.config.repair_attempts,
        )

    def run(
        self,
        task: str,
        memory_query: str = "",
        save_note: str = "",
        export_trace_path: str | Path = "",
        export_sft_path: str | Path = "",
    ) -> SuperSuitResult:
        memory_query = memory_query or task
        memory_pack = self.vault.render_pack(memory_query, top_k=self.config.memory_query_top_k, max_chars=self.config.target_tokens * 3)
        transcript = [f"TASK: {task}"]
        if memory_pack:
            transcript.append(f"MEMORY: {memory_pack}")
        compressed_context = self.compressor.compress_transcript(
            task=task,
            transcript=transcript,
            memory_query=memory_query,
            target_tokens=self.config.target_tokens,
        )
        prepared_task = _build_prepared_task(task, compressed_context)
        trace = self.runtime.run(prepared_task)
        notes_written: list[str] = []
        if save_note:
            note = self.vault.add_note(save_note, compressed_context.to_prompt(), tags=["super-suit", "memory", "agent"])
            notes_written.append(note.title)
        if export_trace_path:
            dataset = TraceDataset.from_agent_trace(trace, trace_id="super-suit-trace")
            dataset.to_jsonl(export_trace_path)
        if export_sft_path:
            dataset = TraceDataset.from_agent_trace(trace, trace_id="super-suit-trace")
            examples = dataset.to_sft_examples()
            Path(export_sft_path).parent.mkdir(parents=True, exist_ok=True)
            Path(export_sft_path).write_text("\n".join(json.dumps(example.to_dict(), ensure_ascii=False) for example in examples))
        active_endpoints = []
        if isinstance(self.backend, MultiEndpointBackend):
            active_endpoints = self.backend.backend_names
        return SuperSuitResult(
            task=task,
            prepared_task=prepared_task,
            compressed_context=compressed_context,
            final=trace.final,
            trace=trace,
            active_endpoints=active_endpoints,
            notes_written=notes_written,
        )


def _build_prepared_task(task: str, context: CompressedContext) -> str:
    parts = [
        "You are running inside LLM Foundry super-suit mode.",
        "Use the tools available to you to solve the task in multiple steps if needed.",
        "Use memory notes, workspace, and verification instead of guessing.",
        "",
        f"TASK:\n{task}",
        "",
        context.to_prompt(),
    ]
    return "\n".join(part for part in parts if part is not None).strip()
