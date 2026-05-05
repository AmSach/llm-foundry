from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import json

from .adapters import build_backend
from .agent import ToolPolicy
from .benchmark import BenchmarkSuite, default_benchmark_cases
from .config import ModelConfig
from .datasets import TraceDataset
from .harnesses import run_all_harnesses
from .memory import CompressionEngine, ObsidianMemoryVault
from .model_training import train_model_from_corpus
from .super_suit import ModelSuperSuit, SuperSuitConfig
from .training import train_from_text


InputFn = Callable[[str], str]
PrintFn = Callable[..., None]
BackendFactory = Callable[..., object]


@dataclass
class StudioOutcome:
    mode: str = ""
    title: str = ""
    task: str = ""
    final: str = ""
    generated_files: list[str] = field(default_factory=list)
    active_endpoints: list[str] = field(default_factory=list)


@dataclass
class EndpointDraft:
    name: str
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 120
    temperature: float = 0.2

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
        }


def run_studio(
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
    backend_factory: BackendFactory = build_backend,
) -> StudioOutcome:
    print_fn("LLM Foundry Studio")
    print_fn("Type a task to run it, or type setup, benchmark, train, agent, proof, or quit.")
    print_fn("If you type a plain task, it runs the super-suit workflow.")
    outcome = StudioOutcome()
    while True:
        raw = _ask(input_fn, "\nWhat do you want to do? ").strip()
        if not raw:
            continue
        command = raw.lower()
        if command in {"quit", "exit", "q"}:
            print_fn("Goodbye.")
            return outcome
        if command == "setup":
            result = _setup_endpoints(input_fn, print_fn)
            outcome = result
            continue
        if command == "benchmark":
            result = _run_benchmark_flow(input_fn, print_fn, backend_factory)
            outcome = result
            continue
        if command == "train":
            result = _run_train_flow(input_fn, print_fn)
            outcome = result
            continue
        if command == "agent":
            result = _run_agent_flow(input_fn, print_fn, backend_factory)
            outcome = result
            continue
        if command == "proof":
            result = _run_proof_flow(input_fn, print_fn, backend_factory)
            outcome = result
            continue
        result = _run_super_suit_flow(raw, input_fn, print_fn, backend_factory)
        outcome = result


def _run_super_suit_flow(task: str, input_fn: InputFn, print_fn: PrintFn, backend_factory: BackendFactory) -> StudioOutcome:
    print_fn("Super-suit workflow")
    backend_kind = _ask(input_fn, "Backend [echo/qwen/openai/anthropic/hf/multi] [echo]: ", "echo").strip() or "echo"
    model = _ask(input_fn, "Model name [blank for default]: ", "").strip() or None
    api_strategy = _ask(input_fn, "Multi-endpoint strategy [failover/round_robin] [failover]: ", "failover").strip() or "failover"
    api_endpoints_file = ""
    api_endpoints_json = ""
    if backend_kind in {"multi", "openai-multi"}:
        loaded = _ask(input_fn, "API endpoints JSON file [blank to create one interactively]: ", "").strip()
        if loaded:
            api_endpoints_file = loaded
        else:
            api_endpoints_file = _setup_endpoints_file(input_fn, print_fn)
    memory_root = _ask(input_fn, "Memory root [memory-vault]: ", "memory-vault").strip() or "memory-vault"
    workspace_root = _ask(input_fn, "Workspace root [/home/workspace/Projects/llm-foundry]: ", "/home/workspace/Projects/llm-foundry").strip() or "/home/workspace/Projects/llm-foundry"
    target_tokens = int(_ask(input_fn, "Target tokens [512]: ", "512").strip() or "512")
    max_steps = int(_ask(input_fn, "Max agent steps [8]: ", "8").strip() or "8")
    allow_web_fetch = _ask_yes_no(input_fn, "Enable web fetch tools? [y/N]: ")
    allow_web_search = _ask_yes_no(input_fn, "Enable web search tools? [y/N]: ")
    allow_github_api = _ask_yes_no(input_fn, "Enable GitHub tools? [y/N]: ")
    allow_shell = _ask_yes_no(input_fn, "Enable shell tools? [y/N]: ")
    save_note = _ask(input_fn, "Save a memory note title [blank to skip]: ", "").strip()
    output_trace = _ask(input_fn, "Export trace JSONL path [blank to skip]: ", "").strip()
    export_sft = _ask(input_fn, "Export SFT JSONL path [blank to skip]: ", "").strip()
    output_json = _ask(input_fn, "Export final JSON summary path [blank to skip]: ", "").strip()
    memory_query = _ask(input_fn, "Memory query [blank uses task]: ", "").strip()

    backend = backend_factory(
        backend_kind,
        model,
        api_endpoints_file=api_endpoints_file or None,
        api_endpoints_json=api_endpoints_json or None,
        api_strategy=api_strategy,
    )
    suit = ModelSuperSuit(
        backend,
        SuperSuitConfig(
            workspace_root=workspace_root,
            memory_root=memory_root,
            max_steps=max_steps,
            target_tokens=target_tokens,
            allow_web_fetch=allow_web_fetch,
            allow_web_search=allow_web_search,
            allow_github_api=allow_github_api,
            allow_shell=allow_shell,
            api_endpoints_file=api_endpoints_file or None,
            api_endpoints_json=api_endpoints_json or None,
            api_strategy=api_strategy,
        ),
    )
    result = suit.run(
        task=task,
        memory_query=memory_query,
        save_note=save_note,
        export_trace_path=output_trace,
        export_sft_path=export_sft,
    )
    print_fn("")
    print_fn(result.final)
    print_fn(f"tokens_before={result.compressed_context.token_estimate_before} tokens_after={result.compressed_context.token_estimate_after}")
    if result.active_endpoints:
        print_fn("endpoints=" + ",".join(result.active_endpoints))
    generated_files = [path for path in [output_trace, export_sft, output_json] if path]
    if output_json:
        result.write_json(output_json)
    print_fn("Tip: type another task, or quit.")
    return StudioOutcome(
        mode="super-suit",
        title=save_note or "",
        task=task,
        final=result.final,
        generated_files=generated_files,
        active_endpoints=result.active_endpoints,
    )




def _run_proof_flow(input_fn: InputFn, print_fn: PrintFn, backend_factory: BackendFactory) -> StudioOutcome:
    provider = _ask(input_fn, "Provider [qwen/openai/anthropic/hf] [qwen]: ", "qwen").strip() or "qwen"
    model = _ask(input_fn, "Model [Qwen/Qwen2.5-0.5B-Instruct]: ", "Qwen/Qwen2.5-0.5B-Instruct").strip() or "Qwen/Qwen2.5-0.5B-Instruct"
    workspace = _ask(input_fn, "Workspace root [/home/workspace/Projects/llm-foundry]: ", "/home/workspace/Projects/llm-foundry").strip() or "/home/workspace/Projects/llm-foundry"
    question = _ask(input_fn, "Question [Use the tools to find where build_backend is defined, cite the exact file path, and explain the answer in one sentence.]: ", "Use the tools to find where build_backend is defined, cite the exact file path, and explain the answer in one sentence.").strip() or "Use the tools to find where build_backend is defined, cite the exact file path, and explain the answer in one sentence."
    backend = backend_factory(provider, model)
    tools = ToolRegistry(workspace_root=workspace, policy=ToolPolicy(allow_web_fetch=False, allow_web_search=False, allow_github_api=False, allow_shell=False))
    runtime = AgentRuntime(backend=backend, tools=tools, max_steps=4)
    trace = runtime.run(question)
    payload = {
        "provider": provider,
        "model": model,
        "workspace": workspace,
        "question": question,
        "final": trace.final,
        "steps": [step.__dict__ for step in trace.steps],
    }
    out = Path("proof-run.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print_fn("Proof run complete")
    print_fn(question)
    print_fn(trace.final)
    print_fn(str(out))
    return StudioOutcome(mode="proof", task=question, final=trace.final, generated_files=[str(out)])

def _run_agent_flow(input_fn: InputFn, print_fn: PrintFn, backend_factory: BackendFactory) -> StudioOutcome:
    task = _ask(input_fn, "Agent task: ").strip()
    backend_kind = _ask(input_fn, "Backend [echo/qwen/openai/anthropic/hf/multi] [echo]: ", "echo").strip() or "echo"
    model = _ask(input_fn, "Model name [blank for default]: ", "").strip() or None
    backend = backend_factory(backend_kind, model)
    tools = []
    print_fn("Agent mode ready. The existing agent runtime handles tool use and traces.")
    if task:
        print_fn(task)
    return StudioOutcome(mode="agent", task=task, final="", generated_files=tools)


def _run_benchmark_flow(input_fn: InputFn, print_fn: PrintFn, backend_factory: BackendFactory) -> StudioOutcome:
    backend_kind = _ask(input_fn, "Backend [echo/qwen/openai/anthropic/hf/multi] [echo]: ", "echo").strip() or "echo"
    model = _ask(input_fn, "Model name [blank for default]: ", "").strip() or None
    backend = backend_factory(backend_kind, model)
    suite = BenchmarkSuite(backend)
    report = suite.run(default_benchmark_cases())
    print_fn(f"Benchmark pass rate: {report.pass_rate:.2%}")
    print_fn("Wrote nothing because this is the interactive quick view. Use the benchmark command for files.")
    return StudioOutcome(mode="benchmark", final=f"{report.pass_rate:.2%}")


def _run_train_flow(input_fn: InputFn, print_fn: PrintFn) -> StudioOutcome:
    corpus = _ask(input_fn, "Corpus path: ").strip()
    context = int(_ask(input_fn, "Context size [64]: ", "64").strip() or "64")
    d_model = int(_ask(input_fn, "D model [128]: ", "128").strip() or "128")
    steps = int(_ask(input_fn, "Steps [100]: ", "100").strip() or "100")
    if not corpus:
        print_fn("No corpus provided.")
        return StudioOutcome(mode="train")
    result = train_from_text(corpus, steps, context, d_model)
    print_fn(str(result))
    return StudioOutcome(mode="train", task=corpus, final=str(result))


def _setup_endpoints(input_fn: InputFn, print_fn: PrintFn) -> StudioOutcome:
    path = _setup_endpoints_file(input_fn, print_fn)
    print_fn(f"Saved endpoint bundle to {path}")
    print_fn("Add the API keys to Settings > Advanced using the api_key_env names you chose.")
    return StudioOutcome(mode="setup", generated_files=[path])


def _setup_endpoints_file(input_fn: InputFn, print_fn: PrintFn) -> str:
    path = _ask(input_fn, "Output path for endpoint config [endpoints.json]: ", "endpoints.json").strip() or "endpoints.json"
    count = int(_ask(input_fn, "How many endpoints? [2]: ", "2").strip() or "2")
    drafts: list[EndpointDraft] = []
    for i in range(count):
        print_fn(f"Endpoint {i + 1}")
        name = _ask(input_fn, "  Name [primary/backup/etc]: ", f"endpoint-{i + 1}").strip() or f"endpoint-{i + 1}"
        base_url = _ask(input_fn, "  Base URL: ").strip()
        model = _ask(input_fn, "  Model name: ").strip()
        api_key_env = _ask(input_fn, "  API key env var [OPENAI_API_KEY]: ", "OPENAI_API_KEY").strip() or "OPENAI_API_KEY"
        timeout_seconds = int(_ask(input_fn, "  Timeout seconds [120]: ", "120").strip() or "120")
        temperature = float(_ask(input_fn, "  Temperature [0.2]: ", "0.2").strip() or "0.2")
        drafts.append(
            EndpointDraft(
                name=name,
                base_url=base_url,
                model=model,
                api_key_env=api_key_env,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
            )
        )
    payload = {"endpoints": [draft.to_dict() for draft in drafts]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def _ask(input_fn: InputFn, prompt: str, default: str | None = None) -> str:
    suffix = ""
    if default not in {None, ""}:
        suffix = f" [{default}]"
    value = input_fn(f"{prompt.rstrip()} {suffix}".rstrip())
    if value == "" and default is not None:
        return default
    return value


def _ask_yes_no(input_fn: InputFn, prompt: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    answer = input_fn(f"{prompt.rstrip()} [{default_text}] ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes", "true", "1"}
