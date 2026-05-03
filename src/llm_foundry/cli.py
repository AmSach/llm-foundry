from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import build_backend
from .agent import ToolPolicy, ToolRegistry
from .benchmark import BenchmarkSuite, default_benchmark_cases
from .config import ModelConfig
from .datasets import TraceDataset
from .evaluation import EvaluationItem, EvaluationSuite
from .harnesses import run_all_harnesses
from .memory import CompressionEngine, ObsidianMemoryVault
from .model_training import train_model_from_corpus
from .rag import build_embedding_index
from .reasoning import ReflectionEngine
from .safety import SafetyLayer
from .studio import run_studio
from .super_suit import ModelSuperSuit, SuperSuitConfig
from .training import train_from_text

PROJECT_NAME = "llm-foundry"
VERSION = "0.3.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROJECT_NAME)
    parser.add_argument("--version", action="version", version=f"{PROJECT_NAME} {VERSION}")
    sub = parser.add_subparsers(dest="cmd", required=False)

    demo = sub.add_parser("demo")
    demo.add_argument("--backend", default="echo")
    demo.add_argument("--model", default=None)
    demo.add_argument("--api-endpoints-file", default="")
    demo.add_argument("--api-endpoints-json", default="")
    demo.add_argument("--api-strategy", default="failover")
    demo.add_argument("--prompt", required=True)

    eval_cmd = sub.add_parser("eval")
    eval_cmd.add_argument("--backend", default="echo")
    eval_cmd.add_argument("--model", default=None)
    eval_cmd.add_argument("--api-endpoints-file", default="")
    eval_cmd.add_argument("--api-endpoints-json", default="")
    eval_cmd.add_argument("--api-strategy", default="failover")
    eval_cmd.add_argument("--prompt", action="append", required=True)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--backend", default="echo")
    benchmark.add_argument("--model", default=None)
    benchmark.add_argument("--api-endpoints-file", default="")
    benchmark.add_argument("--api-endpoints-json", default="")
    benchmark.add_argument("--api-strategy", default="failover")
    benchmark.add_argument("--output", default="benchmark.json")
    benchmark.add_argument("--markdown", default="benchmark.md")
    benchmark.add_argument("--case", action="append")
    benchmark.add_argument("--workspace-memory-root", default="")

    compress = sub.add_parser("compress")
    compress.add_argument("--task", required=True)
    compress.add_argument("--transcript-file", required=True)
    compress.add_argument("--memory-root", default="memory-vault")
    compress.add_argument("--memory-query", default="")
    compress.add_argument("--target-tokens", type=int, default=512)
    compress.add_argument("--save-note", default="")

    harness = sub.add_parser("harness")
    harness.add_argument("--backend", default="echo")
    harness.add_argument("--model", default=None)
    harness.add_argument("--api-endpoints-file", default="")
    harness.add_argument("--api-endpoints-json", default="")
    harness.add_argument("--api-strategy", default="failover")
    harness.add_argument("--workspace", default="/home/workspace/Projects/llm-foundry")
    harness.add_argument("--output-dir", default="reports")

    train_model = sub.add_parser("train-model")
    train_model.add_argument("--corpus", required=True)
    train_model.add_argument("--config", default="")
    train_model.add_argument("--output", default="model-run.json")
    train_model.add_argument("--tokenizer-kind", default="byte")
    train_model.add_argument("--tokenizer-name", default=None)
    train_model.add_argument("--context-length", type=int, default=2048)
    train_model.add_argument("--d-model", type=int, default=256)
    train_model.add_argument("--steps", type=int, default=100)

    agent = sub.add_parser("agent")
    agent.add_argument("--backend", default="echo")
    agent.add_argument("--model", default=None)
    agent.add_argument("--api-endpoints-file", default="")
    agent.add_argument("--api-endpoints-json", default="")
    agent.add_argument("--api-strategy", default="failover")
    agent.add_argument("--workspace", default="/home/workspace/Projects/llm-foundry")
    agent.add_argument("--task", required=True)
    agent.add_argument("--policy", default="safe")
    agent.add_argument("--max-steps", type=int, default=6)
    agent.add_argument("--output-trace", default="")
    agent.add_argument("--export-sft", default="")

    supersuit = sub.add_parser("super-suit")
    supersuit.add_argument("--backend", default="echo")
    supersuit.add_argument("--model", default=None)
    supersuit.add_argument("--api-endpoints-file", default="")
    supersuit.add_argument("--api-endpoints-json", default="")
    supersuit.add_argument("--api-strategy", default="failover")
    supersuit.add_argument("--workspace", default="/home/workspace/Projects/llm-foundry")
    supersuit.add_argument("--memory-root", default="memory-vault")
    supersuit.add_argument("--task", required=True)
    supersuit.add_argument("--memory-query", default="")
    supersuit.add_argument("--save-note", default="")
    supersuit.add_argument("--output-trace", default="")
    supersuit.add_argument("--export-sft", default="")
    supersuit.add_argument("--output-json", default="")
    supersuit.add_argument("--allow-web-search", action="store_true")
    supersuit.add_argument("--allow-web-fetch", action="store_true")
    supersuit.add_argument("--allow-github-api", action="store_true")
    supersuit.add_argument("--allow-shell", action="store_true")
    supersuit.add_argument("--max-steps", type=int, default=8)
    supersuit.add_argument("--target-tokens", type=int, default=512)

    studio = sub.add_parser("studio")

    train = sub.add_parser("train-scratch")
    train.add_argument("--corpus", required=True)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--context", type=int, default=64)
    train.add_argument("--d-model", type=int, default=128)

    index = sub.add_parser("index")
    index.add_argument("--root", required=True)
    index.add_argument("--output", default="embedding-index.json")
    index.add_argument("--query", default="")
    index.add_argument("--top-k", type=int, default=5)

    return parser


def _backend_kwargs(args: argparse.Namespace) -> dict:
    return {
        "api_endpoints_file": getattr(args, "api_endpoints_file", "") or None,
        "api_endpoints_json": getattr(args, "api_endpoints_json", "") or None,
        "api_strategy": getattr(args, "api_strategy", "failover"),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd is None or args.cmd == "studio":
        run_studio(backend_factory=lambda kind, model, **kwargs: build_backend(kind, model, **kwargs))
        return
    if args.cmd == "demo":
        backend = build_backend(args.backend, args.model, **_backend_kwargs(args))
        engine = ReflectionEngine(backend)
        safety = SafetyLayer()
        result = engine.answer(args.prompt)
        score = safety.score(args.prompt, result.final)
        print(result.final)
        print(f"SAFETY delayed_harm_risk={score.delayed_harm_risk:.2f} causal_credit={score.causal_credit:.2f}")
        return
    if args.cmd == "eval":
        backend = build_backend(args.backend, args.model, **_backend_kwargs(args))
        suite = EvaluationSuite(backend)
        results = suite.run(EvaluationItem(prompt=prompt) for prompt in args.prompt)
        for item in results:
            print(f"allowed={item.allowed} risk={item.delayed_harm_risk:.2f} prompt={item.prompt}")
        return
    if args.cmd == "benchmark":
        backend = build_backend(args.backend, args.model, **_backend_kwargs(args))
        suite = BenchmarkSuite(backend)
        cases = default_benchmark_cases()
        if args.case:
            selected = {name.strip() for name in args.case}
            cases = [case for case in cases if case.name in selected]
        report = suite.run(cases)
        json_path = report.write_json(args.output)
        md_path = report.write_markdown(args.markdown)
        print(f"total={report.total} passed={report.passed} pass_rate={report.pass_rate:.2%} mean_risk={report.mean_risk:.3f}")
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
        if args.workspace_memory_root:
            idx = build_embedding_index(args.workspace_memory_root)
            print(f"memory_indexed={len(idx.documents)}")
        return
    if args.cmd == "compress":
        transcript = Path(args.transcript_file).read_text().splitlines()
        vault = ObsidianMemoryVault(args.memory_root)
        engine = CompressionEngine(vault=vault)
        context = engine.compress_transcript(
            task=args.task,
            transcript=transcript,
            memory_query=args.memory_query or args.task,
            target_tokens=args.target_tokens,
        )
        if args.save_note:
            note = engine.save_context_note(context, title=args.save_note)
            if note:
                print(f"saved-note={note.title}")
        print(context.to_prompt())
        print(f"before_tokens={context.token_estimate_before} after_tokens={context.token_estimate_after}")
        return
    if args.cmd == "harness":
        backend = build_backend(args.backend, args.model, **_backend_kwargs(args))
        reports = run_all_harnesses(backend, args.workspace)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, report in reports.items():
            report.write_json(output_dir / f"{name}.json")
            report.write_markdown(output_dir / f"{name}.md")
            print(f"{name} {report.passed}/{report.total} {report.pass_rate:.2%}")
        return
    if args.cmd == "train-model":
        if args.config:
            config = ModelConfig.load(args.config)
        else:
            config = ModelConfig(context_length=args.context_length, d_model=args.d_model, training_steps=args.steps)
            config.tokenizer.kind = args.tokenizer_kind
            config.tokenizer.model_name = args.tokenizer_name
        run = train_model_from_corpus(args.corpus, config)
        Path(args.output).write_text(str(run))
        print(run)
        return
    if args.cmd == "agent":
        backend = build_backend(args.backend, args.model, **_backend_kwargs(args))
        policy = ToolPolicy(allow_web_fetch=args.policy != "safe", allow_web_search=args.policy != "safe", allow_github_api=args.policy != "safe", allow_shell=args.policy == "full")
        tools = ToolRegistry(workspace_root=args.workspace, policy=policy)
        runtime = AgentRuntime(backend=backend, tools=tools, max_steps=args.max_steps)
        trace = runtime.run(args.task)
        print(trace.final)
        if args.output_trace:
            trace_data = TraceDataset.from_agent_trace(trace, trace_id="cli-agent-trace")
            Path(args.output_trace).write_text(json.dumps([record.to_dict() for record in trace_data.records], indent=2, ensure_ascii=False))
        if args.export_sft:
            dataset = TraceDataset.from_agent_trace(trace, trace_id="cli-agent-trace")
            examples = dataset.to_sft_examples()
            Path(args.export_sft).write_text("\n".join(json.dumps(example.to_dict(), ensure_ascii=False) for example in examples))
        return
    if args.cmd == "super-suit":
        backend = build_backend(args.backend, args.model, **_backend_kwargs(args))
        config = SuperSuitConfig(
            workspace_root=args.workspace,
            memory_root=args.memory_root,
            max_steps=args.max_steps,
            target_tokens=args.target_tokens,
            allow_web_fetch=args.allow_web_fetch,
            allow_web_search=args.allow_web_search,
            allow_github_api=args.allow_github_api,
            allow_shell=args.allow_shell,
            api_endpoints_file=args.api_endpoints_file or None,
            api_endpoints_json=args.api_endpoints_json or None,
            api_strategy=args.api_strategy,
        )
        suit = ModelSuperSuit(backend, config)
        result = suit.run(
            task=args.task,
            memory_query=args.memory_query,
            save_note=args.save_note,
            export_trace_path=args.output_trace,
            export_sft_path=args.export_sft,
        )
        print(result.final)
        print(f"tokens_before={result.compressed_context.token_estimate_before} tokens_after={result.compressed_context.token_estimate_after}")
        if result.active_endpoints:
            print("endpoints=" + ",".join(result.active_endpoints))
        if args.output_json:
            result.write_json(args.output_json)
        return
    if args.cmd == "smoke-test":
        corpus = Path(args.corpus)
        backend = build_backend("echo")
        engine = ReflectionEngine(backend)
        result = engine.answer("smoke test")
        print(result.final)
        print(train_from_text(str(corpus), 2, 8, 32))
        return
    if args.cmd == "train-scratch":
        result = train_from_text(args.corpus, args.steps, args.context, args.d_model)
        print(result)
        return
    if args.cmd == "index":
        index = build_embedding_index(args.root)
        path = index.save(args.output)
        print(f"wrote {path}")
        if args.query:
            for item in index.search(args.query, top_k=args.top_k):
                print(f"{item.path}: {item.score:.3f}: {item.text[:140]}")
        return
        print(train_from_text(str(corpus), 2, 8, 32))
        return
    if args.cmd == "train-scratch":
        result = train_from_text(args.corpus, args.steps, args.context, args.d_model)
        print(result)
        return


if __name__ == "__main__":
    main()
