from __future__ import annotations

import argparse
from pathlib import Path
import json

from llm_foundry.adapters import build_backend
from llm_foundry.harnesses import run_all_harnesses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the benchmark and harness suite")
    parser.add_argument("--backend", default="echo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--workspace", default="/home/workspace/Projects/llm-foundry")
    parser.add_argument("--output-dir", default="reports")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = build_backend(args.backend, args.model)
    reports = run_all_harnesses(backend, args.workspace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, report in reports.items():
        json_path = report.write_json(output_dir / f"{name}.json")
        md_path = report.write_markdown(output_dir / f"{name}.md")
        summary[name] = report.to_dict()
        print(f"{name}: {report.passed}/{report.total} -> {json_path} {md_path}")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
