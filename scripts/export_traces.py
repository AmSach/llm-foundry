from __future__ import annotations

import argparse
from pathlib import Path

from llm_foundry.datasets import TraceDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert agent trace JSONL into supervised fine-tuning examples")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = TraceDataset.from_jsonl(args.input)
    examples = dataset.to_sft_examples()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(f"{example.to_dict()}\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
