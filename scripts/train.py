from __future__ import annotations

import argparse
from pathlib import Path

from llm_foundry.config import ModelConfig, TokenizerConfig
from llm_foundry.tokenizer import build_tokenizer, TokenizerConfigView
from llm_foundry.training import train_model_from_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small model from a text corpus")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--output", default="model.json")
    parser.add_argument("--tokenizer-kind", default="byte")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.config:
        config = ModelConfig.load(args.config)
    else:
        config = ModelConfig(context_length=args.context_length, d_model=args.d_model, training_steps=args.steps)
        config.tokenizer = TokenizerConfig(kind=args.tokenizer_kind, model_name=args.tokenizer_name)
    result = train_model_from_corpus(Path(args.corpus), config)
    Path(args.output).write_text(str(result))
    print(result)


if __name__ == "__main__":
    main()
