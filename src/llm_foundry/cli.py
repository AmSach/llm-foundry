from __future__ import annotations

import argparse
from pathlib import Path

from .core import ReflectionEngine, SafetyLayer, build_backend
from .scratch import train_from_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-foundry")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo")
    demo.add_argument("--backend", default="echo")
    demo.add_argument("--model", default=None)
    demo.add_argument("--prompt", required=True)

    train = sub.add_parser("train-scratch")
    train.add_argument("--corpus", required=True)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--context", type=int, default=64)
    train.add_argument("--d-model", type=int, default=128)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "demo":
        backend = build_backend(args.backend, args.model)
        engine = ReflectionEngine(backend)
        safety = SafetyLayer()
        result = engine.answer(args.prompt)
        score = safety.score(args.prompt, result["final"])
        print(result["final"])
        print(f"\nSAFETY delayed_harm_risk={score.delayed_harm_risk:.2f} causal_credit={score.causal_credit:.2f}")
        return
    if args.cmd == "train-scratch":
        result = train_from_text(args.corpus, args.steps, args.context, args.d_model)
        print(result)
        return


if __name__ == "__main__":
    main()
