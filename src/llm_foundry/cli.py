from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import build_backend
from .evaluation import EvaluationItem, EvaluationSuite
from .reasoning import ReflectionEngine
from .safety import SafetyLayer
from .training import train_from_text


PROJECT_NAME = "aegis-foundry"
VERSION = "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROJECT_NAME)
    parser.add_argument("--version", action="version", version=f"{PROJECT_NAME} {VERSION}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo")
    demo.add_argument("--backend", default="echo")
    demo.add_argument("--model", default=None)
    demo.add_argument("--prompt", required=True)

    eval_cmd = sub.add_parser("eval")
    eval_cmd.add_argument("--backend", default="echo")
    eval_cmd.add_argument("--model", default=None)
    eval_cmd.add_argument("--prompt", action="append", required=True)

    smoke = sub.add_parser("smoke-test")
    smoke.add_argument("--corpus", default="examples/corpus.txt")

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
        score = safety.score(args.prompt, result.final)
        print(result.final)
        print(f"SAFETY delayed_harm_risk={score.delayed_harm_risk:.2f} causal_credit={score.causal_credit:.2f}")
        return
    if args.cmd == "eval":
        backend = build_backend(args.backend, args.model)
        suite = EvaluationSuite(backend)
        results = suite.run(EvaluationItem(prompt=prompt) for prompt in args.prompt)
        for item in results:
            print(f"allowed={item.allowed} risk={item.delayed_harm_risk:.2f} prompt={item.prompt}")
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


if __name__ == "__main__":
    main()
