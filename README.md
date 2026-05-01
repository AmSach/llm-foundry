# LLM Foundry

A modular framework for two jobs:

1. train a language model from scratch at toy or research scale
2. wrap, steer, evaluate, and safety-check existing LLMs

LLM Foundry is the production-facing name for the system. It is designed as a reusable control plane for model creation, model integration, reasoning overlays, and safety scoring. The same stack can sit on top of a scratch-trained model, a local Hugging Face model, or a remote OpenAI-compatible endpoint.

## Production-ready positioning

This repo is structured to behave like a real engineering platform rather than a one-off demo:

- backend abstraction for multiple model sources
- testable reasoning and safety modules
- small scratch-training path for local validation
- evaluation harness for repeatable checks
- CLI entrypoints that are easy to automate

## Use cases

- build a new model from scratch at small scale
- wrap an existing model with reflection and safety checks
- compare candidate models under the same evaluation policy
- gate agent actions with delayed-harm scoring
- prototype research ideas before moving to distributed training

## What it includes

- a backend interface for any text model
- an OpenAI-compatible adapter
- a local Hugging Face adapter
- a tiny from-scratch causal LM for demos and experiments
- delayed-reward and causal-attribution helpers
- self-consistency and reflection utilities
- an evaluation suite and CLI for quick experiments

## What it does not pretend to be

This is not a trillion-token training system out of the box. For that, you still need distributed compute, data pipelines, and serious infrastructure.

What this repo does is give you one clean place to plug in:
- training code for new models
- adapters for existing models
- safety layers and evaluation harnesses
- experiment scripts that are easy to extend

## Quick start

```bash
cd /home/workspace/Projects/llm-foundry
python -m pip install -e .
python -m llm_foundry smoke-test
```

## From scratch demo

```bash
python -m llm_foundry train-scratch --corpus examples/corpus.txt --steps 100 --context 64 --d-model 128
```

## Existing model integration

Use the OpenAI-compatible backend for anything that speaks the standard chat-completions API.

```bash
export OPENAI_API_KEY=...
python -m llm_foundry demo \
  --backend openai \
  --model gpt-4o-mini \
  --prompt "Give me three failure modes of delayed reward learning."
```

## Repo layout

- `src/llm_foundry/adapters.py` - local, OpenAI-compatible, and Hugging Face model backends
- `src/llm_foundry/reasoning.py` - reflection, verification, and consensus helpers
- `src/llm_foundry/safety.py` - delayed-harm scoring and reward shaping
- `src/llm_foundry/training.py` - scratch-model training helpers and a small Transformer path
- `src/llm_foundry/evaluation.py` - repeatable prompt evaluation
- `src/llm_foundry/cli.py` - command line entrypoint
- `examples/` - runnable examples and toy corpora

## GitHub

This repo is ready to publish. The source of truth is the GitHub repository, and the paper and usage docs live in the repo root for easy browsing.
