# llm-foundry

A modular framework for two jobs:

1. train a language model from scratch at toy or research scale
2. wrap, steer, evaluate, and safety-check existing LLMs

This repo is intentionally split into small, swappable parts so the same safety and reasoning stack can sit on top of:
- a local model you train yourself
- an OpenAI-compatible endpoint
- a Hugging Face model loaded locally
- any future backend that can answer text prompts

## What it includes

- a backend interface for any text model
- an OpenAI-compatible adapter
- a local Hugging Face adapter
- a tiny from-scratch causal LM for demos and experiments
- delayed-reward and causal-attribution helpers
- self-consistency and reflection utilities
- a CLI for quick experiments

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
python -m llm_foundry.cli demo --backend echo --prompt "Explain delayed reward in one sentence."
```

## From scratch demo

```bash
python -m llm_foundry.cli train-scratch --corpus examples/corpus.txt --steps 100 --context 64 --d-model 128
```

## Existing model integration

Use the OpenAI-compatible backend for anything that speaks the standard chat-completions API.

```bash
export OPENAI_API_KEY=...
python -m llm_foundry.cli demo \
  --backend openai \
  --model gpt-4o-mini \
  --prompt "Give me three failure modes of delayed reward learning."
```

## Repo layout

- `src/llm_foundry/core.py` — model adapters, safety helpers, reasoning harness
- `src/llm_foundry/scratch.py` — tiny from-scratch causal LM and trainer
- `src/llm_foundry/cli.py` — command line entrypoint
- `examples/` — runnable examples and toy corpora

## GitHub

This repo is being prepared for GitHub now. Once GitHub auth is completed, it can be created as a remote repository and pushed immediately.
