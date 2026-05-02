# LLM Foundry

A modular framework for two jobs:

1. train a language model from scratch at toy or research scale
2. wrap, steer, evaluate, compress, and safety-check existing LLMs

LLM Foundry is the production-facing name for the system. It is designed as a reusable control plane for model creation, model integration, reasoning overlays, benchmark generation, memory compression, and safety scoring. The same stack can sit on top of a scratch-trained model, a local Hugging Face model, or a remote OpenAI-compatible endpoint.

## What it is now

This repo is a usable LLM framework, not a frontier model. It gives you:

- a backend abstraction for multiple model sources
- tokenizer support for byte, character, and Hugging Face vocabularies
- a scratch training path for local experiments
- reflection, counterfactual verification, consensus, and cascade reasoning
- a compression engine and Obsidian-like memory vault
- agent traces that can be exported as training data
- benchmark tooling that emits JSON and Markdown reports
- GitHub Actions CI for repeatable testing

## Use cases

- build a new model from scratch at small scale
- wrap an existing model with reflection, compression, memory, and safety checks
- compare candidate models under the same evaluation policy
- gate agent actions with delayed-harm scoring
- prototype research ideas before moving to distributed training
- generate benchmark artifacts for GitHub documentation

## Important reality check

This codebase does not yet compete with GPT-5.5 class or Claude-class frontier models on its own. To get there you would need scale in three places:

- data quality and data volume
- model size and training infrastructure
- much stronger reasoning and verification loops, usually trained jointly with the base model

What this repo does is give you the scaffolding for that work, plus benchmark and packaging hooks so you can iterate fast.

## Quick start

```bash
cd /home/workspace/Projects/llm-foundry
python -m llm_foundry smoke-test
```

## Benchmark suite

```bash
python -m llm_foundry benchmark --backend echo --output benchmark.json --markdown benchmark.md
```

## From scratch demo

```bash
python -m llm_foundry train-scratch --corpus examples/corpus.txt --steps 100 --context 64 --d-model 128
```

## Memory compression demo

```bash
python -m llm_foundry compress --task "Summarize this conversation" --transcript-file path/to/transcript.txt
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

- `src/llm_foundry/config.py` - model and runtime configuration
- `src/llm_foundry/tokenizer.py` - byte, character, and Hugging Face tokenizer adapters
- `src/llm_foundry/memory.py` - compression engine and Obsidian-like memory vault
- `src/llm_foundry/agent.py` - tool-using agent runtime
- `src/llm_foundry/datasets.py` - agent trace and SFT dataset formats
- `src/llm_foundry/harnesses.py` - reasoning, coding, tool-use, and memory harnesses
- `src/llm_foundry/adapters.py` - local, OpenAI-compatible, and Hugging Face model backends
- `src/llm_foundry/reasoning.py` - reflection, verification, and consensus helpers
- `src/llm_foundry/safety.py` - delayed-harm scoring and reward shaping
- `src/llm_foundry/training.py` - scratch-model training helpers and a small Transformer path
- `src/llm_foundry/model_training.py` - config-driven training entrypoints
- `src/llm_foundry/benchmarks.py` - benchmark cases, reports, and scoring
- `src/llm_foundry/cli.py` - command line entrypoint
- `.github/workflows/ci.yml` - GitHub Actions smoke test and benchmark workflow
- `scripts/` - training, benchmark, and trace export scripts
- `examples/` - runnable examples and toy corpora

## Documentation

- `file 'paper.md'` for the long technical research-style paper
- `file 'EXPLAINER.md'` for a plain-English explanation
- `file 'USAGE.md'` for practical commands and workflows
