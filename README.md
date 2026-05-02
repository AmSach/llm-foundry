# LLM Foundry

LLM Foundry is the workshop around an LLM.

It is not the model itself. It is the memory, compression, tokenizer, tool use, traces, harnesses, and training plumbing that make a model useful.

## What it gives you

- a backend abstraction for many model sources
- tokenizer support for byte, character, and Hugging Face vocabularies
- a scratch training path for local experiments
- reflection, counterfactual verification, consensus, and cascade reasoning
- a compression engine and Obsidian-like memory vault
- agent traces that can be exported as training data
- benchmark tooling that emits JSON and Markdown reports
- harnesses for reasoning, coding, tool use, and memory
- a docs site in `docs/`

## What it is and is not

A plain LLM answers.
LLM Foundry helps a model remember, use tools, compress context, check itself, and leave behind traces for later training.

It is **not** yet a frontier model. It is the scaffold you would want around one.

## The important idea

The model is replaceable.
The workshop stays.

Any compatible model can plug into the same interface and immediately get:

- memory compression
- agent tools
- reflection and verification
- harnesses
- trace export
- benchmark reports

## Is it the most advanced model for long-term decision making?

No. That would be an overclaim.

What it is instead is the strongest practical workshop shape for building and improving a small smart model around long-horizon memory, tool use, and delayed-reward style workflows.

## How to use it

```bash
python -m llm_foundry smoke-test
python -m llm_foundry compress --task "Summarize" --transcript-file path/to/transcript.txt
python -m llm_foundry agent --task "Search the repo and answer"
python -m llm_foundry benchmark --backend echo --output benchmark.json --markdown benchmark.md
python -m llm_foundry train-scratch --corpus examples/corpus.txt --steps 100 --context 64 --d-model 128
```

## Existing model integration

Use the OpenAI-compatible backend for anything that speaks the standard chat-completions API.

```bash
export OPENAI_API_KEY=...
python -m llm_foundry demo --backend openai --model gpt-4o-mini --prompt "Explain delayed reward."
```

## Training your own small model

1. Prepare a clean text corpus.
2. Choose a tokenizer and config.
3. Run `train-scratch` for a toy model, or `train-model` for config-driven training.
4. Export traces from agent runs.
5. Fine-tune on those traces and optional reward signals.
6. Rerun harnesses and benchmarks.

## Docs

- `docs/index.html` for the homepage
- `paper.md` for the long technical paper
- `EXPLAINER.md` for a plain-English explanation
- `USAGE.md` for practical commands and workflows
