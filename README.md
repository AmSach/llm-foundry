# LLM Foundry

LLM Foundry is the workshop around an LLM.

It is not the model itself. It is the memory, compression, tokenizer, tool use, traces, harnesses, training plumbing, and API orchestration that make a model useful.

## What it gives you

- a backend abstraction for many model sources
- tokenizer support 

## Provider options

- `qwen` for the built-in Hugging Face Qwen path
- `openai` for OpenAI-compatible BYOK providers
- `anthropic` for Anthropic BYOK providers
- `hf` for any Hugging Face local model
- `multi` or `openai-multi` for failover bundles

Example:

```bash
python -m llm_foundry demo --provider qwen --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hello"
```

## Proof mode

Use `proof` for a terminal-first demo that shows the question, tool call trace, final answer, and JSON artefact. It is designed for screenshots and file explorer use on Windows, macOS, and Linux.

Example:

```bash
python -m llm_foundry proof --provider qwen --model Qwen/Qwen2.5-0.5B-Instruct
```

## Proof images

- [Prompt + answer](assets/proofs/sourcecode-dev-proof/01-prompt-answer.png)
- [Agent trace](assets/proofs/sourcecode-dev-proof/02-agent-trace.png)
- [Tests](assets/proofs/sourcecode-dev-proof/03-tests.png)
- [Path handling](assets/proofs/sourcecode-dev-proof/04-paths.png)
- [Diff](assets/proofs/sourcecode-dev-proof/05-diff.png)
- [Terminal](assets/proofs/sourcecode-dev-proof/06-terminal.png)
