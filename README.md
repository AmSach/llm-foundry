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
