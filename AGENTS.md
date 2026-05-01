# Project memory

- Project: llm-foundry
- Goal: one framework that can train a small language model from scratch, adapt existing LLMs, and add safety/reasoning evaluation layers.
- Prefer Python for implementation.
- Keep the first version modular and runnable on a laptop for toy-scale demos, while leaving clear extension points for larger models.
- Use standard library first; optional heavy dependencies belong behind imports or extras.
- Keep examples honest about compute limits: large from-scratch LLM training is a distributed-training problem, not a single-machine script.
