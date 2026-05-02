# LLM Foundry use cases

## 1. Build a model from scratch

Use the scratch training path to validate tokenization, batching, loss curves, checkpointing, and an inference loop before investing in large-scale training.

## 2. Wrap an existing model

Connect any OpenAI-compatible endpoint, local Hugging Face model, or a multi-endpoint API bundle, then add reflection, compression, memory, and safety controls without changing the model itself.

## 3. Add a memory system

Store important facts in the Obsidian-like vault, then compress long conversations into compact memory notes before they hit the context limit.

## 4. Export agent traces

Turn tool-use traces into supervised fine-tuning examples so real agent behavior can become training data.

## 5. Run reasoning overlays

Use reflection, counterfactual verification, and consensus decoding to reduce brittle single-sample errors.

## 6. Compare models consistently

Run the same prompts through different backends and evaluate them with the same scoring rules.

## 7. Run harnesses

Use the bundled harnesses to test reasoning, coding, tool use, and memory behavior before deploying a model.

## 8. Multi-endpoint orchestration

Configure several API endpoints in one JSON file, then let LLM Foundry fail over between them or round-robin across them when you want redundancy or cost control.

## Core commands

```bash
python -m llm_foundry smoke-test
python -m llm_foundry compress --task "Summarize this" --transcript-file path/to/transcript.txt
python -m llm_foundry agent --task "Search the repo and answer"
python -m llm_foundry benchmark --backend echo --output benchmark.json --markdown benchmark.md
python -m llm_foundry train-scratch --corpus examples/corpus.txt --steps 100 --context 64 --d-model 128
```

## Multi-endpoint example

```json
{
  "endpoints": [
    {
      "name": "fast",
      "base_url": "https://api.example.com/v1",
      "model": "model-a",
      "api_key_env": "MODEL_A_API_KEY"
    },
    {
      "name": "backup",
      "base_url": "https://api.backup.com/v1",
      "model": "model-b",
      "api_key_env": "MODEL_B_API_KEY"
    }
  ]
}
```

Then run:

```bash
python -m llm_foundry super-suit \
  --api-endpoints-file endpoints.json \
  --api-strategy failover \
  --task "Do the whole workflow"
```

## Training scripts

- `scripts/train.py` for config-driven training runs
- `scripts/run_benchmarks.py` for the reasoning, coding, tool-use, and memory harnesses
- `scripts/export_traces.py` for converting agent traces into supervised training examples
