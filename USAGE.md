# LLM Foundry use cases

## 1. Build a model from scratch

Use the scratch training path to validate tokenization, batching, loss curves, checkpointing, and an inference loop before investing in large-scale training.

## 2. Wrap an existing model

Connect any OpenAI-compatible endpoint or local Hugging Face model, then add reflection and safety controls without changing the model itself.

## 3. Add a safety gate

Score each response with delayed-harm heuristics before letting the agent act, send, or commit a result.

## 4. Run reasoning overlays

Use reflection, counterfactual verification, and consensus decoding to reduce brittle single-sample errors.

## 5. Compare models consistently

Run the same prompts through different backends and evaluate them with the same scoring rules.
