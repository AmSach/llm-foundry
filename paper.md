# LLM Foundry: A Modular Framework for Training, Integrating, and Safeguarding Language Models

## Abstract

Large language models are useful when they are not only capable, but also easy to integrate, easy to test, and hard to misuse. This paper presents LLM Foundry, a modular framework that unifies three layers: scratch-scale language model training, drop-in adapters for existing models, and safety and reasoning overlays that can be used at inference time or inside agent loops. The design is intentionally practical. Instead of pretending that a single laptop project can train frontier-scale models, LLM Foundry cleanly separates small-scale from-scratch training from production integration with hosted or locally served models. The framework adds three system-level ideas that are implemented as composable modules: a reflection loop, a counterfactual verification layer, and a causal safety shaper. Together, these modules create a reusable control plane for reasoning quality, delayed-harm screening, and model-agnostic deployment.

## 1. Introduction

Transformer language models became the dominant architecture because self-attention made sequence modeling flexible and parallelizable [1]. Since then, the field has split into two practical tracks. One track is model creation, where a team wants to train or fine-tune a model. The other track is model operation, where a team wants to wrap an existing model with better reliability, evaluation, and safety. Most real deployments need both.

LLM Foundry is built around that reality. It does not assume that every user wants to train a base model from scratch. It also does not assume that all model work is just prompting a remote API. Instead, it provides a single framework that can host a tiny educational causal LM, connect to hosted APIs, connect to local Hugging Face models, and apply the same evaluation and safety logic everywhere.

## 2. Design Goals

The framework is organized around five goals:

1. **One interface for many models**. A single text backend protocol supports echo backends, OpenAI-compatible endpoints, and local Hugging Face inference.
2. **One set of safety controls**. The same delayed-harm and reward-shaping logic can be applied to any backend.
3. **One reasoning control loop**. Reflection, counterfactual checking, and consensus are separated into reusable modules.
4. **One scratch-scale training path**. A toy causal LM and a small Transformer path demonstrate how to train from text locally.
5. **One production path**. The same abstractions can be used in evaluation harnesses, agent systems, and service wrappers.

## 3. System Architecture

The system has four layers.

### 3.1 Backend layer

The backend layer is defined by a tiny protocol: `generate(prompt) -> text`. This keeps the rest of the stack model agnostic. The current implementation includes:

- `EchoBackend` for deterministic tests
- `OpenAICompatibleBackend` for hosted chat-completions APIs
- `HuggingFacePipelineBackend` for local model inference

### 3.2 Reasoning layer

The reasoning layer is built from three modules:

- `ReflectionEngine`, which generates a draft, critiques it, and revises it
- `CounterfactualVerifier`, which asks whether an answer still holds under alternative outcomes
- `MajorityVoteConsensus`, which aggregates multiple candidate answers

This is closely related to the literature on self-consistency and reflective language agents [2, 3].

### 3.3 Safety layer

The safety layer computes a simple but useful score with three outputs:

- `delayed_harm_risk`
- `causal_credit`
- `confidence`

The current implementation is heuristic, not a full causal model. It is meant to be a production-facing control surface, not a claim that keyword heuristics solve alignment. The reward shaper then combines the base reward with the safety score to penalize outputs that look risky.

### 3.4 Training layer

The training layer contains two paths:

- a tiny character-level causal LM for lightweight demos
- a small Transformer path for local research experiments when PyTorch is available

The training path is intentionally modest. It is useful for testing the framework, not for frontier-scale pretraining.

## 4. Our Algorithms

### 4.1 ReflectionEngine

ReflectionEngine implements a three-pass inference procedure:

1. draft an answer
2. critique the draft
3. rewrite using the critique

This is a lightweight self-improvement loop. It does not update model weights. Instead, it uses the model as both generator and reviewer. In practice, this is useful because many errors are not a lack of capability, but a lack of second-pass correction.

### 4.2 CounterfactualVerifier

CounterfactualVerifier asks whether an answer remains justified under alternate assumptions or interventions. Conceptually, it converts simple generation into a verification step. The purpose is not to compute exact causal truth. The purpose is to introduce a disciplined habit of checking whether an answer depends on a narrow path that could be wrong.

### 4.3 MajorityVoteConsensus

MajorityVoteConsensus samples multiple prompts or prompt variants and returns the most common answer. This is the simplest form of consensus decoding. It is cheap, robust, and easy to plug into systems where a single bad sample is risky.

### 4.4 SafetyLayer and RewardShaper

SafetyLayer scores text for risk and confidence. RewardShaper turns that score into a shaped scalar reward.

This is a practical approximation of delayed-harm mitigation. It does not require the agent to perfectly predict the full future. Instead, it supplies a policy-level bias against outputs that are likely to be harmful once they are operationalized.

## 5. Production Use Cases

LLM Foundry is useful in several settings.

### 5.1 Building a new language model

A team can use the scratch training path to validate data pipelines, tokenization, loss computation, and checkpointing before moving to distributed training.

### 5.2 Wrapping an existing model

A hosted or local model can be wrapped behind the backend interface and immediately gain reflection, verification, and safety scoring.

### 5.3 Agent control planes

In tool-using agents, the framework can gate action execution by scoring the model output before the action is allowed to proceed.

### 5.4 Evaluation harnesses

The evaluation layer can run prompts through any backend and record whether the model output is allowed, risky, or acceptable under the current safety policy.

### 5.5 Research prototypes

The framework is useful for studying delayed reward, counterfactual credit assignment, and post-generation verification without building an entire infrastructure stack first.

## 6. Why This Is Different

The novelty is not a single magic model architecture. The novelty is the control plane.

Most model stacks treat reasoning, safety, and training as unrelated tasks. LLM Foundry makes them composable.

That gives three advantages:

- the same safety logic works on scratch models and hosted models
- the same reasoning loop works on local inference and API-backed inference
- the same evaluation surface can be used during development and deployment

That is the part that can scale into a production system.

## 7. Limitations

The current codebase is a framework, not a frontier model.

The scratch training path is tiny by design. The safety layer is heuristic by design. The reflection layer relies on the underlying model's own ability to critique itself. These are strengths for iteration and deployment, but they are not substitutes for strong data, strong compute, and rigorous evaluation.

## 8. Conclusion

LLM Foundry turns language model work into a modular system instead of a pile of one-off scripts. It can be used to build a new model, wrap an existing one, evaluate both, and add safety and reasoning controls at the same time. That makes it a practical research platform and a production-oriented integration layer.

## References

[1] Vaswani et al. Attention Is All You Need. https://arxiv.org/abs/1706.03762

[2] Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. https://arxiv.org/abs/2203.11171

[3] Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning. https://arxiv.org/abs/2303.11366
