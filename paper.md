# LLM Foundry: A Modular Framework for Training, Integrating, Compressing, Remembering, and Evaluating Language Models

## Abstract

LLM Foundry is a modular system for language model work that treats the model itself as only one part of the stack. Real systems need more than a language model. They need tokenization, configuration, memory compression, retrieval, tool use, trace logging, training scripts, and repeatable evaluation. This project combines all of those pieces into a framework that can be attached to an existing model or used as the scaffold for training a new one. The design goal is practical rather than magical. It is not pretending to be a frontier-scale base model by itself. Instead, it is the control plane around a model, with explicit modules for compression, memory, reasoning, safety, dataset export, and benchmark harnesses.

This paper explains the system in two ways at once. First, it gives a plain-language explanation that a non-technical reader can follow. Second, it gives a detailed engineering description that a practitioner can use to extend the system into a larger model stack. The central idea is that useful intelligence in production comes from orchestration, not only from the raw neural network.

## 1. What problem are we solving?

A language model by itself is like a brilliant student with no notebook, no memory, no calculator, and no way to check its work. It may answer quickly, but it can forget, drift, and make expensive mistakes. If you want a system that does real work, you need the model plus the surrounding machinery.

LLM Foundry is built for that surrounding machinery.

The system addresses seven recurring needs:

1. **Model configuration** so the system has a clear blueprint.
2. **Tokenizer support** so text is converted into tokens efficiently.
3. **Compression** so long conversations are reduced to high-value context.
4. **Memory** so important facts can be recovered later.
5. **Agent traces** so tool use can become training data.
6. **Evaluation harnesses** so the system can be tested in consistent ways.
7. **Integration adapters** so the same stack can sit on top of many models.

## 2. Plain-English overview

Think of the system like a workshop.

- The model is the worker.
- The tokenizer is the translator.
- The compression engine is the editor that trims waste.
- The memory vault is the notebook.
- The agent runtime is the person who can use tools.
- The trace dataset is the logbook.
- The harnesses are the exam papers.

If you only have a worker, you get random work.
If you also have notebooks, editors, and exams, you get a system that can improve.

## 3. Architectural overview

The codebase is split into modules with clear responsibilities.

### 3.1 Adapter layer

The adapter layer lets the framework talk to different model sources through one interface. The contract is simple: `generate(prompt) -> text`.

Supported styles include:

- deterministic local backends for testing
- OpenAI-compatible HTTP endpoints
- Hugging Face local models
- sequence and scripted backends for reproducible harnesses

This makes the framework model agnostic. A model can be swapped without rewriting the rest of the stack.

### 3.2 Model configuration layer

The configuration module defines the shape of the system:

- model size
- depth and width
- context length
- output length
- tokenizer type
- whether memory is enabled
- how many tools the agent may call
- how much compression is applied

This is important because production systems need a blueprint. Without configuration, every experiment becomes an ad hoc script.

### 3.3 Tokenizer layer

Tokenization is the conversion from text to IDs.

The framework supports three practical tokenizer modes:

- **byte tokenization** for extreme robustness and tiny dependencies
- **character tokenization** for easy scratch-scale experiments
- **Hugging Face tokenizer integration** for compatibility with real model vocabularies

The tokenizer is the smallest unit of compatibility. If a model can share the tokenizer interface, it can share the rest of the stack.

### 3.4 Compression and memory layer

The compression engine is the key part of the requested memory system.

Its job is to take a long transcript and reduce it to a compact context pack. It does this by:

1. ranking sentences by relevance
2. extracting salient facts
3. extracting action items
4. extracting open questions
5. retrieving relevant notes from the Obsidian-like vault
6. assembling the result into a smaller prompt payload

The memory vault stores notes as markdown files with metadata. This is deliberate. Markdown is easy to inspect, easy to version, and compatible with human editing.

The effect is an Obsidian-like workflow. Important facts become notes. Notes become searchable memory. Memory becomes compressed prompt context.

### 3.5 Reasoning layer

The reasoning layer adds self-checks.

The current modules include:

- reflection, which drafts then critiques then revises
- counterfactual verification, which asks whether an answer still makes sense under alternate outcomes
- majority vote consensus, which samples several variants and selects the most common answer
- cascade reasoning, which uses a draft backend and a judge backend

These are not replacement intelligence. They are scaffolding that makes the model less fragile.

### 3.6 Agent layer

The agent runtime can interpret JSON tool calls, execute tools, and record traces.

This matters because many real tasks are not pure text generation. Real tasks include searching, reading, calculating, and making structured decisions.

The agent layer currently supports tool classes like:

- workspace listing
- workspace reading
- workspace search
- safe arithmetic

That is enough to demonstrate the pattern and to export traces for later training.

### 3.7 Dataset and trace layer

Agent traces are captured as structured records.

Each record contains:

- the task
- each tool call
- each observation
- the final answer

These traces can be exported into supervised fine-tuning examples. That is useful because actual agent behavior becomes training material.

### 3.8 Evaluation layer

The system includes harnesses for four broad skill areas:

- reasoning
- coding
- tool use
- memory

Each harness runs a consistent set of tests and emits JSON and Markdown output. The purpose is not to claim final truth. The purpose is to make model comparison repeatable.

## 4. The compression engine in detail

The compression engine is designed to save tokens without destroying meaning.

### 4.1 Why compression matters

Long conversations are expensive. If you keep everything, the model wastes context on repeated text, filler, and irrelevant details. If you keep nothing, the model forgets what matters.

Compression tries to sit between those extremes.

### 4.2 Compression algorithm

The current algorithm is intentionally simple, fast, and inspectable.

It works like this:

- clean the transcript line by line
- score sentences by relevance to the task
- keep the most relevant sentences
- extract action words and important statements
- extract questions that still need answers
- retrieve relevant notes from the memory vault
- combine all of that into a compressed context pack

This is not magic summarization. It is a controlled reduction of information.

### 4.3 Obsidian-like memory

Obsidian is popular because people like notes they can read, edit, link, and search. The memory vault follows that philosophy.

Each note has:

- title
- body
- tags
- links
- timestamps
- source metadata

Notes are saved in markdown so they stay portable.

### 4.4 How token efficiency is improved

Token efficiency comes from three sources:

1. shorter prompt packets
2. retrieval of only relevant notes
3. removal of repeated transcript clutter

This reduces context waste and makes the system more usable with smaller or cheaper models.

## 5. Model integration with any model

The framework is designed so that the model is not special.

Any model can be connected if it can answer prompts through one of these paths:

- local backend
- OpenAI-compatible backend
- Hugging Face backend
- scripted backend for testing

Once the model is connected, it automatically gets access to:

- compression
- memory
- reasoning overlays
- agent tool calls
- evaluation harnesses
- trace export

This is the main integration idea. The system is model agnostic.

## 6. Training from scratch and adapting existing models

The project supports both directions.

### 6.1 Training from scratch

The scratch training path is useful for:

- validating tokenization
- checking data pipelines
- building small toy models
- prototyping architecture changes

It is intentionally small enough to run locally.

### 6.2 Adapting existing models

Existing models can be wrapped rather than retrained.

That means the system can be used in production as a control layer even when the base model is external.

This is often the most practical path because the biggest gains usually come from orchestration, memory, evaluation, and tool use.

## 7. Agent trace datasets

A very important part of the system is that it can learn from itself.

When an agent runs, it produces a trace. That trace is structured data. Structured data can be turned into training examples.

That creates a loop:

1. agent acts
2. trace is recorded
3. traces are exported
4. traces become training data
5. a stronger model is trained
6. the stronger model generates better traces

This is a practical way to grow capability without hand-writing every example.

## 8. Evaluation harnesses

The harnesses measure different skills.

### 8.1 Reasoning harness
Tests whether the model can explain, verify, and maintain consistency.

### 8.2 Coding harness
Tests whether the model can produce code-like outputs.

### 8.3 Tool use harness
Tests whether the agent can call tools and finish a task.

### 8.4 Memory harness
Tests whether the compression and note vault actually reduce and retrieve context effectively.

This matters because a model that sounds smart but fails these harnesses is not a useful system.

## 9. What is actually built right now?

The current repository is a real framework, not a single model checkpoint.

It includes:

- model config
- tokenizer support
- scratch training helpers
- model adapters
- compression engine
- memory vault
- agent runtime
- trace dataset export
- reasoning overlays
- benchmark and harness scripts
- markdown documentation for GitHub

It does not yet include a frontier-trained base model. That would require major compute, data, and training infrastructure.

But the framework now contains the machinery that a real model project needs.

## 10. Why this is useful for future large models

If you later train or attach a stronger base model, you do not need to rebuild the scaffolding.

You can keep the same:

- config format
- tokenizer contract
- memory format
- trace dataset format
- harnesses
- benchmark reports
- agent protocol

That means the system is future-proofed for bigger models.

## 11. Practical deployment path

A realistic production path looks like this:

1. choose a strong base model
2. connect it with an adapter
3. enable the memory vault
4. enable compression
5. enable tool use
6. run the harnesses
7. export traces from real usage
8. fine-tune on those traces
9. repeat

That loop is how a model becomes more useful over time.

## 12. Limitations

This framework is useful, but it is not the end of the story.

The remaining hard problems are:

- scaling to much larger models
- improving factuality at long horizons
- making memory retrieval more semantically precise
- training the reasoning overlays jointly with the base model
- benchmarking against frontier systems at real scale

Those are the next research steps, not the current completion point.

## 13. Conclusion

LLM Foundry is a modular operating system for language models.

It helps with:

- building
- adapting
- compressing
- remembering
- reasoning
- acting
- evaluating
- exporting traces

In simple words, it turns a model from a chat engine into a working system.

## References

[1] Vaswani et al. Attention Is All You Need. https://arxiv.org/abs/1706.03762

[2] Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. https://arxiv.org/abs/2203.11171

[3] Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning. https://arxiv.org/abs/2303.11366

[4] Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models. https://arxiv.org/abs/2210.03629

[5] Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. https://arxiv.org/abs/2005.11401
