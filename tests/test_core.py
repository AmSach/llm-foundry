from llm_foundry import EchoBackend, FixedBackend, ReflectionEngine, RewardShaper, SafetyLayer, SequenceBackend, ToolRegistry, AgentRuntime, LocalRetriever, CascadeReasoner, build_backend
from llm_foundry.core import ReflectionEngine as CoreReflectionEngine
from llm_foundry.evaluation import EvaluationItem, EvaluationSuite


def test_safety_scores_known_harm_terms():
    safety = SafetyLayer()
    score = safety.score("do something", "This would cause harm and exploit a system")
    assert score.delayed_harm_risk > 0
    assert not safety.allow(score)


def test_reflection_engine_runs():
    engine = ReflectionEngine(EchoBackend())
    result = engine.answer("Say hi")
    assert result.final == result.draft


def test_core_reexports_work():
    engine = CoreReflectionEngine(EchoBackend())
    result = engine.answer("Say hi")
    assert result.final == result.draft


def test_reward_shaper_returns_float():
    safety = SafetyLayer()
    shaper = RewardShaper(safety)
    reward = shaper.shaped_reward("prompt", "safe response", 1.0)
    assert isinstance(reward, float)


def test_eval_suite_runs():
    suite = EvaluationSuite(EchoBackend())
    results = suite.run([EvaluationItem(prompt="hello")])
    assert len(results) == 1
    assert results[0].allowed is True


def test_build_backend_echo():
    backend = build_backend("echo")
    assert backend.generate("x") == "x"


def test_sequence_backend_returns_ordered_values():
    backend = SequenceBackend(["a", "b"], default_response="z")
    assert backend.generate("p") == "a"
    assert backend.generate("p") == "b"
    assert backend.generate("p") == "z"


def test_agent_runtime_can_call_tools():
    tools = ToolRegistry(workspace_root="/home/workspace/Projects/llm-foundry")
    backend = SequenceBackend([
        '{"tool":"workspace.list","arguments":{"path":"."}}',
        '{"tool":"final","arguments":{"answer":"done"}}',
    ])
    runtime = AgentRuntime(backend=backend, tools=tools, max_steps=3)
    trace = runtime.run("list the repo")
    assert trace.final == "done"
    assert trace.steps[0].action == "workspace.list"


def test_retriever_finds_documents():
    retriever = LocalRetriever("/home/workspace/Projects/llm-foundry")
    results = retriever.search("reflection")
    assert results


def test_cascade_reasoner_uses_revision():
    draft = FixedBackend("draft")
    judge = SequenceBackend(["REVISE because it is unclear", "final revised answer"])
    reasoner = CascadeReasoner(draft, judge)
    result = reasoner.answer("prompt")
    assert result.used_revision is True
    assert result.final == "final revised answer"
