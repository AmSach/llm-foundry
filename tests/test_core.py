from llm_foundry.core import EchoBackend, ReflectionEngine, SafetyLayer, RewardShaper


def test_safety_scores_known_harm_terms():
    safety = SafetyLayer()
    score = safety.score("do something", "This would cause harm and exploit a system")
    assert score.delayed_harm_risk > 0
    assert not safety.allow(score)


def test_reflection_engine_runs():
    engine = ReflectionEngine(EchoBackend())
    result = engine.answer("Say hi")
    assert result["final"] == result["draft"]


def test_reward_shaper_returns_float():
    safety = SafetyLayer()
    shaper = RewardShaper(safety)
    reward = shaper.shaped_reward("prompt", "safe response", 1.0)
    assert isinstance(reward, float)
