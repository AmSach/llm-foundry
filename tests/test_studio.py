from llm_foundry import MultiEndpointBackend, SequenceBackend, build_backend, run_studio


def test_studio_runs_with_scripted_input():
    backend = SequenceBackend(['{"tool":"final","arguments":{"answer":"done"}}'])
    inputs = iter([
        "research this repo",
        "echo",
        "",
        "failover",
        "memory-vault-test",
        "/home/workspace/Projects/llm-foundry",
        "256",
        "3",
        "n",
        "n",
        "n",
        "n",
        "",
        "",
        "",
        "",
        "",
        "quit",
    ])
    outputs: list[str] = []

    def fake_input(prompt: str) -> str:
        outputs.append(prompt)
        return next(inputs)

    def fake_print(*args, **kwargs):
        outputs.append(" ".join(str(arg) for arg in args))

    result = run_studio(
        input_fn=fake_input,
        print_fn=fake_print,
        backend_factory=lambda kind, model, **kwargs: backend,
    )
    assert result.final == "done"
    assert result.mode == "super-suit"
    assert any("What do you want to do?" in item for item in outputs)


def test_multi_endpoint_backend_builds_from_json():
    payload = """
    {
      "endpoints": [
        {"name": "primary", "base_url": "https://api.one/v1", "model": "m1", "api_key_env": "KEY_ONE"},
        {"name": "backup", "base_url": "https://api.two/v1", "model": "m2", "api_key_env": "KEY_TWO"}
      ]
    }
    """
    import os

    os.environ["KEY_ONE"] = "one"
    os.environ["KEY_TWO"] = "two"
    backend = build_backend("multi", api_endpoints_json=payload, api_strategy="round_robin")
    assert isinstance(backend, MultiEndpointBackend)
    assert backend.strategy == "round_robin"
    assert len(backend.candidates) == 2
