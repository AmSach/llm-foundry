from llm_foundry import ModelSuperSuit, SequenceBackend, SuperSuitConfig, ToolPolicy, ToolRegistry


def test_super_suit_runs_and_exports():
    backend = SequenceBackend([
        '{"tool":"workspace.list","arguments":{"path":"."}}',
        '{"tool":"final","arguments":{"answer":"done"}}',
    ])
    suit = ModelSuperSuit(
        backend,
        SuperSuitConfig(
            workspace_root="/home/workspace/Projects/llm-foundry",
            memory_root="/home/workspace/Projects/llm-foundry/memory-vault-test",
            max_steps=3,
            target_tokens=256,
        ),
    )
    result = suit.run("list the repo", save_note="test-note")
    assert result.final == "done"
    assert result.compressed_context.token_estimate_after <= 256
    assert result.notes_written
