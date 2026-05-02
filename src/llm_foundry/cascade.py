from __future__ import annotations

from dataclasses import dataclass

from .adapters import TextBackend


@dataclass
class CascadeResult:
    draft: str
    critique: str
    final: str
    used_revision: bool


class CascadeReasoner:
    def __init__(self, draft_backend: TextBackend, judge_backend: TextBackend | None = None) -> None:
        self.draft_backend = draft_backend
        self.judge_backend = judge_backend or draft_backend

    def answer(self, prompt: str) -> CascadeResult:
        draft = self.draft_backend.generate(prompt)
        critique = self.judge_backend.generate(
            "Judge whether the draft is correct, complete, and safe. Reply with KEEP or REVISE and a short reason.\n\n"
            f"PROMPT:\n{prompt}\n\nDRAFT:\n{draft}"
        )
        needs_revision = "revise" in critique.lower()
        if needs_revision:
            final = self.judge_backend.generate(
                "Revise the answer using the critique. Keep it short and precise.\n\n"
                f"PROMPT:\n{prompt}\n\nDRAFT:\n{draft}\n\nCRITIQUE:\n{critique}"
            )
        else:
            final = draft
        return CascadeResult(draft=draft, critique=critique, final=final, used_revision=needs_revision)
