from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import ast
import json
import math
import os
import re
from typing import Callable, Protocol

from .adapters import TextBackend


class ToolFn(Protocol):
    def __call__(self, **kwargs) -> str: ...


@dataclass(frozen=True)
class AgentAction:
    tool: str
    arguments: dict[str, object] = field(default_factory=dict)


@dataclass
class AgentStep:
    raw: str
    action: str
    observation: str = ""
    final: str = ""


@dataclass
class AgentTrace:
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    final: str = ""


class ToolRegistry:
    def __init__(self, workspace_root: str | Path = ".", tools: dict[str, ToolFn] | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.tools = tools or {}
        self.tools.setdefault("workspace.list", self._workspace_list)
        self.tools.setdefault("workspace.read", self._workspace_read)
        self.tools.setdefault("workspace.search", self._workspace_search)
        self.tools.setdefault("math.calc", self._math_calc)

    def describe(self) -> str:
        return "\n".join(sorted(self.tools))

    def run(self, action: AgentAction) -> str:
        if action.tool not in self.tools:
            raise ValueError(f"Unknown tool: {action.tool}")
        return self.tools[action.tool](**action.arguments)

    def _resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = (self.workspace_root / Path(value)).resolve()
        if self.workspace_root not in path.parents and path != self.workspace_root:
            raise ValueError("Path escapes workspace root")
        return path

    def _workspace_list(self, path: str = ".") -> str:
        target = self._resolve_path(path)
        if not target.exists():
            return f"missing: {path}"
        if target.is_file():
            return target.name
        items = sorted(entry.name for entry in target.iterdir())
        return "\n".join(items)

    def _workspace_read(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        target = self._resolve_path(path)
        text = target.read_text(errors="replace")
        lines = text.splitlines()
        start = max(1, start_line or 1)
        end = end_line or len(lines)
        return "\n".join(lines[start - 1 : end])

    def _workspace_search(self, query: str, path: str = ".", max_results: int = 20) -> str:
        root = self._resolve_path(path)
        if root.is_file():
            files = [root]
        else:
            files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches: list[str] = []
        for file in files:
            try:
                for i, line in enumerate(file.read_text(errors="ignore").splitlines(), start=1):
                    if pattern.search(line):
                        rel = file.relative_to(self.workspace_root)
                        matches.append(f"{rel}:{i}:{line.strip()}")
                        if len(matches) >= max_results:
                            return "\n".join(matches)
                        break
            except OSError:
                continue
        return "\n".join(matches) if matches else "no matches"

    def _math_calc(self, expression: str) -> str:
        return str(_safe_eval(expression))


@dataclass
class AgentRuntime:
    backend: TextBackend
    tools: ToolRegistry
    max_steps: int = 6

    def run(self, task: str) -> AgentTrace:
        transcript: list[str] = []
        trace = AgentTrace(task=task)
        for _ in range(self.max_steps):
            prompt = _agent_prompt(task, transcript, self.tools.describe())
            raw = self.backend.generate(prompt)
            action = _parse_action(raw)
            if action is None:
                trace.steps.append(AgentStep(raw=raw, action="final", final=raw))
                trace.final = raw
                return trace
            if action.tool == "final":
                final = str(action.arguments.get("answer", ""))
                trace.steps.append(AgentStep(raw=raw, action="final", final=final))
                trace.final = final
                return trace
            observation = self.tools.run(action)
            transcript.append(f"TOOL {action.tool} {json.dumps(action.arguments, ensure_ascii=False)}")
            transcript.append(f"OBSERVATION {observation}")
            trace.steps.append(AgentStep(raw=raw, action=action.tool, observation=observation))
        trace.final = trace.steps[-1].observation if trace.steps else ""
        return trace


def _agent_prompt(task: str, transcript: list[str], tools: str) -> str:
    transcript_text = "\n".join(transcript) if transcript else "none"
    return (
        "You are an agent. Use JSON only.\n"
        "Choose either a tool call or a final answer.\n\n"
        f"TASK:\n{task}\n\n"
        f"AVAILABLE_TOOLS:\n{tools}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        'Return one of these forms:\n'
        '{"tool":"workspace.search","arguments":{"query":"..."}}\n'
        '{"tool":"final","arguments":{"answer":"..."}}\n'
    )


def _parse_action(raw: str) -> AgentAction | None:
    candidate = raw.strip()
    match = re.search(r"\{.*\}", candidate, re.S)
    if match:
        candidate = match.group(0)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    tool = data.get("tool")
    if not isinstance(tool, str):
        return None
    arguments = data.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return AgentAction(tool=tool, arguments=arguments)


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_ALLOWED_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](visit(node.operand))
        raise ValueError("unsupported expression")

    value = visit(tree)
    return float(value)
