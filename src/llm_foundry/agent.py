from __future__ import annotations

from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import ast
import json
import math
import os
import re
import subprocess
from typing import Callable, Protocol
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from .adapters import TextBackend


class ToolFn(Protocol):
    def __call__(self, **kwargs) -> str: ...


@dataclass(frozen=True)
class ToolPolicy:
    allow_workspace_write: bool = True
    allow_web_fetch: bool = False
    allow_web_search: bool = False
    allow_github_api: bool = False
    allow_shell: bool = False
    web_max_chars: int = 12000
    shell_max_chars: int = 12000


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


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


class ToolRegistry:
    def __init__(self, workspace_root: str | Path = ".", policy: ToolPolicy | None = None, tools: dict[str, ToolFn] | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.policy = policy or ToolPolicy()
        self.tools: dict[str, ToolFn] = {}
        self.tool_descriptions: dict[str, str] = {}
        self._register_default_tools()
        if tools:
            for name, fn in tools.items():
                self.register_tool(name, fn, "custom tool")

    def register_tool(self, name: str, fn: ToolFn, description: str) -> None:
        self.tools[name] = fn
        self.tool_descriptions[name] = description

    def _register_default_tools(self) -> None:
        self.register_tool("workspace.list", self._workspace_list, "list files under the workspace root")
        self.register_tool("workspace.read", self._workspace_read, "read a text file inside the workspace")
        self.register_tool("workspace.search", self._workspace_search, "search text inside workspace files")
        self.register_tool("workspace.write", self._workspace_write, "write a text file inside the workspace")
        self.register_tool("workspace.append", self._workspace_append, "append text to a workspace file")
        self.register_tool("math.calc", self._math_calc, "evaluate safe arithmetic expressions")

        if self.policy.allow_web_fetch:
            self.register_tool("web.fetch", self._web_fetch, "fetch a web page and return readable text")
        if self.policy.allow_web_search:
            self.register_tool("web.search", self._web_search, "search the web and return top results")
        if self.policy.allow_github_api:
            self.register_tool("github.api", self._github_api, "call GitHub API through gh cli")
        if self.policy.allow_shell:
            self.register_tool("shell.run", self._shell_run, "run a shell command in the workspace root")

    def describe(self) -> str:
        lines = []
        for name in sorted(self.tools):
            description = self.tool_descriptions.get(name, "")
            lines.append(f"- {name}: {description}".rstrip())
        return "\n".join(lines)

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

    def _workspace_write(self, path: str, content: str) -> str:
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return str(target.relative_to(self.workspace_root))

    def _workspace_append(self, path: str, content: str) -> str:
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return str(target.relative_to(self.workspace_root))

    def _math_calc(self, expression: str) -> str:
        return str(_safe_eval(expression))

    def _web_fetch(self, url: str, max_chars: int | None = None) -> str:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 LLM-Foundry"})
        with urlopen(request, timeout=30) as resp:
            content_type = resp.headers.get_content_type()
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
        text = raw.decode(charset, errors="replace")
        if "html" in content_type:
            text = _strip_html(text)
        return _shorten(text, max_chars or self.policy.web_max_chars)

    def _web_search(self, query: str, max_results: int = 5) -> str:
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        request = Request(search_url, headers={"User-Agent": "Mozilla/5.0 LLM-Foundry"})
        with urlopen(request, timeout=30) as resp:
            html_text = resp.read().decode("utf-8", errors="replace")
        results: list[str] = []
        pattern = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
        for url, title_html in pattern.findall(html_text):
            title = _strip_html(unescape(title_html))
            results.append(f"{title} | {unescape(url)}")
            if len(results) >= max_results:
                break
        return "\n".join(results) if results else "no results"

    def _github_api(self, endpoint: str, method: str = "GET", body: str = "") -> str:
        cmd = ["gh", "api", endpoint, "--method", method]
        input_bytes = None
        if body:
            cmd.extend(["--input", "-"])
            input_bytes = body.encode("utf-8")
        result = subprocess.run(cmd, input=input_bytes, cwd=self.workspace_root, capture_output=True, text=True)
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        return _shorten(output.strip() or f"gh api exited {result.returncode}", self.policy.web_max_chars)

    def _shell_run(self, command: str, timeout_seconds: int = 30) -> str:
        result = subprocess.run(
            command,
            cwd=self.workspace_root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = [f"exit={result.returncode}"]
        if result.stdout:
            output.append(f"stdout:\n{result.stdout.strip()}")
        if result.stderr:
            output.append(f"stderr:\n{result.stderr.strip()}")
        return _shorten("\n".join(output), self.policy.shell_max_chars)


@dataclass
class AgentRuntime:
    backend: TextBackend
    tools: ToolRegistry
    max_steps: int = 6
    repair_attempts: int = 1

    def run(self, task: str) -> AgentTrace:
        transcript: list[str] = []
        trace = AgentTrace(task=task)
        for _ in range(self.max_steps):
            prompt = _agent_prompt(task, transcript, self.tools.describe())
            raw = self.backend.generate(prompt)
            action = _parse_action(raw)
            if action is None or action.tool not in self.tools.tools:
                action = self._repair_action(task, transcript, raw) or action
            if action is None:
                trace.steps.append(AgentStep(raw=raw, action="final", final=raw))
                trace.final = raw
                return trace
            if action.tool == "final":
                final = str(action.arguments.get("answer", ""))
                trace.steps.append(AgentStep(raw=raw, action="final", final=final))
                trace.final = final
                return trace
            try:
                observation = self.tools.run(action)
            except Exception as exc:
                observation = f"TOOL_ERROR: {exc}"
            transcript.append(f"TOOL {action.tool} {json.dumps(action.arguments, ensure_ascii=False)}")
            transcript.append(f"OBSERVATION {observation}")
            trace.steps.append(AgentStep(raw=raw, action=action.tool, observation=observation))
        trace.final = trace.steps[-1].observation if trace.steps else ""
        return trace

    def _repair_action(self, task: str, transcript: list[str], raw: str) -> AgentAction | None:
        current = raw
        for _ in range(self.repair_attempts):
            prompt = _repair_prompt(task, transcript, self.tools.describe(), current)
            current = self.backend.generate(prompt)
            repaired = _parse_action(current)
            if repaired is not None and repaired.tool in self.tools.tools:
                return repaired
        return None


def _agent_prompt(task: str, transcript: list[str], tools: str) -> str:
    transcript_text = "\n".join(transcript) if transcript else "none"
    return (
        "You are an autonomous agent. Use JSON only.\n"
        "Choose either a tool call or a final answer.\n"
        "Prefer the cheapest tool that can solve the task.\n"
        "Use memory, workspace, math, web, GitHub, and shell tools only when they help.\n\n"
        f"TASK:\n{task}\n\n"
        f"AVAILABLE_TOOLS:\n{tools}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        'Return one of these forms:\n'
        '{"tool":"workspace.search","arguments":{"query":"..."}}\n'
        '{"tool":"final","arguments":{"answer":"..."}}\n'
    )


def _repair_prompt(task: str, transcript: list[str], tools: str, raw: str) -> str:
    transcript_text = "\n".join(transcript) if transcript else "none"
    return (
        "Rewrite the previous response into a single valid JSON object.\n"
        "It must use one registered tool name or final.\n\n"
        f"TASK:\n{task}\n\n"
        f"AVAILABLE_TOOLS:\n{tools}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        f"RAW_RESPONSE:\n{raw}\n\n"
        'Return only one JSON object like:\n'
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


def _strip_html(text: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(text)
    return re.sub(r"\s+", " ", parser.text()).strip()


def _shorten(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 1)].rstrip() + "…"
