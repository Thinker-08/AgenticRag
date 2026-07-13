from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
import uuid

from ..config import SandboxConfig
from ..interfaces.types import ToolResult

_ALLOWED_CALLS = {"abs", "min", "max", "round", "sum", "len", "float", "int", "pow"}
_ALLOWED_NODES = (ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq, ast.Call, ast.Tuple, ast.List, ast.Subscript, ast.Index, ast.Slice)


def validateCode(code: str) -> None:
    tree = ast.parse(code, mode="exec")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            raise ValueError("attribute access is not allowed in the sandbox")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
                raise ValueError(f"disallowed call in sandbox: {ast.dump(node.func)}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are not allowed")
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax in sandbox: {type(node).__name__}")


_RUNNER = r"""
import json, sys, resource, builtins
try:
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
except Exception:
    pass
payload = json.loads(sys.stdin.read())
code, inputs = payload["code"], payload["inputs"]
safe = {{"abs": abs, "min": min, "max": max, "round": round, "sum": sum, "len": len,
         "float": float, "int": int, "pow": pow}}
ns = dict(inputs)
try:
    exec(compile(code, "<sandbox>", "exec"), {{"__builtins__": safe}}, ns)
    print(json.dumps({{"ok": True, "result": ns.get("result")}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""


class SubprocessSandbox:
    def __init__(self, timeout_s: float = 2.0, max_mem_mb: int = 256) -> None:
        self.timeout_s = timeout_s
        self.max_mem_mb = max_mem_mb

    def run(self, code: str, inputs: dict, *, timeout_s: float | None = None) -> ToolResult:
        run_id = "sbx_" + uuid.uuid4().hex[:8]
        t0 = time.monotonic()

        try:
            validateCode(code)
        except (ValueError, SyntaxError) as exc:
            return ToolResult(ok=False, error=f"validation: {exc}", run_id=run_id)

        runner = _RUNNER.format(cpu=int(timeout_s or self.timeout_s) + 1, mem=self.max_mem_mb * 1024 * 1024)
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", runner], input=json.dumps({"code": code, "inputs": inputs}), capture_output=True, text=True, timeout=timeout_s or self.timeout_s, env={"PATH": "/usr/bin"})
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error="timeout", run_id=run_id, wall_ms=(time.monotonic() - t0) * 1000)

        wall = (time.monotonic() - t0) * 1000
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return ToolResult(ok=False, error=proc.stderr.strip() or "no output", run_id=run_id, wall_ms=wall)
        if not out.get("ok"):
            return ToolResult(ok=False, error=out.get("error", "error"), run_id=run_id, stderr=proc.stderr, wall_ms=wall)

        return ToolResult(ok=True, result=out.get("result"), run_id=run_id, stdout=proc.stdout, wall_ms=wall)


def buildSandbox(cfg: SandboxConfig) -> SubprocessSandbox:
    return SubprocessSandbox(timeout_s=cfg.timeout_s, max_mem_mb=cfg.max_mem_mb)
