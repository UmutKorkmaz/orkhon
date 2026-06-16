"""Calculator tool — safe arithmetic via an AST allowlist (no names/calls/imports)."""

from __future__ import annotations

import ast
import operator

from orkhon.serve.tools.base import ToolResult

# Only these binary/unary operators are allowed; everything else (calls, names,
# attributes, imports) raises, so the calculator cannot execute arbitrary code.
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


_MAX_NODES = 200          # cap AST size (DoS guard against pathological expressions)
_MAX_BITS = 4096          # cap intermediate integer size (DoS guard against 2**10**6)
_MAX_POW_EXP = 1000       # cap exponent magnitude


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression; raise on anything non-arithmetical."""
    tree = ast.parse(expression, mode="eval")
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > _MAX_NODES:
        raise ValueError("expression too complex")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):  # numbers (bool is rejected — it's an int subclass)
            if type(node.value) is bool:
                raise ValueError("booleans not allowed")
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("non-numeric constant")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            r = _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(r, int) and r.bit_length() > _MAX_BITS:
                raise ValueError("result too large")
            if type(node.op) is ast.Pow:
                exp = ev(node.right)
                if isinstance(exp, (int, float)) and abs(exp) > _MAX_POW_EXP:
                    raise ValueError("exponent too large")
            return r
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"disallowed node: {type(node).__name__}")

    return ev(tree)


class Calculator:
    name = "calculator"
    description = "Evaluate an arithmetic expression (e.g. '2*(3+4)'). No functions or variables."
    parameters = {"expression": {"type": "string"}}

    def __call__(self, *, expression: str, **_) -> ToolResult:
        try:
            return ToolResult(output=str(safe_eval(expression)))
        except Exception as e:
            return ToolResult(output=f"calculator error: {e}", error=True)
