"""Safe string-expression evaluator shared by `Dataset.calc` and the `pyramids calc` CLI.

A deliberately small AST interpreter for band-algebra expressions over named raster
arrays (``A``, ``B``, …). It never uses ``eval``/``exec``: a hostile expression
(``__import__('os')``, attribute access, comprehensions, …) is rejected with a
``ValueError`` rather than executed. Only arithmetic, comparisons, and a small
whitelist of ``np.<func>`` calls over the bound arrays are permitted.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

import numpy as np

_CALC_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_CALC_UNARYOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_CALC_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
# The only function calls a `calc` expression may use, all from numpy.
_CALC_NP_FUNCS = frozenset(
    {
        "where",
        "clip",
        "log",
        "log10",
        "exp",
        "sqrt",
        "abs",
        "minimum",
        "maximum",
        "power",
    }
)


def _safe_calc_eval(node: ast.AST, variables: dict) -> object:
    """Evaluate one node of a `calc` expression AST against a whitelist.

    This is a deliberately small interpreter — it never uses ``eval``/``exec``,
    so a hostile expression (``__import__('os')``, attribute access, comprehensions,
    ...) is rejected with a ``ValueError`` rather than executed.

    Args:
        node: The AST node to evaluate.
        variables: Band arrays bound to their names (``A``, ``B``, ...).

    Returns:
        The numeric / ndarray value of the node.

    Raises:
        ValueError: The node is not in the allowed grammar.
    """
    if isinstance(node, ast.Expression):
        result = _safe_calc_eval(node.body, variables)
    elif isinstance(node, ast.BinOp) and type(node.op) in _CALC_BINOPS:
        result = _CALC_BINOPS[type(node.op)](
            _safe_calc_eval(node.left, variables),
            _safe_calc_eval(node.right, variables),
        )
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_UNARYOPS:
        result = _CALC_UNARYOPS[type(node.op)](_safe_calc_eval(node.operand, variables))
    elif (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and type(node.ops[0]) in _CALC_COMPARE
    ):
        result = _CALC_COMPARE[type(node.ops[0])](
            _safe_calc_eval(node.left, variables),
            _safe_calc_eval(node.comparators[0], variables),
        )
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        result = node.value
    elif isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"unknown name in calc expression: {node.id!r}")
        result = variables[node.id]
    elif (
        isinstance(node, ast.Call)
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr in _CALC_NP_FUNCS
    ):
        result = getattr(np, node.func.attr)(
            *[_safe_calc_eval(arg, variables) for arg in node.args]
        )
    else:
        raise ValueError(
            "disallowed element in calc expression; only arithmetic over the "
            "input bands (A, B, ...), comparisons, and a small set of np.<func> "
            "calls are permitted."
        )
    return result


def eval_calc(expr: str, variables: dict) -> np.ndarray:
    """Evaluate a `calc` string expression over the bound arrays.

    Parses ``expr`` once and runs it through :func:`_safe_calc_eval` (never
    ``eval``/``exec``), returning the result as a NumPy array.

    Args:
        expr: The band-algebra expression, e.g. ``"(B - A) / (B + A)"``.
        variables: Arrays bound to their single-letter names (``A``, ``B``, ...).

    Returns:
        numpy.ndarray: The evaluated expression as an array.

    Raises:
        ValueError: The expression uses a name not in ``variables`` or an element
            outside the allowed grammar.
        SyntaxError: ``expr`` is not a valid Python expression.
    """
    return np.asarray(_safe_calc_eval(ast.parse(expr, mode="eval"), variables))
