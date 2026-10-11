"""Code generator for the Artificial Cognitive Brain compiler.

Walks the AST produced by :mod:`parser` (optionally after semantic analysis and
optimization) and emits equivalent Python 3 source code.

Mapping summary
---------------

* ``let name = expr``       → ``name = expr``
* ``target = expr``         → ``target = expr`` (incl. compound assignments)
* ``print expr``            → ``print(expr)``
* ``return expr``           → ``return expr``
* ``break`` / ``continue``  → ``break`` / ``continue``
* ``if ... elif ... else``  → ``if ...:`` / ``elif ...:`` / ``else:``
* ``while cond:``            → ``while cond:``
* ``for v in it:``           → ``for v in it:``
* ``def name(...):``         → ``def name(...):``
* ``true`` / ``false``       → ``True`` / ``False``
* ``null``                   → ``None``
* ``and`` / ``or`` / ``not`` → ``and`` / ``or`` / ``not``  (same as Python)

Indentation in the generated code uses 4 spaces, matching PEP 8. The generator
relies on a recursive emitter that tracks the current indentation level.

The generated source is designed to be:
* Self-contained (no imports needed unless the program uses builtins)
* Human-readable (formatted with newlines and indentation)
* Directly executable by ``exec()`` or by saving to a ``.py`` file
"""

from __future__ import annotations

from typing import List

from .ast import (
    AssignmentStatement, BinaryOp, Block, BooleanLiteral, BreakStatement,
    ContinueStatement, PassStatement, Expression, ExpressionStatement, ForStatement,
    FunctionCall, FunctionDef, Identifier, IfStatement, IndexAccess, LetStatement,
    ListLiteral, MemberAccess, NullLiteral, NumberLiteral, PrintStatement, Program,
    ReturnStatement, Statement, StringLiteral, UnaryOp, WhileStatement,
)


# Mapping from ACB operator tokens to Python operator tokens.
# Almost everything is the same; this map exists so future dialects can differ.
_BINARY_OP_MAP = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=",
    "and": "and", "or": "or",
}

_UNARY_OP_MAP = {
    "-": "-",
    "not": "not ",
}

INDENT = "    "  # 4 spaces


class CodeGenError(Exception):
    """Raised when the code generator cannot translate a node."""


class CodeGenerator:
    """Generates Python source code from an AST."""

    def __init__(self) -> None:
        self._lines: List[str] = []
        self._indent_level: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, program: Program) -> str:
        """Generate Python source for ``program`` and return it as a string."""
        self._lines = []
        self._indent_level = 0
        self._emit_program(program)
        # Trailing newline for POSIX-friendliness.
        return "\n".join(self._lines) + "\n"

    # ------------------------------------------------------------------
    # Program & block
    # ------------------------------------------------------------------

    def _emit_program(self, program: Program) -> None:
        # If the program consists solely of function definitions, we still
        # emit them in order. We do NOT auto-insert a ``if __name__ == "__main__":``
        # guard — callers can wrap the result themselves if needed.
        for stmt in program.body:
            self._emit_statement(stmt)

    def _emit_block(self, block: Block) -> None:
        if not block.statements:
            # Empty block — emit ``pass`` to keep Python happy.
            self._emit_line("pass")
            return
        for stmt in block.statements:
            self._emit_statement(stmt)

    # ------------------------------------------------------------------
    # Indentation helpers
    # ------------------------------------------------------------------

    def _emit_line(self, text: str) -> None:
        self._lines.append(self._indent_prefix() + text)

    def _emit_line_raw(self, text: str) -> None:
        # Used for blank lines (no indentation).
        self._lines.append(text)

    def _indent_prefix(self) -> str:
        return INDENT * self._indent_level

    def _indent(self) -> None:
        self._indent_level += 1

    def _dedent(self) -> None:
        if self._indent_level > 0:
            self._indent_level -= 1

    # ------------------------------------------------------------------
    # Statement emitters
    # ------------------------------------------------------------------

    def _emit_statement(self, stmt: Statement) -> None:
        if isinstance(stmt, LetStatement):
            self._emit_let(stmt)
        elif isinstance(stmt, AssignmentStatement):
            self._emit_assignment(stmt)
        elif isinstance(stmt, ExpressionStatement):
            self._emit_line(self._emit_expression(stmt.expr))
        elif isinstance(stmt, PrintStatement):
            self._emit_line(f"print({self._emit_expression(stmt.value)})")
        elif isinstance(stmt, ReturnStatement):
            if stmt.value is not None:
                self._emit_line(f"return {self._emit_expression(stmt.value)}")
            else:
                self._emit_line("return")
        elif isinstance(stmt, BreakStatement):
            self._emit_line("break")
        elif isinstance(stmt, ContinueStatement):
            self._emit_line("continue")
        elif isinstance(stmt, PassStatement):
            self._emit_line("pass")
        elif isinstance(stmt, IfStatement):
            self._emit_if(stmt)
        elif isinstance(stmt, WhileStatement):
            self._emit_while(stmt)
        elif isinstance(stmt, ForStatement):
            self._emit_for(stmt)
        elif isinstance(stmt, FunctionDef):
            self._emit_function_def(stmt)
        elif isinstance(stmt, Block):
            # Inline nested block — emit children directly with current indent.
            self._emit_block(stmt)
        else:
            raise CodeGenError(f"cannot generate code for statement type {type(stmt).__name__}")

    def _emit_let(self, stmt: LetStatement) -> None:
        self._emit_line(f"{stmt.name} = {self._emit_expression(stmt.value)}")

    def _emit_assignment(self, stmt: AssignmentStatement) -> None:
        target = self._emit_expression(stmt.target)
        value = self._emit_expression(stmt.value)
        self._emit_line(f"{target} = {value}")

    def _emit_if(self, stmt: IfStatement) -> None:
        # First branch uses 'if'; subsequent branches use 'elif'.
        first = True
        for cond, body in stmt.branches:
            keyword = "if" if first else "elif"
            first = False
            cond_str = self._emit_expression(cond)
            self._emit_line(f"{keyword} {cond_str}:")
            self._indent()
            self._emit_block(body)
            self._dedent()
        if stmt.else_body is not None:
            self._emit_line("else:")
            self._indent()
            self._emit_block(stmt.else_body)
            self._dedent()

    def _emit_while(self, stmt: WhileStatement) -> None:
        cond_str = self._emit_expression(stmt.condition)
        self._emit_line(f"while {cond_str}:")
        self._indent()
        self._emit_block(stmt.body)
        self._dedent()

    def _emit_for(self, stmt: ForStatement) -> None:
        iter_str = self._emit_expression(stmt.iterable)
        self._emit_line(f"for {stmt.var} in {iter_str}:")
        self._indent()
        self._emit_block(stmt.body)
        self._dedent()

    def _emit_function_def(self, stmt: FunctionDef) -> None:
        params = ", ".join(stmt.params)
        self._emit_line(f"def {stmt.name}({params}):")
        self._indent()
        if stmt.body is not None:
            self._emit_block(stmt.body)
        else:
            self._emit_line("pass")
        self._dedent()
        # Blank line after function definition for readability.
        self._emit_line_raw("")

    # ------------------------------------------------------------------
    # Expression emitters
    # ------------------------------------------------------------------

    def _emit_expression(self, expr: Expression) -> str:
        if isinstance(expr, NumberLiteral):
            # Use repr to get proper float formatting (e.g. ``3.14``).
            return repr(expr.value)
        if isinstance(expr, StringLiteral):
            # Use repr to get a properly escaped Python string literal.
            return repr(expr.value)
        if isinstance(expr, BooleanLiteral):
            return "True" if expr.value else "False"
        if isinstance(expr, NullLiteral):
            return "None"
        if isinstance(expr, Identifier):
            return expr.name
        if isinstance(expr, ListLiteral):
            elements = ", ".join(self._emit_expression(e) for e in expr.elements)
            return f"[{elements}]"
        if isinstance(expr, BinaryOp):
            return self._emit_binary(expr)
        if isinstance(expr, UnaryOp):
            op = _UNARY_OP_MAP.get(expr.op, expr.op)
            return f"{op}{self._emit_expression(expr.operand)}"
        if isinstance(expr, FunctionCall):
            args = ", ".join(self._emit_expression(a) for a in expr.args)
            callee = self._emit_expression(expr.callee)
            return f"{callee}({args})"
        if isinstance(expr, MemberAccess):
            obj = self._emit_expression(expr.obj)
            return f"{obj}.{expr.member}"
        if isinstance(expr, IndexAccess):
            obj = self._emit_expression(expr.obj)
            idx = self._emit_expression(expr.index)
            return f"{obj}[{idx}]"
        raise CodeGenError(
            f"cannot generate code for expression type {type(expr).__name__}"
        )

    def _emit_binary(self, expr: BinaryOp) -> str:
        op = _BINARY_OP_MAP.get(expr.op, expr.op)
        left = self._emit_expression(expr.left)
        right = self._emit_expression(expr.right)
        # Wrap operands in parentheses to preserve associativity and precedence
        # regardless of how the AST was shaped by parsing or optimization.
        return f"({left} {op} {right})"


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def generate(program: Program) -> str:
    """Generate Python source code for ``program``."""
    return CodeGenerator().generate(program)


__all__ = ["CodeGenerator", "CodeGenError", "generate"]
