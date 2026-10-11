"""AST node definitions for the Artificial Cognitive Brain compiler.

This module defines the Abstract Syntax Tree (AST) node types used throughout
the compiler pipeline. Every node carries source position information
(line, column) so that downstream stages (parser, semantic analyzer,
optimizer, code generator) can produce accurate error messages.

The AST is divided into two top-level hierarchies:

* :class:`Expression` — nodes that produce a value when evaluated.
* :class:`Statement`  — nodes that perform an action and do not yield a value.

A :class:`Program` is the root node and contains a list of top-level statements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class ASTNode:
    """Base class for every AST node.

    Every concrete node stores the ``(line, col)`` position of the first token
    that produced it. This information is propagated through the pipeline so
    that error reports always point back to the original source location.
    """

    __slots__ = ("line", "col")

    def __init__(self, line: int = 0, col: int = 0) -> None:
        self.line: int = line
        self.col: int = col

    def children(self) -> List["ASTNode"]:
        """Return direct child AST nodes (used by traversals / visitors)."""
        return []

    def __repr__(self) -> str:  # pragma: no cover — debugging helper
        attrs = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items()
            if k not in ("line", "col")
        )
        return f"{type(self).__name__}({attrs})"


class Expression(ASTNode):
    """Marker base class for nodes that evaluate to a value."""


class Statement(ASTNode):
    """Marker base class for nodes that perform an action."""


# ---------------------------------------------------------------------------
# Literal expressions
# ---------------------------------------------------------------------------

@dataclass
class NumberLiteral(Expression):
    """A numeric literal — covers both integers and floats.

    The ``is_float`` flag is preserved so the code generator can emit
    Python ``int`` or ``float`` literals correctly.
    """
    value: Any  # int | float
    is_float: bool = False
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return []


@dataclass
class StringLiteral(Expression):
    """A double-quoted string literal. Escapes are already resolved by the lexer."""
    value: str
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return []


@dataclass
class BooleanLiteral(Expression):
    """A ``true`` / ``false`` literal."""
    value: bool
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return []


@dataclass
class NullLiteral(Expression):
    """A ``null`` literal — emitted as ``None`` in the generated Python code."""
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return []


# ---------------------------------------------------------------------------
# Identifier & composite expressions
# ---------------------------------------------------------------------------

@dataclass
class Identifier(Expression):
    """A bare identifier reference such as ``x`` or ``my_func``."""
    name: str
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return []


@dataclass
class ListLiteral(Expression):
    """A list literal ``[expr, expr, ...]``."""
    elements: List[Expression] = field(default_factory=list)
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return list(self.elements)


@dataclass
class BinaryOp(Expression):
    """A binary operator expression ``left OP right``.

    Supported operators:
      * arithmetic: ``+ - * / %``
      * comparison: ``== != < > <= >=``
      * logical:    ``and or``
    """
    op: str
    left: Expression
    right: Expression
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.left, self.right]


@dataclass
class UnaryOp(Expression):
    """A unary operator expression ``OP operand``.

    Supported operators: ``-`` (negation), ``not`` (logical NOT),
    ``+`` (no-op promoted by parser).
    """
    op: str
    operand: Expression
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.operand]


@dataclass
class FunctionCall(Expression):
    """A call expression ``callee(arg1, arg2, ...)``.

    ``callee`` is typically an :class:`Identifier`, but may also be a
    :class:`MemberAccess` to support calling methods like ``list.append(1)``.
    """
    callee: Expression
    args: List[Expression] = field(default_factory=list)
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.callee, *self.args]


@dataclass
class MemberAccess(Expression):
    """A member access ``obj.member`` (e.g. ``obj.x`` or ``module.func``)."""
    obj: Expression
    member: str
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.obj]


@dataclass
class IndexAccess(Expression):
    """An index access ``obj[index]`` (e.g. ``my_list[0]``)."""
    obj: Expression
    index: Expression
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.obj, self.index]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class Program(ASTNode):
    """Root AST node. Holds a list of top-level statements."""
    body: List[Statement] = field(default_factory=list)
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return list(self.body)


@dataclass
class Block(Statement):
    """A brace-or-indent delimited block of statements (function body, loop body, etc.)."""
    statements: List[Statement] = field(default_factory=list)
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return list(self.statements)


@dataclass
class LetStatement(Statement):
    """A variable binding ``let name = expr``."""
    name: str
    value: Expression
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.value]


@dataclass
class AssignmentStatement(Statement):
    """An assignment ``target = expr``.

    ``target`` may be an :class:`Identifier`, :class:`MemberAccess`, or
    :class:`IndexAccess` — the semantic analyzer verifies the target is
    assignable.
    """
    target: Expression
    value: Expression
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.target, self.value]


@dataclass
class ExpressionStatement(Statement):
    """A bare expression used as a statement (e.g. a function call on its own line)."""
    expr: Expression
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.expr]


@dataclass
class IfStatement(Statement):
    """An ``if`` / ``elif`` / ``else`` chain.

    ``branches`` is a list of ``(condition, body)`` tuples covering the
    ``if`` clause first and any ``elif`` clauses afterwards, in source order.
    ``else_body`` is ``None`` when no ``else`` is present.
    """
    branches: List[Tuple[Expression, Block]] = field(default_factory=list)
    else_body: Optional[Block] = None
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        out: List[ASTNode] = []
        for cond, body in self.branches:
            out.append(cond)
            out.append(body)
        if self.else_body is not None:
            out.append(self.else_body)
        return out


@dataclass
class WhileStatement(Statement):
    """A ``while condition: body`` loop."""
    condition: Expression
    body: Block
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.condition, self.body]


@dataclass
class ForStatement(Statement):
    """A ``for var in iterable: body`` loop."""
    var: str
    iterable: Expression
    body: Block
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.iterable, self.body]


@dataclass
class FunctionDef(Statement):
    """A function definition ``def name(param1, param2, ...): body``."""
    name: str
    params: List[str] = field(default_factory=list)
    body: Optional[Block] = None
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.body] if self.body is not None else []


@dataclass
class ReturnStatement(Statement):
    """A ``return expr`` statement. ``value`` is ``None`` for bare ``return``."""
    value: Optional[Expression] = None
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.value] if self.value is not None else []


@dataclass
class PrintStatement(Statement):
    """A ``print expr`` statement — emits as a Python ``print(...)`` call."""
    value: Expression
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return [self.value]


@dataclass
class BreakStatement(Statement):
    """A ``break`` statement (only valid inside a loop body)."""
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return []


@dataclass
class ContinueStatement(Statement):
    """A ``continue`` statement (only valid inside a loop body)."""
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return []


@dataclass
class PassStatement(Statement):
    """A ``pass`` statement — a no-op, useful for empty function bodies.

    The code generator emits Python's ``pass`` keyword verbatim.
    """
    line: int = 0
    col: int = 0

    def __post_init__(self) -> None:
        ASTNode.__init__(self, self.line, self.col)

    def children(self) -> List[ASTNode]:
        return []


# ---------------------------------------------------------------------------
# Convenience: full node-type registry (used by visitors & external tooling)
# ---------------------------------------------------------------------------

ALL_EXPRESSION_TYPES: Tuple[type, ...] = (
    NumberLiteral, StringLiteral, BooleanLiteral, NullLiteral,
    Identifier, ListLiteral, BinaryOp, UnaryOp,
    FunctionCall, MemberAccess, IndexAccess,
)

ALL_STATEMENT_TYPES: Tuple[type, ...] = (
    Program, Block, LetStatement, AssignmentStatement,
    ExpressionStatement, IfStatement, WhileStatement, ForStatement,
    FunctionDef, ReturnStatement, PrintStatement,
    BreakStatement, ContinueStatement, PassStatement,
)

ALL_NODE_TYPES: Tuple[type, ...] = (ASTNode,) + ALL_EXPRESSION_TYPES + ALL_STATEMENT_TYPES


__all__ = [
    # Base classes
    "ASTNode", "Expression", "Statement",
    # Literals
    "NumberLiteral", "StringLiteral", "BooleanLiteral", "NullLiteral",
    # Composite expressions
    "Identifier", "ListLiteral", "BinaryOp", "UnaryOp",
    "FunctionCall", "MemberAccess", "IndexAccess",
    # Statements
    "Program", "Block", "LetStatement", "AssignmentStatement",
    "ExpressionStatement", "IfStatement", "WhileStatement", "ForStatement",
    "FunctionDef", "ReturnStatement", "PrintStatement",
    "BreakStatement", "ContinueStatement", "PassStatement",
    # Registries
    "ALL_EXPRESSION_TYPES", "ALL_STATEMENT_TYPES", "ALL_NODE_TYPES",
]
