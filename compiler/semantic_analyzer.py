"""Semantic analyzer for the Artificial Cognitive Brain compiler.

Walks the AST produced by :mod:`parser` and enforces static semantic rules that
the grammar alone cannot express:

* Variables and functions must be declared before use.
* Functions must be defined with a unique name in the same scope.
* ``return`` statements must appear inside a function body.
* ``break`` / ``continue`` must appear inside a loop body.
* Assignment targets must be assignable lvalues.
* Type checking: tracks inferred types (``int``, ``float``, ``bool``, ``str``,
  ``list``, ``function``, ``unknown``) and flags obvious type mismatches such
  as ``"abc" + 1`` or ``true[0]``. The checker is intentionally lenient — many
  operations yield ``unknown`` rather than raising an error, mirroring the
  dynamic nature of the target Python runtime.

The analyzer also annotates the AST in-place with a ``inferred_type`` attribute
on every :class:`~ast.Expression` node, which the optimizer can use to make
smarter decisions (e.g. only fold numeric constants).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from . import ast as ast_mod
from .ast import (
    AssignmentStatement, BinaryOp, Block, BooleanLiteral, BreakStatement,
    ContinueStatement, PassStatement, Expression, ExpressionStatement, ForStatement,
    FunctionCall, FunctionDef, Identifier, IfStatement, IndexAccess, LetStatement,
    ListLiteral, MemberAccess, NullLiteral, NumberLiteral, PrintStatement, Program,
    ReturnStatement, Statement, StringLiteral, UnaryOp, WhileStatement,
)


# ---------------------------------------------------------------------------
# Type representation
# ---------------------------------------------------------------------------

class TypeKind(Enum):
    """Kinds of types the analyzer can infer."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"
    LIST = "list"
    NULL = "null"
    FUNCTION = "function"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_python(cls, value: object) -> "TypeKind":
        if isinstance(value, bool):
            return cls.BOOL
        if isinstance(value, int):
            return cls.INT
        if isinstance(value, float):
            return cls.FLOAT
        if isinstance(value, str):
            return cls.STR
        if value is None:
            return cls.NULL
        if isinstance(value, list):
            return cls.LIST
        return cls.UNKNOWN


@dataclass(frozen=True)
class FunctionType:
    """Type descriptor for a function — params and (optional) return type."""
    params: Tuple[TypeKind, ...]
    return_type: TypeKind = TypeKind.UNKNOWN

    def __str__(self) -> str:
        return f"function({', '.join(str(p) for p in self.params)})"


@dataclass
class Symbol:
    """A single entry in a :class:`SymbolTable`."""
    name: str
    kind: str  # "variable" | "function" | "parameter"
    type: TypeKind = TypeKind.UNKNOWN
    func_type: Optional[FunctionType] = None
    line: int = 0
    col: int = 0


class SymbolTable:
    """A scoped symbol table.

    Each :class:`SymbolTable` has a parent (or ``None`` for the global scope)
    and a dict of locally-defined symbols. Lookups walk the parent chain so
    inner scopes see outer declarations.
    """

    def __init__(self, parent: Optional["SymbolTable"] = None, name: str = "<scope>") -> None:
        self.parent: Optional[SymbolTable] = parent
        self.name: str = name
        self._symbols: Dict[str, Symbol] = {}

    def declare(self, sym: Symbol) -> None:
        """Add ``sym`` to this scope. Raises if a same-named symbol exists here."""
        if sym.name in self._symbols:
            existing = self._symbols[sym.name]
            raise SemanticError(
                f"name {sym.name!r} is already declared in this scope "
                f"(previous declaration at line {existing.line})",
                line=sym.line, col=sym.col,
            )
        self._symbols[sym.name] = sym

    def lookup(self, name: str) -> Optional[Symbol]:
        if name in self._symbols:
            return self._symbols[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def lookup_local(self, name: str) -> Optional[Symbol]:
        return self._symbols.get(name)

    def child(self, name: str) -> "SymbolTable":
        return SymbolTable(parent=self, name=name)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SymbolTable(name={self.name!r}, symbols={list(self._symbols)})"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SemanticError(Exception):
    """Raised when the semantic analyzer detects a violation."""

    def __init__(self, message: str, line: int = 0, col: int = 0,
                 node: Optional[ast_mod.ASTNode] = None) -> None:
        loc = f" at line {line}, col {col}" if line else ""
        super().__init__(f"SemanticError{loc}: {message}")
        self.message = message
        self.line = line
        self.col = col
        self.node = node


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class SemanticAnalyzer:
    """Walks the AST and performs static semantic checks.

    Usage::

        analyzer = SemanticAnalyzer()
        analyzer.analyze(program)        # raises on error
        # AST nodes now carry an ``inferred_type`` attribute
    """

    # Binary operators that require numeric operands.
    ARITHMETIC_OPS: Set[str] = {"+", "-", "*", "/", "%"}
    COMPARISON_OPS: Set[str] = {"==", "!=", "<", ">", "<=", ">="}
    LOGICAL_OPS: Set[str] = {"and", "or"}

    def __init__(self) -> None:
        self.global_scope: SymbolTable = SymbolTable(name="<global>")
        # Current scope — pushed/popped as we enter blocks and functions.
        self._current: SymbolTable = self.global_scope
        # Stack of loop depths so we can validate break/continue.
        self._loop_depth: int = 0
        # Stack of function-return types so we can validate ``return``.
        self._function_stack: List[TypeKind] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, program: Program) -> None:
        """Walk the entire program; raises :class:`SemanticError` on violations."""
        self._visit_program(program)

    # ------------------------------------------------------------------
    # Program & block visitors
    # ------------------------------------------------------------------

    def _visit_program(self, program: Program) -> None:
        # First pass: register all top-level function definitions so they can
        # be referenced before their textual definition (mutual recursion).
        for stmt in program.body:
            if isinstance(stmt, FunctionDef):
                self._declare_function(stmt)
        # Second pass: analyze function bodies and non-function statements.
        for stmt in program.body:
            if isinstance(stmt, FunctionDef):
                # Skip — the function was already declared above; we will visit
                # its body when we encounter it again in the loop below. This
                # means we visit function bodies exactly once.
                self._visit_function_def(stmt)
            else:
                self._visit_statement(stmt)

    def _declare_function(self, fn: FunctionDef) -> None:
        params = tuple(TypeKind.UNKNOWN for _ in fn.params)
        sym = Symbol(
            name=fn.name, kind="function",
            type=TypeKind.FUNCTION,
            func_type=FunctionType(params=params, return_type=TypeKind.UNKNOWN),
            line=fn.line, col=fn.col,
        )
        self.global_scope.declare(sym)

    def _visit_block(self, block: Block) -> None:
        # Each block gets a fresh child scope so local variables do not leak.
        previous = self._current
        self._current = self._current.child("<block>")
        try:
            for stmt in block.statements:
                self._visit_statement(stmt)
        finally:
            self._current = previous

    # ------------------------------------------------------------------
    # Statement visitors
    # ------------------------------------------------------------------

    def _visit_statement(self, stmt: Statement) -> None:
        if isinstance(stmt, LetStatement):
            self._visit_let(stmt)
        elif isinstance(stmt, AssignmentStatement):
            self._visit_assignment(stmt)
        elif isinstance(stmt, ExpressionStatement):
            self._visit_expression(stmt.expr)
        elif isinstance(stmt, IfStatement):
            self._visit_if(stmt)
        elif isinstance(stmt, WhileStatement):
            self._visit_while(stmt)
        elif isinstance(stmt, ForStatement):
            self._visit_for(stmt)
        elif isinstance(stmt, FunctionDef):
            self._visit_function_def(stmt)
        elif isinstance(stmt, ReturnStatement):
            self._visit_return(stmt)
        elif isinstance(stmt, PrintStatement):
            self._visit_expression(stmt.value)
        elif isinstance(stmt, BreakStatement):
            if self._loop_depth == 0:
                raise SemanticError("'break' outside of a loop", stmt.line, stmt.col, stmt)
        elif isinstance(stmt, ContinueStatement):
            if self._loop_depth == 0:
                raise SemanticError("'continue' outside of a loop", stmt.line, stmt.col, stmt)
        elif isinstance(stmt, PassStatement):
            # ``pass`` is always valid — a no-op statement.
            pass
        else:
            # Unknown statement type — ignore gracefully.
            pass

    def _visit_let(self, stmt: LetStatement) -> None:
        value_type = self._visit_expression(stmt.value)
        sym = Symbol(
            name=stmt.name, kind="variable", type=value_type,
            line=stmt.line, col=stmt.col,
        )
        self._current.declare(sym)
        self._annotate(stmt, value_type)

    def _visit_assignment(self, stmt: AssignmentStatement) -> None:
        target = stmt.target
        value_type = self._visit_expression(stmt.value)
        if isinstance(target, Identifier):
            existing = self._current.lookup(target.name)
            if existing is None:
                raise SemanticError(
                    f"assignment to undeclared variable {target.name!r}",
                    stmt.line, stmt.col, stmt,
                )
            # Update the inferred type — we keep it lenient and only widen
            # to UNKNOWN when types conflict.
            if existing.type != value_type and existing.type != TypeKind.UNKNOWN:
                existing.type = TypeKind.UNKNOWN
            else:
                existing.type = value_type
        elif isinstance(target, (MemberAccess, IndexAccess)):
            # Assignment to attribute/index — we only check the receiver expr
            # is well-formed; full type-checking of containers is left to the
            # runtime (Pythonic semantics).
            self._visit_expression(target)
        else:
            raise SemanticError(
                "invalid assignment target",
                stmt.line, stmt.col, stmt,
            )
        self._annotate(stmt, value_type)

    def _visit_if(self, stmt: IfStatement) -> None:
        for cond, body in stmt.branches:
            cond_type = self._visit_expression(cond)
            # We do not raise on non-bool conditions because Python itself
            # treats any value as truthy; instead we annotate and warn later.
            self._annotate(cond, cond_type)
            self._visit_block(body)
        if stmt.else_body is not None:
            self._visit_block(stmt.else_body)

    def _visit_while(self, stmt: WhileStatement) -> None:
        cond_type = self._visit_expression(stmt.condition)
        self._annotate(stmt.condition, cond_type)
        self._loop_depth += 1
        try:
            self._visit_block(stmt.body)
        finally:
            self._loop_depth -= 1

    def _visit_for(self, stmt: ForStatement) -> None:
        iter_type = self._visit_expression(stmt.iterable)
        # Enter a new scope for the loop variable.
        previous = self._current
        self._current = self._current.child("<for>")
        # If the iterable is a list, the loop variable inherits the element
        # type — but we conservatively mark it UNKNOWN otherwise.
        loop_var_type = TypeKind.UNKNOWN
        if iter_type == TypeKind.LIST:
            loop_var_type = TypeKind.UNKNOWN  # element type unknown statically
        sym = Symbol(
            name=stmt.var, kind="variable", type=loop_var_type,
            line=stmt.line, col=stmt.col,
        )
        self._current.declare(sym)
        self._loop_depth += 1
        try:
            self._visit_block(stmt.body)
        finally:
            self._loop_depth -= 1
            self._current = previous

    def _visit_function_def(self, fn: FunctionDef) -> None:
        # Look up the previously-declared symbol to obtain its func_type.
        sym = self.global_scope.lookup_local(fn.name)
        if sym is None:
            # Late-defined function — declare it now. This only happens when
            # _visit_program is called directly without _declare_function.
            self._declare_function(fn)
            sym = self.global_scope.lookup_local(fn.name)

        # Push a fresh scope for the function.
        previous = self._current
        self._current = self.global_scope.child(f"<function {fn.name}>")
        # Parameters become local variables.
        if sym is not None and sym.func_type is not None:
            for p_name in fn.params:
                self._current.declare(Symbol(
                    name=p_name, kind="parameter", type=TypeKind.UNKNOWN,
                    line=fn.line, col=fn.col,
                ))
        # Track that we are inside a function so return statements are valid.
        return_type = sym.func_type.return_type if sym and sym.func_type else TypeKind.UNKNOWN
        self._function_stack.append(return_type)
        try:
            if fn.body is not None:
                self._visit_block(fn.body)
        finally:
            self._function_stack.pop()
            self._current = previous

    def _visit_return(self, stmt: ReturnStatement) -> None:
        if not self._function_stack:
            raise SemanticError("'return' outside of a function", stmt.line, stmt.col, stmt)
        if stmt.value is not None:
            value_type = self._visit_expression(stmt.value)
            self._annotate(stmt.value, value_type)

    # ------------------------------------------------------------------
    # Expression visitors
    # ------------------------------------------------------------------

    def _visit_expression(self, expr: Expression) -> TypeKind:
        """Dispatch on expression type and return the inferred :class:`TypeKind`."""
        if isinstance(expr, NumberLiteral):
            t = TypeKind.FLOAT if expr.is_float else TypeKind.INT
            self._annotate(expr, t)
            return t
        if isinstance(expr, StringLiteral):
            t = TypeKind.STR
            self._annotate(expr, t)
            return t
        if isinstance(expr, BooleanLiteral):
            t = TypeKind.BOOL
            self._annotate(expr, t)
            return t
        if isinstance(expr, NullLiteral):
            t = TypeKind.NULL
            self._annotate(expr, t)
            return t
        if isinstance(expr, Identifier):
            sym = self._current.lookup(expr.name)
            if sym is None:
                raise SemanticError(
                    f"use of undeclared name {expr.name!r}",
                    expr.line, expr.col, expr,
                )
            t = sym.type
            self._annotate(expr, t)
            return t
        if isinstance(expr, ListLiteral):
            for el in expr.elements:
                self._visit_expression(el)
            t = TypeKind.LIST
            self._annotate(expr, t)
            return t
        if isinstance(expr, BinaryOp):
            return self._visit_binary_op(expr)
        if isinstance(expr, UnaryOp):
            return self._visit_unary_op(expr)
        if isinstance(expr, FunctionCall):
            return self._visit_function_call(expr)
        if isinstance(expr, MemberAccess):
            return self._visit_member_access(expr)
        if isinstance(expr, IndexAccess):
            return self._visit_index_access(expr)
        # Fallback for anything we forgot — keep it lenient.
        t = TypeKind.UNKNOWN
        self._annotate(expr, t)
        return t

    def _visit_binary_op(self, expr: BinaryOp) -> TypeKind:
        left_t = self._visit_expression(expr.left)
        right_t = self._visit_expression(expr.right)
        op = expr.op
        if op in self.ARITHMETIC_OPS:
            # int op int -> int; float anywhere -> float; otherwise unknown
            if left_t == TypeKind.INT and right_t == TypeKind.INT and op != "/":
                t = TypeKind.INT
            elif left_t in (TypeKind.INT, TypeKind.FLOAT) and right_t in (TypeKind.INT, TypeKind.FLOAT):
                t = TypeKind.FLOAT
            elif left_t == TypeKind.STR and right_t == TypeKind.STR and op == "+":
                t = TypeKind.STR  # string concatenation
            elif left_t == TypeKind.UNKNOWN or right_t == TypeKind.UNKNOWN:
                t = TypeKind.UNKNOWN
            else:
                raise SemanticError(
                    f"operator {op!r} cannot be applied to types {left_t} and {right_t}",
                    expr.line, expr.col, expr,
                )
        elif op in self.COMPARISON_OPS:
            t = TypeKind.BOOL
        elif op in self.LOGICAL_OPS:
            if left_t not in (TypeKind.BOOL, TypeKind.UNKNOWN) or right_t not in (TypeKind.BOOL, TypeKind.UNKNOWN):
                raise SemanticError(
                    f"operator {op!r} requires boolean operands, got {left_t} and {right_t}",
                    expr.line, expr.col, expr,
                )
            t = TypeKind.BOOL
        else:
            t = TypeKind.UNKNOWN
        self._annotate(expr, t)
        return t

    def _visit_unary_op(self, expr: UnaryOp) -> TypeKind:
        operand_t = self._visit_expression(expr.operand)
        if expr.op == "-":
            if operand_t not in (TypeKind.INT, TypeKind.FLOAT, TypeKind.UNKNOWN):
                raise SemanticError(
                    f"unary '-' requires numeric operand, got {operand_t}",
                    expr.line, expr.col, expr,
                )
            t = operand_t
        elif expr.op == "not":
            if operand_t not in (TypeKind.BOOL, TypeKind.UNKNOWN):
                raise SemanticError(
                    f"'not' requires boolean operand, got {operand_t}",
                    expr.line, expr.col, expr,
                )
            t = TypeKind.BOOL
        else:
            t = TypeKind.UNKNOWN
        self._annotate(expr, t)
        return t

    def _visit_function_call(self, expr: FunctionCall) -> TypeKind:
        # Visit callee to ensure identifiers exist (e.g. module.attr.func())
        callee_t = self._visit_expression(expr.callee)
        # Visit arguments
        for arg in expr.args:
            self._visit_expression(arg)
        # If callee is an identifier pointing at a known function symbol, we
        # can be more precise about the return type.
        ret = TypeKind.UNKNOWN
        if isinstance(expr.callee, Identifier):
            sym = self._current.lookup(expr.callee.name)
            if sym is not None and sym.func_type is not None:
                # Arity check
                if len(expr.args) != len(sym.func_type.params):
                    raise SemanticError(
                        f"function {expr.callee.name!r} expects "
                        f"{len(sym.func_type.params)} argument(s), got {len(expr.args)}",
                        expr.line, expr.col, expr,
                    )
                ret = sym.func_type.return_type
            elif sym is not None and sym.kind != "function":
                raise SemanticError(
                    f"name {expr.callee.name!r} is not callable",
                    expr.line, expr.col, expr,
                )
            elif sym is None:
                # Unknown identifier — left to runtime
                pass
        self._annotate(expr, ret)
        return ret

    def _visit_member_access(self, expr: MemberAccess) -> TypeKind:
        obj_t = self._visit_expression(expr.obj)
        # Without a type system for objects, member access yields UNKNOWN.
        t = TypeKind.UNKNOWN
        self._annotate(expr, t)
        return t

    def _visit_index_access(self, expr: IndexAccess) -> TypeKind:
        obj_t = self._visit_expression(expr.obj)
        idx_t = self._visit_expression(expr.index)
        if obj_t == TypeKind.UNKNOWN:
            t = TypeKind.UNKNOWN
        elif obj_t == TypeKind.LIST:
            t = TypeKind.UNKNOWN  # element type not tracked
        elif obj_t == TypeKind.STR:
            t = TypeKind.STR
        else:
            raise SemanticError(
                f"indexing not supported on type {obj_t}",
                expr.line, expr.col, expr,
            )
        self._annotate(expr, t)
        return t

    # ------------------------------------------------------------------
    # Annotation helper
    # ------------------------------------------------------------------

    def _annotate(self, node: ast_mod.ASTNode, type_: TypeKind) -> None:
        """Attach an ``inferred_type`` attribute to a node. Safe to call repeatedly."""
        # ``__slots__`` on ASTNode only includes ``line, col``, so we need to
        # use ``__dict__`` instead. Since dataclass instances have a __dict__
        # by default (no slots were declared on the dataclasses), this works.
        try:
            setattr(node, "inferred_type", type_)
        except AttributeError:
            # Some non-dataclass nodes (Program, etc.) may not allow this.
            pass


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def analyze(program: Program) -> SymbolTable:
    """Run the semantic analyzer on ``program`` and return the global scope."""
    analyzer = SemanticAnalyzer()
    analyzer.analyze(program)
    return analyzer.global_scope


__all__ = [
    "SemanticAnalyzer", "SemanticError", "SymbolTable", "Symbol",
    "TypeKind", "FunctionType", "analyze",
]
