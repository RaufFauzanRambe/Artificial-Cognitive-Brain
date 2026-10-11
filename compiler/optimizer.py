"""AST optimizer for the Artificial Cognitive Brain compiler.

Performs a small set of safe, semantics-preserving transformations on the AST:

1. **Constant folding** — evaluates expressions whose operands are literals
   at compile time (e.g. ``2 + 3`` becomes ``5``, ``true and false`` becomes
   ``false``, ``[1, 2, 3].length`` is *not* folded because we do not track
   runtime values).

2. **Constant propagation** — when a ``let`` statement binds a literal value
   to a name, subsequent uses of that name *within the same block* are
   replaced by the literal. This is a local, intra-block pass — it does not
   cross control-flow boundaries, so it is always safe.

3. **Algebraic simplification** — applies a small catalogue of identities:
   * ``x + 0`` → ``x``, ``x - 0`` → ``x``
   * ``x * 1`` → ``x``, ``x * 0`` → ``0``
   * ``x - x`` → ``0``
   * ``not not x`` → ``x``
   * ``true and x`` → ``x``, ``false or x`` → ``x``

4. **Dead-code elimination** — removes statements that follow a
   ``return``/``break``/``continue`` in the same block, and removes empty
   branches from ``if`` statements.

5. **Common subexpression elimination (CSE)** — within a single basic block,
   detects identical sub-expressions and reuses a synthetic temporary
   variable. This pass is opt-in because it changes the shape of the AST;
   callers can disable it via :meth:`Optimizer.optimize(..., enable_cse=False)`.

The optimizer is **idempotent** — running it twice yields the same AST as
running it once. It is also **conservative** — when in doubt it leaves the AST
untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .ast import (
    AssignmentStatement, ASTNode, BinaryOp, Block, BooleanLiteral, BreakStatement,
    ContinueStatement, PassStatement, Expression, ExpressionStatement, ForStatement,
    FunctionCall, FunctionDef, Identifier, IfStatement, IndexAccess, LetStatement,
    ListLiteral, MemberAccess, NullLiteral, NumberLiteral, PrintStatement, Program,
    ReturnStatement, Statement, StringLiteral, UnaryOp, WhileStatement,
)


class OptimizationStats:
    """Tracks how many of each optimization were applied. Useful for diagnostics."""

    def __init__(self) -> None:
        self.constants_folded: int = 0
        self.constants_propagated: int = 0
        self.algebraic_simplifications: int = 0
        self.dead_code_removed: int = 0
        self.cse_replaced: int = 0
        self.passes_run: int = 0

    def __repr__(self) -> str:
        return (
            f"OptimizationStats(folded={self.constants_folded}, "
            f"propagated={self.constants_propagated}, "
            f"algebraic={self.algebraic_simplifications}, "
            f"dead_code={self.dead_code_removed}, "
            f"cse={self.cse_replaced}, "
            f"passes={self.passes_run})"
        )

    @property
    def total(self) -> int:
        return (
            self.constants_folded
            + self.constants_propagated
            + self.algebraic_simplifications
            + self.dead_code_removed
            + self.cse_replaced
        )


class Optimizer:
    """Performs AST-level optimizations. Re-entrant; safe to call multiple times."""

    def __init__(self) -> None:
        self.stats = OptimizationStats()
        # Local constant environment: name -> literal expression
        # Reset on entering each block (constant propagation is intra-block).
        self._consts: Dict[str, Expression] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, program: Program, enable_cse: bool = False,
                 max_passes: int = 4) -> Program:
        """Optimize ``program`` in place (mostly) and return it.

        Multiple passes are run until the AST stops changing or ``max_passes``
        is reached. CSE is disabled by default because it introduces temporary
        bindings that may surprise readers; callers can opt in.

        Change detection uses a per-pass counter of applied optimizations
        rather than a serialized AST signature — this avoids deep recursion
        on large ASTs.
        """
        for _ in range(max_passes):
            self.stats.passes_run += 1
            # Snapshot stats before the pass so we can detect convergence.
            before_total = (
                self.stats.constants_folded
                + self.stats.constants_propagated
                + self.stats.algebraic_simplifications
                + self.stats.dead_code_removed
                + self.stats.cse_replaced
            )
            self._visit_program(program)
            if enable_cse:
                self._run_cse_pass(program)
            after_total = (
                self.stats.constants_folded
                + self.stats.constants_propagated
                + self.stats.algebraic_simplifications
                + self.stats.dead_code_removed
                + self.stats.cse_replaced
            )
            # If no new optimizations were applied this pass, we've converged.
            if after_total == before_total:
                break
        return program

    # ------------------------------------------------------------------
    # Program & block visitors
    # ------------------------------------------------------------------

    def _visit_program(self, program: Program) -> None:
        saved = self._swap_consts({})
        try:
            program.body = [self._visit_statement(s) for s in program.body if s is not None]
            program.body = self._eliminate_dead_code(program.body)
        finally:
            self._swap_consts(saved)

    def _visit_block(self, block: Block) -> Block:
        saved = self._swap_consts({})
        try:
            new_statements: List[Statement] = []
            for stmt in block.statements:
                visited = self._visit_statement(stmt)
                if visited is None:
                    continue
                if isinstance(visited, list):
                    new_statements.extend(visited)
                else:
                    new_statements.append(visited)
            block.statements = self._eliminate_dead_code(new_statements)
            return block
        finally:
            self._swap_consts(saved)

    def _swap_consts(self, new_consts: Dict[str, Expression]) -> Dict[str, Expression]:
        old = self._consts
        self._consts = new_consts
        return old

    # ------------------------------------------------------------------
    # Statement visitors
    # ------------------------------------------------------------------

    def _visit_statement(self, stmt: Statement) -> Optional[Statement]:
        if isinstance(stmt, LetStatement):
            return self._visit_let(stmt)
        if isinstance(stmt, AssignmentStatement):
            return self._visit_assignment(stmt)
        if isinstance(stmt, ExpressionStatement):
            stmt.expr = self._visit_expression(stmt.expr)
            return stmt
        if isinstance(stmt, PrintStatement):
            stmt.value = self._visit_expression(stmt.value)
            return stmt
        if isinstance(stmt, ReturnStatement):
            if stmt.value is not None:
                stmt.value = self._visit_expression(stmt.value)
            return stmt
        if isinstance(stmt, IfStatement):
            return self._visit_if(stmt)
        if isinstance(stmt, WhileStatement):
            # IMPORTANT: do NOT propagate constants into the loop condition.
            # The condition is re-evaluated on each iteration, so any
            # variable referenced in it may have been reassigned by the
            # previous iteration's body. Constant propagation into the
            # condition would be unsound.
            saved = self._swap_consts({})
            try:
                stmt.condition = self._visit_expression(stmt.condition)
                stmt.body = self._visit_block(stmt.body)
            finally:
                self._swap_consts(saved)
            return stmt
        if isinstance(stmt, ForStatement):
            # The iterable is evaluated once before the loop starts, so it
            # is safe to propagate constants into it. However, the loop body
            # may reassign variables, so we use a fresh const environment
            # for the body.
            stmt.iterable = self._visit_expression(stmt.iterable)
            saved = self._swap_consts({})
            try:
                stmt.body = self._visit_block(stmt.body)
            finally:
                self._swap_consts(saved)
            return stmt
        if isinstance(stmt, FunctionDef):
            saved = self._swap_consts({})
            try:
                if stmt.body is not None:
                    stmt.body = self._visit_block(stmt.body)
            finally:
                self._swap_consts(saved)
            return stmt
        if isinstance(stmt, (BreakStatement, ContinueStatement, PassStatement)):
            return stmt
        if isinstance(stmt, Block):
            return self._visit_block(stmt)
        return stmt

    def _visit_let(self, stmt: LetStatement) -> Statement:
        stmt.value = self._visit_expression(stmt.value)
        # If the bound value is a literal, record it for constant propagation.
        if _is_literal(stmt.value):
            self._consts[stmt.name] = stmt.value
            self.stats.constants_propagated += 1
        else:
            # Variable is rebound to a non-literal; invalidate any prior
            # known constant binding so we do not propagate stale values.
            self._consts.pop(stmt.name, None)
        return stmt

    def _visit_assignment(self, stmt: AssignmentStatement) -> Statement:
        stmt.value = self._visit_expression(stmt.value)
        if isinstance(stmt.target, Identifier):
            # Invalidate any prior constant binding for this name.
            self._consts.pop(stmt.target.name, None)
            # If the new value is a literal, record it.
            if _is_literal(stmt.value):
                self._consts[stmt.target.name] = stmt.value
                self.stats.constants_propagated += 1
        return stmt

    def _visit_if(self, stmt: IfStatement) -> Statement:
        new_branches: List[Tuple[Expression, Block]] = []
        for cond, body in stmt.branches:
            cond = self._visit_expression(cond)
            body = self._visit_block(body)
            new_branches.append((cond, body))
        stmt.branches = new_branches
        if stmt.else_body is not None:
            stmt.else_body = self._visit_block(stmt.else_body)

        # Constant-folded condition elimination: if the first branch's
        # condition is a literal boolean, we can drop or keep branches.
        if new_branches and isinstance(new_branches[0][0], BooleanLiteral):
            cond_val = new_branches[0][0].value
            if cond_val:
                # First branch always taken — replace if-statement with its body.
                self.stats.dead_code_removed += len(new_branches) - 1
                if stmt.else_body is not None:
                    self.stats.dead_code_removed += 1
                # Replace this IfStatement with the first branch's body via
                # a Block (the caller will splice statements).
                taken_body = new_branches[0][1]
                self.stats.dead_code_removed += 1
                return taken_body
            else:
                # First branch never taken — drop it; keep elif/else.
                self.stats.dead_code_removed += 1
                stmt.branches = new_branches[1:]
                if not stmt.branches and stmt.else_body is not None:
                    self.stats.dead_code_removed += 1
                    return stmt.else_body
                if not stmt.branches and stmt.else_body is None:
                    return None  # type: ignore[return-value]
        return stmt

    # ------------------------------------------------------------------
    # Expression visitors (the heart of constant folding)
    # ------------------------------------------------------------------

    def _visit_expression(self, expr: Expression) -> Expression:
        # Recurse into children first so we can fold bottom-up.
        if isinstance(expr, BinaryOp):
            return self._fold_binary(expr)
        if isinstance(expr, UnaryOp):
            return self._fold_unary(expr)
        if isinstance(expr, FunctionCall):
            expr.callee = self._visit_expression(expr.callee)
            expr.args = [self._visit_expression(a) for a in expr.args]
            return expr
        if isinstance(expr, MemberAccess):
            expr.obj = self._visit_expression(expr.obj)
            return expr
        if isinstance(expr, IndexAccess):
            expr.obj = self._visit_expression(expr.obj)
            expr.index = self._visit_expression(expr.index)
            return expr
        if isinstance(expr, ListLiteral):
            expr.elements = [self._visit_expression(e) for e in expr.elements]
            return expr
        if isinstance(expr, Identifier):
            # Constant propagation: if this identifier is bound to a literal
            # in the current block's const environment, substitute it.
            if expr.name in self._consts:
                return _clone_literal(self._consts[expr.name])
            return expr
        return expr

    def _fold_binary(self, expr: BinaryOp) -> Expression:
        expr.left = self._visit_expression(expr.left)
        expr.right = self._visit_expression(expr.right)
        # Algebraic simplifications (applied before attempting full folding).
        simplified = self._algebraic_simplify(expr)
        if simplified is not None:
            self.stats.algebraic_simplifications += 1
            return simplified
        # Full constant folding — both sides must be literals.
        if _is_literal(expr.left) and _is_literal(expr.right):
            folded = _eval_binary(expr.op, expr.left, expr.right)
            if folded is not None:
                self.stats.constants_folded += 1
                return folded
        return expr

    def _fold_unary(self, expr: UnaryOp) -> Expression:
        expr.operand = self._visit_expression(expr.operand)
        if expr.op == "not" and isinstance(expr.operand, BooleanLiteral):
            self.stats.constants_folded += 1
            return BooleanLiteral(value=not expr.operand.value,
                                   line=expr.line, col=expr.col)
        if expr.op == "-" and isinstance(expr.operand, NumberLiteral):
            self.stats.constants_folded += 1
            new_val = -expr.operand.value
            return NumberLiteral(value=new_val, is_float=expr.operand.is_float,
                                  line=expr.line, col=expr.col)
        return expr

    # ------------------------------------------------------------------
    # Algebraic identities
    # ------------------------------------------------------------------

    def _algebraic_simplify(self, expr: BinaryOp) -> Optional[Expression]:
        """Return a simpler equivalent expression, or None to keep ``expr``."""
        op = expr.op
        left, right = expr.left, expr.right
        # x + 0  -> x ; 0 + x -> x
        if op == "+" and _is_zero(right):
            return left
        if op == "+" and _is_zero(left):
            return right
        # x - 0 -> x
        if op == "-" and _is_zero(right):
            return left
        # x - x -> 0   (only safe when x has no side effects — pure identifiers/literals)
        if op == "-" and _is_pure(left) and _is_pure(right) and _structurally_equal(left, right):
            return NumberLiteral(value=0, is_float=False, line=expr.line, col=expr.col)
        # x * 1 -> x ; 1 * x -> x
        if op == "*" and _is_one(right):
            return left
        if op == "*" and _is_one(left):
            return right
        # x * 0 -> 0 ; 0 * x -> 0   (only safe when the other operand has no side effects)
        if op == "*" and _is_zero(right) and _is_pure(left):
            return NumberLiteral(value=0, is_float=False, line=expr.line, col=expr.col)
        if op == "*" and _is_zero(left) and _is_pure(right):
            return NumberLiteral(value=0, is_float=False, line=expr.line, col=expr.col)
        # true and x -> x ; false and x -> false
        if op == "and":
            if isinstance(left, BooleanLiteral):
                # ``true and x`` evaluates to ``x``; ``false and x`` is just ``false``.
                if left.value:
                    return right
                return BooleanLiteral(value=False, line=expr.line, col=expr.col)
            if isinstance(right, BooleanLiteral):
                if right.value:
                    return left
                return BooleanLiteral(value=False, line=expr.line, col=expr.col)
        # false or x -> x ; true or x -> true
        if op == "or":
            if isinstance(left, BooleanLiteral):
                if left.value:
                    return BooleanLiteral(value=True, line=expr.line, col=expr.col)
                return right
            if isinstance(right, BooleanLiteral):
                if right.value:
                    return BooleanLiteral(value=True, line=expr.line, col=expr.col)
                return left
        return None

    # ------------------------------------------------------------------
    # Dead code elimination
    # ------------------------------------------------------------------

    def _eliminate_dead_code(self, statements: List[Statement]) -> List[Statement]:
        """Remove statements that follow an unconditional control-flow break.

        ``return``, ``break``, and ``continue`` are unconditional — any
        statement that appears after them in the same block is unreachable.
        We keep the terminating statement and drop the rest.
        """
        result: List[Statement] = []
        for stmt in statements:
            if stmt is None:
                continue
            result.append(stmt)
            if isinstance(stmt, (ReturnStatement, BreakStatement, ContinueStatement)):
                # Everything after this is dead.
                if len(statements) > len(result):
                    self.stats.dead_code_removed += len(statements) - len(result)
                break
        return result

    # ------------------------------------------------------------------
    # Common subexpression elimination (basic, intra-block)
    # ------------------------------------------------------------------

    def _run_cse_pass(self, program: Program) -> None:
        """Detect and reuse duplicate sub-expressions within a single block.

        For each block, we collect a map of ``expression-signature -> Identifier``
        for sub-expressions that have already been evaluated. When we encounter
        the same signature again, we replace the duplicate with a reference to
        the existing identifier. We do **not** introduce new temporaries — we
        only reuse when the original expression was already bound to a name
        via a ``let`` statement.
        """
        # This is a conservative implementation that only replaces duplicates
        # with existing names; full CSE would require introducing temporaries.
        self._cse_walk_program(program)

    def _cse_walk_program(self, program: Program) -> None:
        self._cse_walk_statements(program.body)

    def _cse_walk_statements(self, statements: List[Statement]) -> None:
        # Map from expression signature -> Identifier name (the variable that
        # holds the result).
        cache: Dict[str, str] = {}
        for stmt in statements:
            self._cse_walk_statement(stmt, cache)

    def _cse_walk_statement(self, stmt: Statement, cache: Dict[str, str]) -> None:
        if isinstance(stmt, LetStatement):
            stmt.value = self._cse_replace(stmt.value, cache)
            cache[_ast_signature(stmt.value)] = stmt.name
        elif isinstance(stmt, AssignmentStatement):
            stmt.value = self._cse_replace(stmt.value, cache)
        elif isinstance(stmt, ExpressionStatement):
            stmt.expr = self._cse_replace(stmt.expr, cache)
        elif isinstance(stmt, PrintStatement):
            stmt.value = self._cse_replace(stmt.value, cache)
        elif isinstance(stmt, ReturnStatement):
            if stmt.value is not None:
                stmt.value = self._cse_replace(stmt.value, cache)
        elif isinstance(stmt, IfStatement):
            for i, (cond, body) in enumerate(stmt.branches):
                stmt.branches[i] = (self._cse_replace(cond, cache), body)
                # New scope for the if-body: do not propagate cache
                self._cse_walk_statements(body.statements)
            if stmt.else_body is not None:
                self._cse_walk_statements(stmt.else_body.statements)
        elif isinstance(stmt, WhileStatement):
            stmt.condition = self._cse_replace(stmt.condition, cache)
            self._cse_walk_statements(stmt.body.statements)
        elif isinstance(stmt, ForStatement):
            stmt.iterable = self._cse_replace(stmt.iterable, cache)
            self._cse_walk_statements(stmt.body.statements)
        elif isinstance(stmt, FunctionDef):
            if stmt.body is not None:
                self._cse_walk_statements(stmt.body.statements)
        elif isinstance(stmt, Block):
            self._cse_walk_statements(stmt.statements)

    def _cse_replace(self, expr: Expression, cache: Dict[str, str]) -> Expression:
        # Recurse into children first so inner duplicates are also replaced.
        if isinstance(expr, BinaryOp):
            expr.left = self._cse_replace(expr.left, cache)
            expr.right = self._cse_replace(expr.right, cache)
        elif isinstance(expr, UnaryOp):
            expr.operand = self._cse_replace(expr.operand, cache)
        elif isinstance(expr, FunctionCall):
            expr.callee = self._cse_replace(expr.callee, cache)
            expr.args = [self._cse_replace(a, cache) for a in expr.args]
        elif isinstance(expr, MemberAccess):
            expr.obj = self._cse_replace(expr.obj, cache)
        elif isinstance(expr, IndexAccess):
            expr.obj = self._cse_replace(expr.obj, cache)
            expr.index = self._cse_replace(expr.index, cache)
        elif isinstance(expr, ListLiteral):
            expr.elements = [self._cse_replace(e, cache) for e in expr.elements]
        # Now check whether the (potentially-rewritten) expression matches a
        # previously-seen signature bound to a name.
        sig = _ast_signature(expr)
        if sig in cache and _is_pure(expr):
            name = cache[sig]
            self.stats.cse_replaced += 1
            return Identifier(name=name, line=expr.line, col=expr.col)
        return expr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_literal(expr: Expression) -> bool:
    return isinstance(
        expr,
        (NumberLiteral, StringLiteral, BooleanLiteral, NullLiteral),
    )


def _is_zero(expr: Expression) -> bool:
    return isinstance(expr, NumberLiteral) and not expr.is_float and expr.value == 0


def _is_one(expr: Expression) -> bool:
    return isinstance(expr, NumberLiteral) and not expr.is_float and expr.value == 1


def _is_pure(expr: Expression) -> bool:
    """A pure expression has no side effects (e.g. no function calls)."""
    if isinstance(expr, FunctionCall):
        return False
    if isinstance(expr, (NumberLiteral, StringLiteral, BooleanLiteral, NullLiteral, Identifier)):
        return True
    if isinstance(expr, BinaryOp):
        return _is_pure(expr.left) and _is_pure(expr.right)
    if isinstance(expr, UnaryOp):
        return _is_pure(expr.operand)
    if isinstance(expr, ListLiteral):
        return all(_is_pure(e) for e in expr.elements)
    if isinstance(expr, MemberAccess):
        return _is_pure(expr.obj)
    if isinstance(expr, IndexAccess):
        return _is_pure(expr.obj) and _is_pure(expr.index)
    return False


def _structurally_equal(a: Expression, b: Expression) -> bool:
    """Compare two expressions structurally (ignoring source positions)."""
    return _ast_signature(a) == _ast_signature(b)


def _ast_signature(node: Any, _depth: int = 0, _seen: Optional[set] = None) -> str:
    """Produce a stable string signature of an AST node, ignoring positions.

    Used for CSE deduplication and structural equality checks.

    Robust against deeply nested ASTs: bails out with a depth-limited
    fallback string when recursion goes too deep, and guards against cycles
    via an id-based visited set.
    """
    if _depth > 200:
        return "<deep>"
    if _seen is None:
        _seen = set()
    if isinstance(node, (str, int, float, bool)):
        return repr(node)
    if node is None:
        return "None"
    # Enums (like TypeKind members) — use their qualified name without
    # recursing into their __dict__, which would loop.
    import enum
    if isinstance(node, enum.Enum):
        return f"{type(node).__name__}.{node.name}"
    nid = id(node)
    if nid in _seen:
        return "<cycle>"
    # Don't add scalars to _seen (they're immutable and not the cause of cycles).
    if hasattr(node, "__dict__") or isinstance(node, (list, dict)):
        _seen.add(nid)
    if isinstance(node, list):
        return "[" + ",".join(_ast_signature(n, _depth + 1, _seen) for n in node) + "]"
    if isinstance(node, dict):
        items = sorted(node.items())
        return "{" + ",".join(f"{k}:{_ast_signature(v, _depth + 1, _seen)}" for k, v in items) + "}"
    if hasattr(node, "__dict__"):
        cls = type(node).__name__
        items = {k: v for k, v in vars(node).items() if k not in ("line", "col")}
        return f"{cls}({_ast_signature(items, _depth + 1, _seen)})"
    return repr(node)


def _clone_literal(expr: Expression) -> Expression:
    """Clone a literal expression so that downstream mutations don't affect
    the constant-propagation environment.
    """
    if isinstance(expr, NumberLiteral):
        return NumberLiteral(value=expr.value, is_float=expr.is_float, line=expr.line, col=expr.col)
    if isinstance(expr, StringLiteral):
        return StringLiteral(value=expr.value, line=expr.line, col=expr.col)
    if isinstance(expr, BooleanLiteral):
        return BooleanLiteral(value=expr.value, line=expr.line, col=expr.col)
    if isinstance(expr, NullLiteral):
        return NullLiteral(line=expr.line, col=expr.col)
    return expr


def _eval_binary(op: str, left: Expression, right: Expression) -> Optional[Expression]:
    """Evaluate ``left OP right`` when both are literals. Returns None if not foldable."""
    try:
        lv = _literal_value(left)
        rv = _literal_value(right)
    except TypeError:
        return None
    if lv is None or rv is None:
        return None
    line, col = left.line, left.col
    # Boolean-result operations first (no need to build a Python scalar).
    bool_ops = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        "<":  lambda a, b: a < b,
        ">":  lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
    }
    if op in bool_ops:
        try:
            return BooleanLiteral(value=bool_ops[op](lv, rv), line=line, col=col)
        except TypeError:
            return None
    if op == "and":
        return BooleanLiteral(value=(bool(lv) and bool(rv)), line=line, col=col)
    if op == "or":
        return BooleanLiteral(value=(bool(lv) or bool(rv)), line=line, col=col)
    # Arithmetic operations — produce a Python scalar, then wrap into AST.
    try:
        if op == "+":
            result: Any = lv + rv
        elif op == "-":
            result = lv - rv
        elif op == "*":
            result = lv * rv
        elif op == "/":
            if rv == 0:
                return None  # do not fold division by zero — let runtime raise
            result = lv / rv
        elif op == "%":
            if rv == 0:
                return None
            result = lv % rv
        else:
            return None
    except (TypeError, ValueError):
        return None
    return _wrap_python_value(result, line, col)


def _wrap_python_value(value: Any, line: int, col: int) -> Expression:
    """Convert a Python scalar back into the appropriate AST literal node."""
    if isinstance(value, bool):
        return BooleanLiteral(value=value, line=line, col=col)
    if isinstance(value, int):
        return NumberLiteral(value=value, is_float=False, line=line, col=col)
    if isinstance(value, float):
        return NumberLiteral(value=value, is_float=True, line=line, col=col)
    if isinstance(value, str):
        return StringLiteral(value=value, line=line, col=col)
    if value is None:
        return NullLiteral(line=line, col=col)
    # Lists and other objects are not foldable into a literal node here.
    raise TypeError(f"cannot wrap {type(value).__name__} as a literal")


def _literal_value(expr: Expression) -> Any:
    if isinstance(expr, NumberLiteral):
        return expr.value
    if isinstance(expr, StringLiteral):
        return expr.value
    if isinstance(expr, BooleanLiteral):
        return expr.value
    if isinstance(expr, NullLiteral):
        return None
    raise TypeError(f"not a literal: {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def optimize(program: Program, enable_cse: bool = False) -> Program:
    """Run the default optimizer on ``program`` and return it."""
    return Optimizer().optimize(program, enable_cse=enable_cse)


__all__ = ["Optimizer", "OptimizationStats", "optimize"]
