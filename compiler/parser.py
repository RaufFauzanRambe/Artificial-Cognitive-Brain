"""Recursive-descent parser for the Artificial Cognitive Brain language.

Consumes the token stream produced by :class:`tokenizer.Tokenizer` and builds
an :class:`ast.Program` AST.

Grammar (informal)
------------------

::

    program        := statement*
    statement      := let_stmt | assignment_stmt | if_stmt | while_stmt
                     | for_stmt | def_stmt | return_stmt | print_stmt
                     | break_stmt | continue_stmt | expr_stmt
    block          := NEWLINE INDENT statement+ DEDENT
    let_stmt       := 'let' IDENT '=' expr NEWLINE
    assignment     := target '=' expr NEWLINE
                     where target is IDENT, member access, or index access
    if_stmt        := 'if' expr ':' block
                       ('elif' expr ':' block)*
                       ('else' ':' block)?
    while_stmt     := 'while' expr ':' block
    for_stmt       := 'for' IDENT 'in' expr ':' block
    def_stmt       := 'def' IDENT '(' params? ')' ':' block
                     params := IDENT (',' IDENT)*
    return_stmt    := 'return' expr? NEWLINE
    print_stmt    := 'print' expr NEWLINE
    break_stmt    := 'break' NEWLINE
    continue_stmt := 'continue' NEWLINE
    expr_stmt      := expr NEWLINE

    expr           := or_expr
    or_expr        := and_expr ('or' and_expr)*
    and_expr       := not_expr ('and' not_expr)*
    not_expr       := 'not' not_expr | comparison
    comparison     := additive (comp_op additive)*
                     comp_op := '==' '!=' '<' '>' '<=' '>='
    additive       := multiplicative (('+' | '-') multiplicative)*
    multiplicative := unary (('*' | '/' | '%') unary)*
    unary          := '-' unary | '+' unary | postfix
    postfix        := primary (('(' args? ')') | ('.' IDENT) | ('[' expr ']'))*
    primary        := INT | FLOAT | STRING | 'true' | 'false' | 'null'
                     | IDENT | '(' expr ')' | list_literal
    list_literal   := '[' (expr (',' expr)*)? ']'

Operator precedence (lowest to highest):
    or → and → not → comparison → +/- → * / % → unary → postfix → primary
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .ast import (
    ASTNode, AssignmentStatement, BinaryOp, Block, BooleanLiteral, BreakStatement,
    ContinueStatement, PassStatement, Expression, ExpressionStatement, ForStatement,
    FunctionCall, FunctionDef, Identifier, IfStatement, IndexAccess, LetStatement,
    ListLiteral, MemberAccess, NullLiteral, NumberLiteral, PrintStatement, Program,
    ReturnStatement, Statement, StringLiteral, UnaryOp, WhileStatement,
)
from .tokenizer import Token, TokenType, Tokenizer


class ParseError(Exception):
    """Raised when the parser cannot continue given the current token stream."""

    def __init__(self, message: str, token: Optional[Token] = None) -> None:
        if token is not None:
            super().__init__(f"ParseError at line {token.line}, col {token.col}: {message}")
            self.line = token.line
            self.col = token.col
        else:
            super().__init__(f"ParseError: {message}")
            self.line = 0
            self.col = 0
        self.message = message
        self.token = token


# Maps binary operator token text to its AST representation.
COMPARISON_OPS = {"==", "!=", "<", ">", "<=", ">="}
ADDITIVE_OPS = {"+", "-"}
MULTIPLICATIVE_OPS = {"*", "/", "%"}


class Parser:
    """Recursive-descent parser. Construct with a token list, call :meth:`parse`."""

    def __init__(self, tokens: List[Token]) -> None:
        # Strip leading NEWLINEs so the parser doesn't have to handle them
        # specially — they would otherwise create empty statements.
        self._tokens: List[Token] = tokens
        self._pos: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> Program:
        """Parse the entire token stream and return a :class:`Program` AST."""
        statements: List[Statement] = []
        # Skip any leading NEWLINE / DEDENT tokens (these can occur if the
        # source starts with blank lines or trailing indentation artifacts).
        self._skip_newlines()
        while not self._at_end():
            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)
            self._skip_newlines()
        return Program(body=statements, line=1, col=0)

    # ------------------------------------------------------------------
    # Token cursor helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        # Return the EOF token (last one) for safety.
        return self._tokens[-1]

    def _advance(self) -> Token:
        tok = self._peek()
        if not self._at_end():
            self._pos += 1
        return tok

    def _at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _check(self, type_: TokenType, value: Optional[str] = None) -> bool:
        tok = self._peek()
        if tok.type != type_:
            return False
        if value is not None and tok.value != value:
            return False
        return True

    def _match(self, type_: TokenType, value: Optional[str] = None) -> Optional[Token]:
        if self._check(type_, value):
            return self._advance()
        return None

    def _expect(self, type_: TokenType, value: Optional[str] = None) -> Token:
        tok = self._peek()
        if not self._check(type_, value):
            expected = value if value is not None else type_.name
            raise ParseError(f"expected {expected!r} but got {tok.value!r}", tok)
        return self._advance()

    def _skip_newlines(self) -> None:
        while self._check(TokenType.NEWLINE):
            self._advance()

    # ------------------------------------------------------------------
    # Statement parsing
    # ------------------------------------------------------------------

    def _parse_statement(self) -> Optional[Statement]:
        """Dispatch on the current token to the appropriate statement parser."""
        tok = self._peek()
        if tok.type == TokenType.KEYWORD:
            if tok.value == "let":
                return self._parse_let()
            if tok.value == "if":
                return self._parse_if()
            if tok.value == "while":
                return self._parse_while()
            if tok.value == "for":
                return self._parse_for()
            if tok.value == "def":
                return self._parse_def()
            if tok.value == "return":
                return self._parse_return()
            if tok.value == "print":
                return self._parse_print()
            if tok.value == "break":
                self._advance()
                self._expect_newline_or_eof()
                return BreakStatement(line=tok.line, col=tok.col)
            if tok.value == "continue":
                self._advance()
                self._expect_newline_or_eof()
                return ContinueStatement(line=tok.line, col=tok.col)
            if tok.value == "pass":
                self._advance()
                self._expect_newline_or_eof()
                return PassStatement(line=tok.line, col=tok.col)
        # Fall through to assignment-or-expression statement.
        return self._parse_assignment_or_expr_statement()

    def _parse_let(self) -> LetStatement:
        kw = self._expect(TokenType.KEYWORD, "let")
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.OP, "=")
        value = self._parse_expr()
        self._expect_newline_or_eof()
        return LetStatement(name=name_tok.value, value=value, line=kw.line, col=kw.col)

    def _parse_assignment_or_expr_statement(self) -> Statement:
        """Parse either an assignment ``target = expr`` or a bare expression.

        We parse an expression first; if the next token is ``=``, the parsed
        expression is treated as an assignment target (and must be a valid
        lvalue: identifier, member access, or index access).
        """
        tok = self._peek()
        expr = self._parse_expr()
        if self._check(TokenType.OP, "="):
            self._advance()
            value = self._parse_expr()
            self._expect_newline_or_eof()
            if not isinstance(expr, (Identifier, MemberAccess, IndexAccess)):
                raise ParseError(
                    "invalid assignment target — must be identifier, "
                    "member access, or index access",
                    tok,
                )
            return AssignmentStatement(target=expr, value=value, line=tok.line, col=tok.col)
        # Compound assignment (e.g. ``x += 1``) is lowered into a regular
        # assignment with a binary op on the right-hand side.
        for op_text, bin_op in (("+=", "+"), ("-=", "-"), ("*=", "*"), ("/=", "/")):
            if self._check(TokenType.OP, op_text):
                self._advance()
                rhs = self._parse_expr()
                self._expect_newline_or_eof()
                if not isinstance(expr, (Identifier, MemberAccess, IndexAccess)):
                    raise ParseError(
                        f"invalid compound-assignment target for {op_text!r}",
                        tok,
                    )
                new_value = BinaryOp(
                    op=bin_op, left=expr, right=rhs,
                    line=tok.line, col=tok.col,
                )
                return AssignmentStatement(
                    target=expr, value=new_value, line=tok.line, col=tok.col,
                )
        self._expect_newline_or_eof()
        return ExpressionStatement(expr=expr, line=tok.line, col=tok.col)

    def _parse_if(self) -> IfStatement:
        kw = self._expect(TokenType.KEYWORD, "if")
        condition = self._parse_expr()
        self._expect(TokenType.DELIM, ":")
        body = self._parse_block()
        branches: List[Tuple[Expression, Block]] = [(condition, body)]
        else_body: Optional[Block] = None
        # Skip blank lines between branches — common Python idiom.
        self._skip_newlines()
        while self._check(TokenType.KEYWORD, "elif"):
            elif_kw = self._advance()
            elif_cond = self._parse_expr()
            self._expect(TokenType.DELIM, ":")
            elif_body = self._parse_block()
            branches.append((elif_cond, elif_body))
            self._skip_newlines()
        if self._check(TokenType.KEYWORD, "else"):
            self._advance()
            self._expect(TokenType.DELIM, ":")
            else_body = self._parse_block()
        return IfStatement(
            branches=branches, else_body=else_body, line=kw.line, col=kw.col,
        )

    def _parse_while(self) -> WhileStatement:
        kw = self._expect(TokenType.KEYWORD, "while")
        cond = self._parse_expr()
        self._expect(TokenType.DELIM, ":")
        body = self._parse_block()
        return WhileStatement(condition=cond, body=body, line=kw.line, col=kw.col)

    def _parse_for(self) -> ForStatement:
        kw = self._expect(TokenType.KEYWORD, "for")
        var_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.KEYWORD, "in")
        iterable = self._parse_expr()
        self._expect(TokenType.DELIM, ":")
        body = self._parse_block()
        return ForStatement(
            var=var_tok.value, iterable=iterable, body=body,
            line=kw.line, col=kw.col,
        )

    def _parse_def(self) -> FunctionDef:
        kw = self._expect(TokenType.KEYWORD, "def")
        name_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.DELIM, "(")
        params: List[str] = []
        if not self._check(TokenType.DELIM, ")"):
            while True:
                param_tok = self._expect(TokenType.IDENT)
                params.append(param_tok.value)
                if self._match(TokenType.DELIM, ","):
                    continue
                break
        self._expect(TokenType.DELIM, ")")
        self._expect(TokenType.DELIM, ":")
        body = self._parse_block()
        return FunctionDef(
            name=name_tok.value, params=params, body=body,
            line=kw.line, col=kw.col,
        )

    def _parse_return(self) -> ReturnStatement:
        kw = self._expect(TokenType.KEYWORD, "return")
        # ``return`` may be bare (no expression). A bare return is detected by
        # the next token being NEWLINE, EOF, or DEDENT.
        if self._check(TokenType.NEWLINE) or self._check(TokenType.EOF) or self._check(TokenType.DEDENT):
            self._expect_newline_or_eof()
            return ReturnStatement(value=None, line=kw.line, col=kw.col)
        value = self._parse_expr()
        self._expect_newline_or_eof()
        return ReturnStatement(value=value, line=kw.line, col=kw.col)

    def _parse_print(self) -> PrintStatement:
        kw = self._expect(TokenType.KEYWORD, "print")
        value = self._parse_expr()
        self._expect_newline_or_eof()
        return PrintStatement(value=value, line=kw.line, col=kw.col)

    def _expect_newline_or_eof(self) -> None:
        """Consume exactly one NEWLINE, or accept EOF / DEDENT as end-of-statement."""
        if self._check(TokenType.NEWLINE):
            self._advance()
            return
        if self._check(TokenType.EOF) or self._check(TokenType.DEDENT):
            return
        # If we're at something else, that's an error.
        tok = self._peek()
        raise ParseError(f"expected end of statement but got {tok.value!r}", tok)

    # ------------------------------------------------------------------
    # Block parsing
    # ------------------------------------------------------------------

    def _parse_block(self) -> Block:
        """Parse a statement block introduced by ``:``.

        A block is one of:
          * ``NEWLINE INDENT statement+ DEDENT``  (multi-line block)
          * a single simple statement on the same line (e.g. ``if x: print(x)``)
        """
        # Same-line simple statement form: ``if x: print(x) NEWLINE``
        if not self._check(TokenType.NEWLINE):
            stmt = self._parse_statement()
            return Block(statements=[stmt] if stmt is not None else [], line=0, col=0)
        self._expect(TokenType.NEWLINE)
        self._skip_newlines()
        self._expect(TokenType.INDENT)
        self._skip_newlines()
        statements: List[Statement] = []
        while not self._check(TokenType.DEDENT) and not self._at_end():
            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)
            self._skip_newlines()
        # DEDENT is optional at EOF — if we hit EOF first, that's fine.
        self._match(TokenType.DEDENT)
        return Block(statements=statements, line=0, col=0)

    # ------------------------------------------------------------------
    # Expression parsing (precedence climbing)
    # ------------------------------------------------------------------

    def _parse_expr(self) -> Expression:
        return self._parse_or()

    def _parse_or(self) -> Expression:
        left = self._parse_and()
        while self._check(TokenType.KEYWORD, "or"):
            op_tok = self._advance()
            right = self._parse_and()
            left = BinaryOp(op="or", left=left, right=right, line=op_tok.line, col=op_tok.col)
        return left

    def _parse_and(self) -> Expression:
        left = self._parse_not()
        while self._check(TokenType.KEYWORD, "and"):
            op_tok = self._advance()
            right = self._parse_not()
            left = BinaryOp(op="and", left=left, right=right, line=op_tok.line, col=op_tok.col)
        return left

    def _parse_not(self) -> Expression:
        if self._check(TokenType.KEYWORD, "not"):
            op_tok = self._advance()
            operand = self._parse_not()
            return UnaryOp(op="not", operand=operand, line=op_tok.line, col=op_tok.col)
        return self._parse_comparison()

    def _parse_comparison(self) -> Expression:
        left = self._parse_additive()
        while True:
            tok = self._peek()
            if tok.type == TokenType.OP and tok.value in COMPARISON_OPS:
                self._advance()
                right = self._parse_additive()
                left = BinaryOp(op=tok.value, left=left, right=right, line=tok.line, col=tok.col)
            else:
                break
        return left

    def _parse_additive(self) -> Expression:
        left = self._parse_multiplicative()
        while True:
            tok = self._peek()
            if tok.type == TokenType.OP and tok.value in ADDITIVE_OPS:
                self._advance()
                right = self._parse_multiplicative()
                left = BinaryOp(op=tok.value, left=left, right=right, line=tok.line, col=tok.col)
            else:
                break
        return left

    def _parse_multiplicative(self) -> Expression:
        left = self._parse_unary()
        while True:
            tok = self._peek()
            if tok.type == TokenType.OP and tok.value in MULTIPLICATIVE_OPS:
                self._advance()
                right = self._parse_unary()
                left = BinaryOp(op=tok.value, left=left, right=right, line=tok.line, col=tok.col)
            else:
                break
        return left

    def _parse_unary(self) -> Expression:
        tok = self._peek()
        if tok.type == TokenType.OP and tok.value in ("-", "+"):
            self._advance()
            operand = self._parse_unary()
            # ``+x`` is a no-op; ``-x`` becomes UnaryOp('-', x)
            if tok.value == "-":
                return UnaryOp(op="-", operand=operand, line=tok.line, col=tok.col)
            return operand
        return self._parse_postfix()

    def _parse_postfix(self) -> Expression:
        expr = self._parse_primary()
        while True:
            tok = self._peek()
            if tok.type == TokenType.DELIM and tok.value == "(":
                # function call
                self._advance()
                args: List[Expression] = []
                if not self._check(TokenType.DELIM, ")"):
                    while True:
                        args.append(self._parse_expr())
                        if self._match(TokenType.DELIM, ","):
                            continue
                        break
                close = self._expect(TokenType.DELIM, ")")
                expr = FunctionCall(callee=expr, args=args, line=tok.line, col=tok.col)
                continue
            if tok.type == TokenType.DELIM and tok.value == ".":
                self._advance()
                member_tok = self._expect(TokenType.IDENT)
                expr = MemberAccess(
                    obj=expr, member=member_tok.value,
                    line=tok.line, col=tok.col,
                )
                continue
            if tok.type == TokenType.DELIM and tok.value == "[":
                self._advance()
                index = self._parse_expr()
                self._expect(TokenType.DELIM, "]")
                expr = IndexAccess(obj=expr, index=index, line=tok.line, col=tok.col)
                continue
            break
        return expr

    def _parse_primary(self) -> Expression:
        tok = self._peek()
        # Literals
        if tok.type == TokenType.INT:
            self._advance()
            return NumberLiteral(value=tok.literal, is_float=False, line=tok.line, col=tok.col)
        if tok.type == TokenType.FLOAT:
            self._advance()
            return NumberLiteral(value=tok.literal, is_float=True, line=tok.line, col=tok.col)
        if tok.type == TokenType.STRING:
            self._advance()
            return StringLiteral(value=str(tok.literal), line=tok.line, col=tok.col)
        if tok.type == TokenType.KEYWORD:
            if tok.value == "true":
                self._advance()
                return BooleanLiteral(value=True, line=tok.line, col=tok.col)
            if tok.value == "false":
                self._advance()
                return BooleanLiteral(value=False, line=tok.line, col=tok.col)
            if tok.value == "null":
                self._advance()
                return NullLiteral(line=tok.line, col=tok.col)
        if tok.type == TokenType.IDENT:
            self._advance()
            return Identifier(name=tok.value, line=tok.line, col=tok.col)
        if tok.type == TokenType.DELIM and tok.value == "(":
            self._advance()
            inner = self._parse_expr()
            self._expect(TokenType.DELIM, ")")
            return inner
        if tok.type == TokenType.DELIM and tok.value == "[":
            return self._parse_list_literal()
        raise ParseError(f"unexpected token {tok.value!r} in expression", tok)

    def _parse_list_literal(self) -> ListLiteral:
        open_tok = self._expect(TokenType.DELIM, "[")
        elements: List[Expression] = []
        if not self._check(TokenType.DELIM, "]"):
            while True:
                elements.append(self._parse_expr())
                if self._match(TokenType.DELIM, ","):
                    # Trailing comma: ``[1, 2, 3,]`` is allowed.
                    if self._check(TokenType.DELIM, "]"):
                        break
                    continue
                break
        self._expect(TokenType.DELIM, "]")
        return ListLiteral(elements=elements, line=open_tok.line, col=open_tok.col)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def parse(source: str) -> Program:
    """Tokenize and parse ``source`` in one call. Raises :class:`ParseError`."""
    from .tokenizer import Tokenizer
    tokens = Tokenizer(source).tokenize()
    return Parser(tokens).parse()


__all__ = ["Parser", "ParseError", "parse"]
