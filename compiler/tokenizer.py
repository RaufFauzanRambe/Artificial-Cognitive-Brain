"""Tokenizer (lexer) for the Artificial Cognitive Brain language.

Transforms raw source text into a flat stream of :class:`Token` objects that
the parser can consume. The tokenizer follows Python-style indentation rules:
significant leading whitespace produces synthetic INDENT / DEDENT tokens so
that the parser can be written as a simple recursive-descent parser without
worrying about whitespace.

Features
--------
* Line / column tracking on every token (1-based line, 0-based column)
* Numeric literals (int and float — distinguished by presence of ``.``)
* String literals with escape sequences: ``\\n \\t \\r \\\\ \\\"``
* Comments (``#`` to end of line) — discarded entirely
* String concatenation across physical lines via ``\\`` line continuation
* INDENT / DEDENT emission based on a stack of indentation widths
* Skips blank lines and comment-only lines when computing indentation
* Emits a final NEWLINE before EOF so the parser can always end a statement cleanly

Token categories
----------------
* ``KEYWORD`` — reserved words (``let``, ``def``, ``if``, …)
* ``IDENT``   — user-defined identifiers
* ``INT``     — integer literals
* ``FLOAT``   — floating-point literals
* ``STRING``  — string literals (already unescaped)
* ``OP``      — operators (``+``, ``==``, ``and`` is keyword here)
* ``DELIM``   — delimiters: ``( ) { } [ ] : , .``
* ``NEWLINE`` — logical newline (statement separator)
* ``INDENT``  — synthetic token emitted at start of an indented block
* ``DEDENT``  — synthetic token emitted at end of an indented block
* ``EOF``     — end of input
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Set


class TokenType(Enum):
    """Enumeration of all token categories produced by :class:`Tokenizer`."""
    KEYWORD = auto()
    IDENT = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    OP = auto()
    DELIM = auto()
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


@dataclass
class Token:
    """A single token with type, lexeme, value (for literals) and position."""
    type: TokenType
    value: str
    line: int
    col: int
    # For literal tokens, the parsed Python value is stored here
    literal: Optional[object] = None

    def __repr__(self) -> str:
        loc = f"@{self.line}:{self.col}"
        if self.literal is not None:
            return f"Token({self.type.name}, {self.value!r}, {self.literal!r} {loc})"
        return f"Token({self.type.name}, {self.value!r} {loc})"

    def __eq__(self, other: object) -> bool:
        # Convenience: ``tok == "+"`` compares against the lexeme value.
        if isinstance(other, Token):
            return (self.type, self.value, self.line, self.col) == (
                other.type, other.value, other.line, other.col
            )
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.type, self.value, self.line, self.col))


# Reserved words — these are recognized as KEYWORD tokens even though they
# would otherwise match the IDENT pattern.
KEYWORDS: Set[str] = {
    "let", "def", "if", "elif", "else", "while", "for", "in",
    "return", "print", "true", "false", "null",
    "and", "or", "not", "break", "continue", "pass",
}

# Multi-character operators, longest first so that the matcher is greedy.
MULTI_CHAR_OPS: List[str] = [
    "==", "!=", "<=", ">=",
    "+=", "-=", "*=", "/=",
]
SINGLE_CHAR_OPS: Set[str] = {"+", "-", "*", "/", "%", "=", "<", ">", "!"}

# Single-character delimiters.
DELIMITERS: Set[str] = {"(", ")", "{", "}", "[", "]", ":", ",", "."}


class TokenizerError(Exception):
    """Raised when the tokenizer encounters input it cannot consume."""

    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"TokenizerError at line {line}, col {col}: {message}")
        self.line = line
        self.col = col
        self.message = message


class Tokenizer:
    """Converts a source string into a list of tokens."""

    def __init__(self, source: str) -> None:
        # Normalize line endings and ensure a trailing newline so the final
        # logical line is properly terminated.
        self.source: str = source.replace("\r\n", "\n").replace("\r", "\n")
        if not self.source.endswith("\n"):
            self.source += "\n"
        self.pos: int = 0
        self.line: int = 1
        self.col: int = 0
        # Indentation stack: starts with 0 (top-level). Each new block pushes
        # a strictly greater width; DEDENT tokens are emitted when the current
        # line's indentation drops below the top of the stack.
        self._indent_stack: List[int] = [0]
        # Tracks whether we are at the very beginning of a logical line (i.e.
        # the next non-whitespace, non-comment character determines INDENT/DEDENT).
        self._at_line_start: bool = True
        # Pending DEDENT tokens that need to be flushed before EOF.
        self._pending_dedents: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self) -> List[Token]:
        """Tokenize the source and return a list of tokens ending with EOF."""
        tokens: List[Token] = []
        for tok in self._iter_tokens():
            tokens.append(tok)
        return tokens

    # ------------------------------------------------------------------
    # Internal iteration
    # ------------------------------------------------------------------

    def _iter_tokens(self) -> Iterator[Token]:
        while self.pos < len(self.source):
            if self._at_line_start:
                # Handle a fresh logical line: figure out indentation level,
                # possibly emit INDENT / DEDENT tokens, and skip blank/comment lines.
                yield from self._handle_line_start()
                if self.pos >= len(self.source):
                    break
                if self._at_line_start:
                    # Blank or comment-only line — keep going.
                    continue

            ch = self.source[self.pos]
            if ch == "\n":
                # End of a logical line — emit NEWLINE. The parser coalesces
                # consecutive NEWLINEs, so blank lines do not produce empty
                # statements.
                self._advance()
                yield Token(TokenType.NEWLINE, "\\n", self.line, 0)
                self._at_line_start = True
                self.line += 1
                self.col = 0
                continue
            if ch == "#":
                # Comment — skip to end of line. The newline itself will be
                # handled on the next iteration.
                while self.pos < len(self.source) and self.source[self.pos] != "\n":
                    self._advance()
                continue
            if ch in (" ", "\t"):
                # Intra-line whitespace — skip without emitting anything.
                self._advance()
                continue
            if ch == "\\" and self._peek(1) == "\n":
                # Line continuation — join physical lines without emitting NEWLINE.
                self._advance()
                self._advance()
                self.line += 1
                self.col = 0
                continue
            if ch == '"':
                yield self._read_string()
                continue
            if ch.isdigit():
                yield self._read_number()
                continue
            if ch.isalpha() or ch == "_":
                yield self._read_identifier_or_keyword()
                continue
            yield self._read_operator_or_delim()

        # Flush any remaining NEWLINE if we are mid-line.
        if not self._at_line_start:
            yield Token(TokenType.NEWLINE, "\\n", self.line, 0)

        # Emit pending DEDENTs for any open blocks.
        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            yield Token(TokenType.DEDENT, "<DEDENT>", self.line, 0)

        yield Token(TokenType.EOF, "<EOF>", self.line, 0)

    # ------------------------------------------------------------------
    # Line-start handling (indentation tracking)
    # ------------------------------------------------------------------

    def _handle_line_start(self) -> Iterator[Token]:
        """At the start of a logical line, compute the indentation level and
        emit INDENT / DEDENT tokens as appropriate. Skips blank lines and
        comment-only lines so they do not affect indentation.
        """
        start_line = self.line
        indent = 0
        # Consume leading whitespace, counting columns (tabs count as 8 — a
        # pragmatic choice; mixing tabs and spaces is the user's problem).
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch == " ":
                indent += 1
                self._advance()
            elif ch == "\t":
                indent += 8
                self._advance()
            else:
                break

        # If the line is blank or a comment, do not emit INDENT/DEDENT —
        # simply consume the newline and let the caller continue.
        if self.pos >= len(self.source):
            self._at_line_start = False
            return
        if self.source[self.pos] == "\n":
            self._advance()
            self.line += 1
            self.col = 0
            # still at_line_start so caller will re-invoke us on next iter
            return
        if self.source[self.pos] == "#":
            while self.pos < len(self.source) and self.source[self.pos] != "\n":
                self._advance()
            return

        # Non-blank, non-comment line — emit INDENT or DEDENT(s).
        top = self._indent_stack[-1]
        if indent > top:
            self._indent_stack.append(indent)
            yield Token(TokenType.INDENT, "<INDENT>", start_line, 0)
        else:
            while indent < self._indent_stack[-1]:
                self._indent_stack.pop()
                yield Token(TokenType.DEDENT, "<DEDENT>", start_line, 0)
            if indent != self._indent_stack[-1]:
                # Inconsistent indentation — recover by treating the current
                # indentation as a new level aligned with what we popped to.
                # This is more forgiving than CPython but matches the
                # production-lite scope of this compiler.
                raise TokenizerError(
                    f"inconsistent indentation: expected {self._indent_stack[-1]} "
                    f"spaces but found {indent}",
                    start_line, indent,
                )

        self._at_line_start = False

    # ------------------------------------------------------------------
    # Lexeme readers
    # ------------------------------------------------------------------

    def _read_string(self) -> Token:
        start_line, start_col = self.line, self.col
        # Skip opening quote.
        self._advance()
        buf: List[str] = []
        while True:
            if self.pos >= len(self.source):
                raise TokenizerError("unterminated string literal", start_line, start_col)
            ch = self.source[self.pos]
            if ch == '"':
                self._advance()
                break
            if ch == "\\":
                self._advance()
                if self.pos >= len(self.source):
                    raise TokenizerError("unterminated escape in string", self.line, self.col)
                esc = self.source[self.pos]
                mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "0": "\0"}
                if esc in mapping:
                    buf.append(mapping[esc])
                else:
                    # Unknown escape — keep the character literally.
                    buf.append(esc)
                self._advance()
                continue
            if ch == "\n":
                raise TokenizerError("newline in string literal — use \\n", self.line, self.col)
            buf.append(ch)
            self._advance()
        value = "".join(buf)
        return Token(TokenType.STRING, f'"{value}"', start_line, start_col, literal=value)

    def _read_number(self) -> Token:
        start_line, start_col = self.line, self.col
        digits: List[str] = []
        is_float = False
        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            digits.append(self.source[self.pos])
            self._advance()
        if self.pos < len(self.source) and self.source[self.pos] == ".":
            # Make sure the next char is a digit so we don't confuse a method
            # call like ``5.bit_length()`` with a float.
            if self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit():
                is_float = True
                digits.append(".")
                self._advance()
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    digits.append(self.source[self.pos])
                    self._advance()
        text = "".join(digits)
        literal: object = float(text) if is_float else int(text)
        return Token(
            TokenType.FLOAT if is_float else TokenType.INT,
            text, start_line, start_col, literal=literal,
        )

    def _read_identifier_or_keyword(self) -> Token:
        start_line, start_col = self.line, self.col
        chars: List[str] = []
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch.isalnum() or ch == "_":
                chars.append(ch)
                self._advance()
            else:
                break
        text = "".join(chars)
        if text in KEYWORDS:
            return Token(TokenType.KEYWORD, text, start_line, start_col)
        return Token(TokenType.IDENT, text, start_line, start_col)

    def _read_operator_or_delim(self) -> Token:
        start_line, start_col = self.line, self.col
        ch = self.source[self.pos]
        # Try multi-character operators first (greedy match).
        for op in MULTI_CHAR_OPS:
            if self.source.startswith(op, self.pos):
                for _ in op:
                    self._advance()
                return Token(TokenType.OP, op, start_line, start_col)
        if ch in SINGLE_CHAR_OPS:
            self._advance()
            return Token(TokenType.OP, ch, start_line, start_col)
        if ch in DELIMITERS:
            self._advance()
            return Token(TokenType.DELIM, ch, start_line, start_col)
        raise TokenizerError(
            f"unexpected character {ch!r}",
            start_line, start_col,
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _advance(self) -> None:
        """Advance the cursor by one character, updating the column counter.

        Note that line counter bumps are handled by the callers that consume
        newlines, because newline handling has different control flow per call
        site (line-start vs. mid-line).
        """
        if self.pos < len(self.source):
            self.col += 1
            self.pos += 1

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return ""


# ---------------------------------------------------------------------------
# Convenience: keyword lookup tables for the parser
# ------------------------------------------------------------------

STATEMENT_KEYWORDS: Set[str] = {"let", "def", "if", "while", "for", "return", "print", "break", "continue", "pass"}
LITERAL_KEYWORDS: Set[str] = {"true", "false", "null"}
BINARY_KEYWORD_OPS: Dict[str, str] = {"and": "and", "or": "or"}
UNARY_KEYWORD_OPS: Dict[str, str] = {"not": "not"}


def tokenize(source: str) -> List[Token]:
    """Module-level convenience wrapper around :meth:`Tokenizer.tokenize`."""
    return Tokenizer(source).tokenize()


__all__ = [
    "TokenType", "Token", "Tokenizer", "TokenizerError",
    "KEYWORDS", "STATEMENT_KEYWORDS", "LITERAL_KEYWORDS",
    "BINARY_KEYWORD_OPS", "UNARY_KEYWORD_OPS",
    "MULTI_CHAR_OPS", "SINGLE_CHAR_OPS", "DELIMITERS",
    "tokenize",
]
