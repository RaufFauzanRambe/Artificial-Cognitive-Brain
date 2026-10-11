"""Compiler driver for the Artificial Cognitive Brain language.

This module ties together the entire pipeline:

    source code ─► [Tokenizer] ─► [Parser] ─► [SemanticAnalyzer]
                                              │
                                              ▼
                                          [Optimizer]
                                              │
                                              ▼
                                       [CodeGenerator] ─► Python source

The driver exposes both a Python API (:class:`Compiler`, :func:`compile_source`,
:func:`compile_file`) and a command-line entry point (``python -m compiler.compiler``
or directly via the ``acb-compile`` script once installed).

CLI usage
---------

::

    # Compile a file to Python source on stdout
    python -m compiler.compiler hello.acb

    # Write the output to a file
    python -m compiler.compiler hello.acb -o hello.py

    # Print the AST (post-parse) as a debug aid
    python -m compiler.compiler hello.acb --ast

    # Print tokens only
    python -m compiler.compiler hello.acb --tokens

    # Disable the optimizer
    python -m compiler.compiler hello.acb --no-optimize

    # Run the compiled program immediately via exec()
    python -m compiler.compiler hello.acb --run

    # Show optimizer statistics
    python -m compiler.compiler hello.acb --stats

A source file conventionally uses the ``.acb`` extension (Artificial Cognitive
Brain), but the compiler does not enforce this — any text file will do.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from .tokenizer import Token, Tokenizer, TokenizerError
from .parser import ParseError, Parser
from .semantic_analyzer import SemanticAnalyzer, SemanticError, SymbolTable
from .optimizer import OptimizationStats, Optimizer
from .code_generator import CodeGenerator, CodeGenError
from .ast import Program


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CompilationResult:
    """The output of a single compilation run. Holds all intermediate artifacts."""
    source: str
    tokens: List[Token] = field(default_factory=list)
    program: Optional[Program] = None
    symbol_table: Optional[SymbolTable] = None
    optimized_program: Optional[Program] = None
    generated_code: str = ""
    stats: OptimizationStats = field(default_factory=OptimizationStats)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """True iff the compilation succeeded (i.e. ``generated_code`` was produced)."""
        return bool(self.generated_code)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CompilerError(Exception):
    """Top-level error type wrapping any failure across the pipeline."""

    def __init__(self, stage: str, message: str,
                 line: int = 0, col: int = 0) -> None:
        loc = f" at line {line}, col {col}" if line else ""
        super().__init__(f"[{stage}] {message}{loc}")
        self.stage = stage
        self.message = message
        self.line = line
        self.col = col


# ---------------------------------------------------------------------------
# Compiler driver
# ---------------------------------------------------------------------------

class Compiler:
    """High-level driver that orchestrates the full compilation pipeline.

    Construct a :class:`Compiler`, optionally toggle stages via the ``flags``
    parameter, then call :meth:`compile` with a source string.

    Example
    -------

    ::

        from compiler import Compiler

        c = Compiler()
        result = c.compile("let x = 2 + 3\\nprint x\\n")
        print(result.generated_code)
        # Output:
        # x = (2 + 3)
        # print(x)
    """

    def __init__(self, optimize: bool = True, enable_cse: bool = False,
                 analyze: bool = True) -> None:
        self.optimize_enabled: bool = optimize
        self.enable_cse: bool = enable_cse
        self.analyze_enabled: bool = analyze

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, source: str) -> CompilationResult:
        """Run the full pipeline on ``source`` and return a :class:`CompilationResult`."""
        import time
        result = CompilationResult(source=source)
        start = time.perf_counter()

        # Stage 1: Tokenize
        try:
            tokenizer = Tokenizer(source)
            result.tokens = tokenizer.tokenize()
        except TokenizerError as exc:
            raise CompilerError("tokenizer", exc.message, exc.line, exc.col) from exc

        # Stage 2: Parse
        try:
            parser = Parser(result.tokens)
            result.program = parser.parse()
        except ParseError as exc:
            raise CompilerError("parser", exc.message, exc.line, exc.col) from exc

        # Stage 3: Semantic analysis (optional)
        if self.analyze_enabled:
            analyzer = SemanticAnalyzer()
            try:
                analyzer.analyze(result.program)
                result.symbol_table = analyzer.global_scope
            except SemanticError as exc:
                raise CompilerError("semantic", exc.message, exc.line, exc.col) from exc

        # Stage 4: Optimize (optional)
        if self.optimize_enabled:
            opt = Optimizer()
            result.optimized_program = opt.optimize(result.program, enable_cse=self.enable_cse)
            result.stats = opt.stats
        else:
            result.optimized_program = result.program

        # Stage 5: Generate code
        try:
            gen = CodeGenerator()
            result.generated_code = gen.generate(result.optimized_program)
        except CodeGenError as exc:
            raise CompilerError("codegen", str(exc)) from exc

        result.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def compile_file(self, path: str) -> CompilationResult:
        """Read ``path`` from disk and compile it."""
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        return self.compile(source)

    def compile_and_run(self, source: str, run_globals: Optional[dict] = None) -> dict:
        """Compile ``source`` and immediately ``exec()`` the generated Python code.

        Returns the globals dict the code was executed in, so callers can
        inspect any top-level bindings the program created.
        """
        result = self.compile(source)
        ns: dict = run_globals if run_globals is not None else {}
        exec(compile(result.generated_code, "<acb>", "exec"), ns)
        return ns


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def compile_source(source: str, optimize: bool = True) -> str:
    """Compile ``source`` and return the generated Python code.

    Raises :class:`CompilerError` on any pipeline failure.
    """
    return Compiler(optimize=optimize).compile(source).generated_code


def compile_file(path: str, optimize: bool = True) -> str:
    """Compile the file at ``path`` and return the generated Python code."""
    return Compiler(optimize=optimize).compile_file(path).generated_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_tokens(tokens: List[Token]) -> str:
    """Pretty-print tokens one per line for the ``--tokens`` debug flag."""
    lines: List[str] = []
    for t in tokens:
        if t.type.name in ("NEWLINE",):
            lines.append(f"{t.line:4d}:{t.col:3d}  NEWLINE")
        elif t.type.name in ("INDENT", "DEDENT", "EOF"):
            lines.append(f"{t.line:4d}:{t.col:3d}  {t.type.name}")
        elif t.literal is not None:
            lines.append(f"{t.line:4d}:{t.col:3d}  {t.type.name:8s} {t.value!r}  → {t.literal!r}")
        else:
            lines.append(f"{t.line:4d}:{t.col:3d}  {t.type.name:8s} {t.value!r}")
    return "\n".join(lines)


def _format_ast(program: Program, indent: int = 0) -> str:
    """Pretty-print the AST as an indented tree (for ``--ast`` flag)."""
    lines: List[str] = []

    def emit(node, depth: int) -> None:
        prefix = "  " * depth
        if node is None:
            lines.append(f"{prefix}None")
            return
        # Use repr to show node + field values (children are listed below).
        cls = type(node).__name__
        # Show only simple fields, recurse into children separately.
        simple_fields = []
        for k, v in vars(node).items():
            if k in ("line", "col"):
                continue
            if isinstance(v, (str, int, float, bool, type(None))):
                simple_fields.append(f"{k}={v!r}")
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                simple_fields.append(f"{k}={v!r}")
        lines.append(f"{prefix}{cls}({', '.join(simple_fields)})")
        for child in node.children() if hasattr(node, "children") else []:
            emit(child, depth + 1)

    emit(program, 0)
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acb-compile",
        description="Compile Artificial Cognitive Brain source to Python.",
    )
    p.add_argument("source", nargs="?", help="path to a .acb source file "
                                             "(if omitted, reads from stdin)")
    p.add_argument("-o", "--output", help="write generated Python to this file "
                                          "(default: stdout)")
    p.add_argument("--tokens", action="store_true",
                   help="print the token stream and exit")
    p.add_argument("--ast", action="store_true",
                   help="print the AST (post-parse) and exit")
    p.add_argument("--no-optimize", action="store_true",
                   help="disable the optimizer pass")
    p.add_argument("--no-analyze", action="store_true",
                   help="disable the semantic analyzer pass (unsafe)")
    p.add_argument("--cse", action="store_true",
                   help="enable common subexpression elimination (experimental)")
    p.add_argument("--run", action="store_true",
                   help="execute the generated Python immediately via exec()")
    p.add_argument("--stats", action="store_true",
                   help="print optimizer statistics to stderr")
    p.add_argument("--time", action="store_true",
                   help="print elapsed compilation time to stderr")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Read source
    if args.source is None or args.source == "-":
        source = sys.stdin.read()
        source_name = "<stdin>"
    else:
        if not os.path.exists(args.source):
            print(f"error: file not found: {args.source}", file=sys.stderr)
            return 2
        with open(args.source, "r", encoding="utf-8") as f:
            source = f.read()
        source_name = args.source

    # Build the compiler
    compiler = Compiler(
        optimize=not args.no_optimize,
        enable_cse=args.cse,
        analyze=not args.no_analyze,
    )

    # Handle --tokens early (don't run the full pipeline)
    if args.tokens:
        try:
            tokens = Tokenizer(source).tokenize()
        except TokenizerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(_format_tokens(tokens))
        return 0

    try:
        result = compiler.compile(source)
    except CompilerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Handle --ast
    if args.ast:
        # Use the optimized program if optimization ran; otherwise the parsed one.
        ast_to_show = result.optimized_program or result.program
        if ast_to_show is None:
            print("error: no AST available", file=sys.stderr)
            return 1
        print(_format_ast(ast_to_show))
        return 0

    # Optionally run the program
    if args.run:
        try:
            ns: dict = {"__name__": "__main__", "__file__": source_name}
            exec(compile(result.generated_code, source_name, "exec"), ns)
        except Exception as exc:  # noqa: BLE001 — runtime error in user code
            print(f"runtime error: {exc}", file=sys.stderr)
            return 1

    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.generated_code)
        if args.stats:
            print(f"# optimizer stats: {result.stats}", file=sys.stderr)
        if args.time:
            print(f"# compiled in {result.elapsed_ms:.2f} ms", file=sys.stderr)
    else:
        sys.stdout.write(result.generated_code)
        if args.stats:
            print(f"\n# --- optimizer stats: {result.stats}", file=sys.stderr)
        if args.time:
            print(f"# --- compiled in {result.elapsed_ms:.2f} ms", file=sys.stderr)

    return 0


# Make the module executable as a script: ``python compiler.py hello.acb``
if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "Compiler", "CompilerError", "CompilationResult",
    "compile_source", "compile_file", "main",
]
