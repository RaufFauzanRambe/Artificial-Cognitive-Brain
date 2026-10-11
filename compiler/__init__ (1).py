"""Artificial Cognitive Brain — Compiler Package

A small, self-contained compiler for a Python-like source language.
The compiler turns ``.acb`` source files into runnable Python source code
through a classic multi-stage pipeline:

    ┌────────────┐   ┌─────────┐   ┌──────────────┐   ┌────────────┐   ┌───────────────┐
    │ Tokenizer │ ─►│ Parser  │ ─►│  Semantic    │ ─►│ Optimizer  │ ─►│ CodeGenerator │
    │           │   │         │   │  Analyzer    │   │            │   │               │
    │ source →  │   │ tokens →│   │ AST → AST    │   │ AST → AST  │   │ AST → Python  │
    │ tokens    │   │ AST     │   │ (annotated)  │   │ (optimized)│   │ source code   │
    └────────────┘   └─────────┘   └──────────────┘   └────────────┘   └───────────────┘

Modules
-------

* :mod:`compiler.tokenizer`         — lexer / token stream
* :mod:`compiler.ast`                — AST node definitions
* :mod:`compiler.parser`             — recursive-descent parser
* :mod:`compiler.semantic_analyzer`  — symbol table, scope & type checks
* :mod:`compiler.optimizer`          — constant folding, dead-code elim, CSE
* :mod:`compiler.code_generator`      — AST → Python source
* :mod:`compiler.compiler`           — driver / CLI entry point

Quick start
-----------

::

    from compiler import Compiler

    src = '''
    let x = 2 + 3 * 4
    let y = x - 5
    print y
    if y > 0:
        print "positive"
    else:
        print "non-positive"
    '''
    result = Compiler().compile(src)
    print(result.generated_code)
    # or, run it directly:
    ns = Compiler().compile_and_run(src)
    print(ns["y"])  # → 9

Language overview
-----------------

A Python-like syntax with first-class support for variables, arithmetic,
control flow, and user-defined functions. Indentation is significant, just
like Python — blocks are introduced by ``:`` and delimited by indentation
level.

::

    # Comments start with '#'
    let name = "World"
    def greet(who):
        print "Hello, " + who + "!"
    greet(name)
    for i in [1, 2, 3]:
        if i % 2 == 0:
            print i
        else:
            print "odd"

Keywords: ``let def if elif else while for in return print true false null
and or not break continue``.
"""

from __future__ import annotations

# Sub-modules
from . import ast, tokenizer, parser, semantic_analyzer, optimizer, code_generator, compiler

# AST nodes — re-exported so callers can ``from compiler import Identifier`` etc.
from .ast import (
    ASTNode, Expression, Statement,
    NumberLiteral, StringLiteral, BooleanLiteral, NullLiteral,
    Identifier, ListLiteral, BinaryOp, UnaryOp,
    FunctionCall, MemberAccess, IndexAccess,
    Program, Block, LetStatement, AssignmentStatement,
    ExpressionStatement, IfStatement, WhileStatement, ForStatement,
    FunctionDef, ReturnStatement, PrintStatement,
    BreakStatement, ContinueStatement, PassStatement,
)

# Tokenizer
from .tokenizer import (
    Token, TokenType, Tokenizer, TokenizerError, tokenize,
    KEYWORDS,
)

# Parser
from .parser import Parser, ParseError, parse as parse_source

# Semantic analyzer
from .semantic_analyzer import (
    SemanticAnalyzer, SemanticError, SymbolTable, Symbol,
    TypeKind, FunctionType,
)

# Optimizer
from .optimizer import Optimizer, OptimizationStats, optimize

# Code generator
from .code_generator import CodeGenerator, CodeGenError, generate

# Compiler driver
from .compiler import (
    Compiler, CompilerError, CompilationResult,
    compile_source, compile_file, main as cli_main,
)

__version__ = "1.0.0"
__all__ = [
    # Sub-modules
    "ast", "tokenizer", "parser", "semantic_analyzer", "optimizer",
    "code_generator", "compiler",
    # AST nodes
    "ASTNode", "Expression", "Statement",
    "NumberLiteral", "StringLiteral", "BooleanLiteral", "NullLiteral",
    "Identifier", "ListLiteral", "BinaryOp", "UnaryOp",
    "FunctionCall", "MemberAccess", "IndexAccess",
    "Program", "Block", "LetStatement", "AssignmentStatement",
    "ExpressionStatement", "IfStatement", "WhileStatement", "ForStatement",
    "FunctionDef", "ReturnStatement", "PrintStatement",
    "BreakStatement", "ContinueStatement", "PassStatement",
    # Tokenizer
    "Token", "TokenType", "Tokenizer", "TokenizerError", "tokenize", "KEYWORDS",
    # Parser
    "Parser", "ParseError", "parse_source",
    # Semantic analyzer
    "SemanticAnalyzer", "SemanticError", "SymbolTable", "Symbol",
    "TypeKind", "FunctionType",
    # Optimizer
    "Optimizer", "OptimizationStats", "optimize",
    # Code generator
    "CodeGenerator", "CodeGenError", "generate",
    # Driver
    "Compiler", "CompilerError", "CompilationResult",
    "compile_source", "compile_file", "cli_main",
]
