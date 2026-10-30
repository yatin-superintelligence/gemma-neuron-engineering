"""Recognize fixed candidate sites using only an already generated prefix."""
import ast
import hashlib
import io
import token
import tokenize


def _code(text, maximum_characters):
    if not isinstance(text, str):
        raise TypeError("The generated prefix must be text")
    if len(text) > maximum_characters:
        return None
    if text.startswith("```python\n"):
        text = text[len("```python\n"):]
    elif text.startswith("```\n"):
        text = text[len("```\n"):]
    if "```" in text:
        return None
    return text.rstrip()


def candidate_prefix(text, maximum_characters=16384):
    """Return a declared trailing name after return/if, or None.

    This recognizes an incomplete syntactic opportunity. A later token may
    extend the name or produce a noncomparison expression. Nothing here
    predicts that future token or tests correctness.
    """
    code = _code(text, maximum_characters)
    if not code:
        return None
    scanned = []
    try:
        for item in tokenize.generate_tokens(io.StringIO(code).readline):
            if item.type == token.ERRORTOKEN and not item.string.isspace():
                return None
            scanned.append(item)
    except tokenize.TokenError as error:
        if error.args[0] not in {"EOF in multi-line statement",
                                 "unexpected EOF in multi-line statement"}:
            return None
    except (IndentationError, SyntaxError):
        return None
    ignored = {token.INDENT, token.DEDENT, token.NEWLINE, tokenize.NL,
               token.ENDMARKER, token.ENCODING, tokenize.COMMENT}
    items = [item for item in scanned if item.type not in ignored]
    if not items or items[-1].type != token.NAME:
        return None
    trailing = items[-1]
    lines = code.splitlines(keepends=True)
    end = sum(len(line) for line in lines[:trailing.end[0] - 1]) + trailing.end[1]
    if end != len(code):
        return None
    # Parse only a complete generated function header. Defaults/annotations
    # are AST syntax and are never evaluated or executed.
    starts = [i for i, item in enumerate(items) if item.string == "def"]
    if not starts:
        return None
    start = starts[-1]
    depth = 0
    stop = None
    for index in range(start + 1, len(items)):
        value = items[index].string
        if value in {"(", "[", "{"}:
            depth += 1
        elif value in {")", "]", "}"}:
            depth -= 1
        elif value == ":" and depth == 0:
            stop = index
            break
    if stop is None or stop >= len(items) - 1:
        return None
    first, last = items[start], items[stop]
    if (trailing.start[0] != last.start[0]
            and len(lines[trailing.start[0] - 1])
            - len(lines[trailing.start[0] - 1].lstrip()) <= first.start[1]):
        return None
    begin = sum(len(line) for line in lines[:first.start[0] - 1]) + first.start[1]
    finish = sum(len(line) for line in lines[:last.end[0] - 1]) + last.end[1]
    try:
        function = ast.parse(code[begin:finish] + "\n    pass\n").body[0]
    except (SyntaxError, ValueError):
        return None
    if not isinstance(function, ast.FunctionDef):
        return None
    args = function.args
    declared = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
    declared.update(arg.arg for arg in (args.vararg, args.kwarg) if arg)
    # A simple generated for-target also declares a name. Tuple/unpacking
    # targets deliberately remain outside this fixed grammar.
    for index in range(stop + 1, len(items) - 2):
        if (items[index].string == "for" and items[index + 1].type == token.NAME
                and items[index + 2].string == "in"):
            declared.add(items[index + 1].string)
    if trailing.string not in declared:
        return None
    index = len(items) - 2
    parentheses = negations = 0
    while index > stop and items[index].string in {"(", "not"}:
        parentheses += items[index].string == "("
        negations += items[index].string == "not"
        index -= 1
    if parentheses > 2 or negations > 1 or index <= stop:
        return None
    if items[index].string not in {"return", "if"}:
        return None
    return {"name": trailing.string, "keyword": items[index].string,
            "canonical_sha256": hashlib.sha256(code.encode()).hexdigest()}


class PrefixGate:
    """Bound and deduplicate prefix opportunities, without model answers."""

    def __init__(self, event_cap=4, maximum_characters=16384):
        if (type(event_cap) is not int or event_cap <= 0
                or type(maximum_characters) is not int or maximum_characters <= 0):
            raise ValueError("Positive integer bounds are required")
        self.event_cap = event_cap
        self.maximum_characters = maximum_characters
        self.seen = set()

    def consider(self, prefix, *, has_cache):
        if not isinstance(prefix, str):
            raise TypeError("The generated prefix must be text")
        if not has_cache or len(self.seen) >= self.event_cap:
            return None
        candidate = candidate_prefix(prefix, self.maximum_characters)
        if candidate is None or candidate["canonical_sha256"] in self.seen:
            return None
        self.seen.add(candidate["canonical_sha256"])
        return {**candidate, "event": len(self.seen)}
