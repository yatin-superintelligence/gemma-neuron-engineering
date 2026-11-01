"""Describe saved Python sources without executing, compiling or grading them."""

import ast
from collections import Counter
import hashlib
import json
import re


CATEGORIES = (
    'identical_emitted_tokens',
    'different_tokens_same_extracted_source',
    'different_source_same_AST',
    'different_AST',
)
OUTCOMES = ('preserved_correct', 'both_wrong', 'gain', 'loss')
CONTROL_NODES = (
    'If', 'IfExp', 'For', 'AsyncFor', 'While', 'Try', 'TryStar', 'With',
    'AsyncWith', 'Match', 'Return', 'Break', 'Continue', 'Raise', 'Assert',
    'comprehension',
)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def observe(record, function_name):
    """Parse one already audited reply. Outer stripping/fences match extraction.

    Names, literals, docstrings, contexts and operators remain in the AST.
    Only line/column attributes are excluded from its structural comparison.
    This does not establish semantic equivalence or revalidate correctness.
    """
    if not isinstance(function_name, str) or not function_name.isidentifier():
        raise ValueError('Explicit valid function name required')
    if record.get('complete') is not True or record.get('termination') != 'eos':
        raise ValueError('Only completed native-EOS replies may be described')
    tokens = record.get('generated_ids')
    if (not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError('Nonempty integer generated-token sequence required')
    if type(record.get('generated_tokens')) is not int or record['generated_tokens'] != len(tokens):
        raise ValueError('Saved token count mismatch')
    text = record.get('text')
    if not isinstance(text, str) or len(text) > 200_000:
        raise ValueError('Bounded saved text required')
    source = text.strip()
    if source.startswith('```'):
        match = re.fullmatch(r'```(?:python|python3|py)?\s*\n([\s\S]*?)\n```', source)
        if not match:
            raise ValueError('One complete Python/plain fence without outside prose required')
        source = match[1]
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != function_name:
        raise ValueError('Exactly one complete expected function required')
    if any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
           and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)) for node in tree.body):
        raise ValueError('Outside prose or nondefinition module statement refused')
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    operators = Counter()
    controls = Counter()
    for node in ast.walk(tree):
        if isinstance(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
            operators[type(node).__name__] += 1
        if type(node).__name__ in CONTROL_NODES:
            controls[type(node).__name__] += 1
    return {
        'tokens': tuple(tokens), 'text': text, 'source': source, 'AST': normalized,
        'token_sha256': digest(json.dumps(tokens, separators=(',', ':'))),
        'source_sha256': digest(source), 'AST_sha256': digest(normalized),
        'operator_nodes': dict(sorted(operators.items())),
        'control_nodes': {key: controls[key] for key in CONTROL_NODES},
    }


def saved_outcome(original, changed):
    """Label existing grades; never execute a candidate or call an oracle."""
    for grade in (original, changed):
        if type(grade.get('correct')) is not bool or grade.get('source_valid') is not True:
            raise ValueError('Exact existing source-valid boolean grades required')
        if grade.get('status') != ('passed' if grade['correct'] else 'wrong_answer'):
            raise ValueError('Only complete saved executable coding outcomes accepted')
    return {
        (True, True): 'preserved_correct', (False, False): 'both_wrong',
        (False, True): 'gain', (True, False): 'loss',
    }[original['correct'], changed['correct']]


def compare(original, changed, original_grade, changed_grade):
    """Partition one pair in the fixed protocol order, preserving all outcomes."""
    tokens_same = original['tokens'] == changed['tokens']
    source_same = original['source'] == changed['source']
    ast_same = original['AST'] == changed['AST']
    if tokens_same and original['text'] != changed['text']:
        raise ValueError('Identical token IDs with conflicting decoded text')
    if tokens_same:
        category = CATEGORIES[0]
    elif source_same:
        category = CATEGORIES[1]
    elif ast_same:
        category = CATEGORIES[2]
    else:
        category = CATEGORIES[3]
    if (source_same and not ast_same) or (tokens_same and not source_same):
        raise ValueError('Contradictory observation lineage')
    outcome = saved_outcome(original_grade, changed_grade)
    if ast_same and outcome in ('gain', 'loss'):
        raise ValueError('Identical AST with conflicting saved deterministic grades')
    return {
        'category': category, 'saved_outcome': outcome,
        'emitted_tokens_identical': tokens_same, 'extracted_source_identical': source_same,
        'AST_identical': ast_same,
        'operator_multiset_identical': original['operator_nodes'] == changed['operator_nodes'],
        'control_counts_identical': original['control_nodes'] == changed['control_nodes'],
        'operator_delta': {key: changed['operator_nodes'].get(key, 0) - original['operator_nodes'].get(key, 0)
                           for key in sorted(original['operator_nodes'].keys() | changed['operator_nodes'].keys())},
        'control_delta': {key: changed['control_nodes'][key] - original['control_nodes'][key]
                          for key in CONTROL_NODES},
    }
