"""Describe output text regions without grading or executing a program."""

import ast
import re


def python_regions(text):
    pattern = re.compile(r'```(?:python3?|py)?[ \t]*\r?\n(?P<body>.*?)```', re.I | re.S)
    regions = [list(m.span('body')) for m in pattern.finditer(text)]
    if not regions:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = None
        if tree and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)):
            regions = [[0, len(text)]]
    return regions


def label_prefixes(text, prefixes, token_ids, special_ids, coding=True):
    """Unstable Unicode prefix decoding remains explicitly unaligned.

    Labels describe textual regions, not executed computation or neuron roles.
    The caller supplies decoded native prefixes; no generation occurs here.
    """
    if len(prefixes) != len(token_ids) or not prefixes or prefixes[-1] != text:
        raise ValueError('Complete exact native decoding is required')
    regions = python_regions(text) if coding else []
    previous, previous_ok = '', True
    result = []
    for index, (decoded, token) in enumerate(zip(prefixes, token_ids)):
        current_ok = text.startswith(decoded)
        start, end = len(previous), len(decoded)
        aligned = previous_ok and current_ok and start <= end
        if token in special_ids:
            label = 'special'
        elif not aligned:
            label = 'unaligned'
        elif start == end:
            label = 'zero_width'
        elif not coding:
            label = 'retention_content'
        elif any(a <= start and end <= b for a, b in regions):
            label = 'python_region'
        elif any(start < b and end > a for a, b in regions):
            label = 'boundary'
        else:
            label = 'outside_python_region'
        result.append({'generated_index': index, 'category': label, 'start': start if aligned else None, 'end': end if aligned else None})
        previous, previous_ok = decoded, current_ok
    return {'python_regions': regions, 'tokens': result}
