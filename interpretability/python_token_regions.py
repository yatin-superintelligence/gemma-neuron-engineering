"""Label Python text syntax without execution, grading or model inference."""

import io
import keyword
import token
import tokenize


def lexical_regions(text, regions):
    output, errors = [], []
    for start, end in regions:
        body = text[start:end]
        lines = body.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        def absolute(position):
            line, column = position
            return start + offsets[min(line - 1, len(offsets) - 1)] + column
        try:
            for item in tokenize.generate_tokens(io.StringIO(body).readline):
                kind = {
                    token.COMMENT: 'comment', token.STRING: 'string',
                    token.NUMBER: 'number', token.OP: 'operator',
                    token.NAME: 'keyword' if keyword.iskeyword(item.string) else 'name',
                }.get(item.type)
                if kind is not None:
                    a, b = absolute(item.start), absolute(item.end)
                    output.append({'start': a, 'end': b, 'category': kind})
        except (tokenize.TokenError, IndentationError, SyntaxError) as error:
            errors.append({'region_start': start, 'type': type(error).__name__})
    return output, errors


def label_lexical(text, annotation):
    syntax, errors = lexical_regions(text, annotation['python_regions'])
    result = []
    for item in annotation['tokens']:
        category = item['category']
        if category == 'python_region':
            kinds = {s['category'] for s in syntax if item['start'] < s['end'] and item['end'] > s['start']}
            category = next(iter(kinds)) if len(kinds) == 1 else 'mixed_python' if kinds else 'python_whitespace_or_unparsed'
        result.append({**item, 'category': category})
    return {'tokens': result, 'tokenization_errors': errors}
