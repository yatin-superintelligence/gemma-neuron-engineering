"""Explicit record and forward-trajectory manifests for N34."""


def conditions(protocol):
    result = [{'name': n, 'kind': 'parent', 'operation': n, 'layers': []} for n in protocol['parent_conditions']]
    for block in protocol['blocks']:
        for operation in protocol['operations']:
            result.append({'name': f"block{block['id']}_{operation}", 'kind': 'rescue' if operation == 'rescue' else 'new',
                           'operation': operation, 'layers': block['layers'], 'block': block['id']})
    assert len(result) == 49 and len({c['name'] for c in result}) == 49
    return result


def record_manifest(protocol, rows):
    result = [{'condition': c['name'], 'id': r['id'], 'kind': c['kind']} for c in conditions(protocol) for r in rows]
    assert len(result) == 1176
    assert {k: sum(r['kind'] == k for r in result) for k in ['parent', 'new', 'rescue']} == {'parent': 168, 'new': 840, 'rescue': 168}
    return result


def trajectories(protocol, rows, metadata, native_lengths):
    events = []
    by = {r['id']: r for r in rows}
    def add(kind, condition, identity, **extra):
        events.append({'event_id': f'{kind}:{condition}:{identity}', 'kind': kind, 'condition': condition,
                       'id': identity, 'input_tokens': metadata[identity]['input_tokens'], **extra})
    for row in rows:
        add('source_teacher', 'native', row['id'], reference_tokens=native_lengths[row['id']])
    for block in protocol['blocks']:
        for row in rows:
            add('self', f"block{block['id']}", row['id'], block=block['id'], layers=block['layers'])
    for condition in conditions(protocol):
        for row in rows:
            if condition['kind'] == 'parent':
                if condition['name'] != 'base': add('parent_current', condition['name'], row['id'])
            else:
                add('rescue' if condition['kind'] == 'rescue' else 'new_tail', condition['name'], row['id'],
                    block=condition['block'], layers=condition['layers'])
        for identity in protocol['lineage']['restoration_sentinels']:
            assert identity in by
            add('restoration', condition['name'], identity)
    assert len({e['event_id'] for e in events}) == len(events)
    expected = {'source_teacher': 24, 'self': 168, 'parent_current': 144, 'rescue': 168, 'new_tail': 840, 'restoration': 294}
    assert {k: sum(e['kind'] == k for e in events) for k in expected} == expected
    return events


def event_counter(event, generated_tokens=0):
    n = event['input_tokens']
    if event['kind'] == 'source_teacher':
        g = event['reference_tokens']
        return {'calls': g, 'input_token_exposures': n + g - 1}
    if event['kind'] == 'new_tail':
        assert 0 < generated_tokens <= 32
        return {'calls': generated_tokens, 'input_token_exposures': n + generated_tokens - 1}
    assert generated_tokens == 0
    return {'calls': 1, 'input_token_exposures': n}


def coverage(rows, identities, fields=('family', 'style', 'direction', 'order')):
    return {field: sorted({r[field] for r in rows if r['id'] in identities}) for field in fields}


def assess(protocol, rows, records, originals):
    by = {(r['condition'], r['id']): r for r in records}
    expected = {(r['condition'], r['id']) for r in record_manifest(protocol, rows)}
    assert len(by) == len(records) == len(expected) and set(by) == expected
    role = [r for r in rows if r['role'] == 'coding']
    retention = [r for r in rows if r['role'] != 'coding']
    assert len(role) == 16 and len(retention) == 8
    all_coverage = coverage(role, {r['id'] for r in role})
    result = []
    for block in protocol['blocks']:
        prefix = f"block{block['id']}_"
        changing = {}
        clauses = {}
        for policy in protocol['primary_per_block']['token_change_policies']:
            hits = {r['id'] for r in role if by[prefix + policy, r['id']]['generated_ids'][0] == originals[r[policy]]['generated_ids'][0]}
            assert all(originals[r[policy]]['generated_ids'][0] != originals[r['id']]['generated_ids'][0] for r in role)
            cover = coverage(role, hits)
            changing[policy] = {'copies': len(hits), 'cases': 16, 'coverage': cover}
            clauses[policy + '_copy_and_coverage'] = len(hits) >= 12 and cover == all_coverage
        preserving = {}
        for policy in protocol['primary_per_block']['token_preserve_policies']:
            hits = sum(by[prefix + policy, r['id']]['generated_ids'][0] == originals[r['id']]['generated_ids'][0] for r in role)
            preserving[policy] = hits
            clauses[policy + '_native_choice'] = hits >= 14
        same_correct = sum(by[prefix + 'same_same_order', r['id']]['grade']['correct'] for r in role)
        clauses['same_order_program_control'] = same_correct >= 14
        losses = {r['id'] for r in role if not by[prefix + 'zero', r['id']]['grade']['correct']
                  and by[prefix + 'zero', r['id']]['generated_ids'][0] != originals[r['id']]['generated_ids'][0]}
        zero_families = sorted({r['family'] for r in role if r['id'] in losses})
        clauses['zero_loss_with_changed_choice_both_families'] = len(losses) >= 4 and zero_families == sorted({r['family'] for r in role})
        retention_scores = {op: sum(by[prefix + op, r['id']]['grade']['correct'] for r in retention) for op in protocol['operations']}
        clauses['all_retention_under_every_operation'] = all(v == 8 for v in retention_scores.values())
        result.append({'block': block['id'], 'layers': block['layers'], 'changing': changing, 'preserving': preserving,
                       'same_order_program_correct': same_correct, 'zero_losses_and_changed_choice': len(losses),
                       'zero_loss_families': zero_families, 'retention': retention_scores,
                       'clauses': clauses, 'eligible': all(clauses.values())})
    eligible = [r['block'] for r in result if r['eligible']]
    return {'blocks': result, 'eligible_blocks': eligible, 'selected_block': min(eligible) if eligible else None,
            'selection_rule': 'Earliest eligible fixed block; every clause required. Reused development only.'}
