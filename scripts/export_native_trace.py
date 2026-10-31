"""Convert one audited local observation into the offline viewer format.

No checkpoint weights, inference service or upload are used. Token labels may
come from an already-cached tokenizer; numeric labels work without one.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N38_COORDINATE_SHA256 = 'aa156bcebb717f86c444dbdf5a24f7941bf6b6f2dcbda725702c1328758e2084'


def make_trace(record, values, coordinates, down_norms, provenance, token_label=None):
    values = np.asarray(values)
    norms = np.asarray(down_norms)
    coordinates = [tuple(u) for u in coordinates]
    positions = record['positions']
    if not positions or not coordinates or coordinates != sorted(set(coordinates)):
        raise ValueError('Nonempty aligned positions and unique sorted units required')
    if values.shape != (len(positions), len(coordinates)) or values.size > 2000000:
        raise ValueError('Selected values do not align with tokens and units')
    if norms.shape != (len(coordinates),) or not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms < 0).any():
        raise ValueError('Finite measurements and nonnegative column norms required')
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('Explicit evidence provenance required')
    label = token_label or (lambda token: 'token ' + str(token))
    tokens = []
    for position in positions:
        processed, predicted = position['processed_token_id'], position['predicted_token_id']
        if type(processed) is not int or processed < 0 or predicted is not None and (type(predicted) is not int or predicted < 0):
            raise ValueError('Invalid token identity')
        if predicted is None and position['decision'] is not None:
            raise ValueError('A measured decision requires its predicted token')
        processed_label = label(processed)
        predicted_label = label(predicted) if predicted is not None else 'no next-token measurement'
        if not isinstance(processed_label, str) or not isinstance(predicted_label, str):
            raise ValueError('Token labels must be text')
        tokens.append({**position, 'processed': processed_label, 'predicted': predicted_label})
    units = []
    for i, (layer, channel) in enumerate(coordinates):
        if type(layer) is not int or type(channel) is not int or min(layer, channel) < 0:
            raise ValueError('Invalid native coordinate')
        units.append({'layer': layer, 'channel': channel, 'down_norm': float(norms[i]), 'values': values[:, i].astype(float).tolist()})
    return {'schema': 'digital-surgery-trace-v1', 'provenance': provenance, 'private': True, 'tokens': tokens, 'units': units, 'scope': 'Selected actual channel values. Column-weighted magnitude is before normalization and is not a causal contribution.', 'observed_decision_trace_sha256': record.get('observed_decision_trace_sha256')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--moments', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--audit-receipt', type=Path, required=True)
    parser.add_argument('--condition', choices=['recipient', 'donor'], required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', help='Optional local tokenizer directory or cached model ID. Never downloaded.')
    args = parser.parse_args()
    receipt = json.loads(args.audit_receipt.read_text())
    if receipt.get('status') != 'complete_private_host_audited_closed' or receipt.get('all_mandatory_guards_pass') is not True:
        raise ValueError('A completed passing observation audit is required')
    if args.output.exists():
        raise FileExistsError('Choose a new private trace destination')
    records = [json.loads(s) for s in args.observations.read_text().splitlines() if s]
    matched = [r for r in records if r['condition'] == args.condition and r['id'] == args.case_id]
    if len(matched) != 1:
        raise ValueError('Exactly one audited observation must match')
    record = matched[0]
    output_hashes = json.loads((args.audit_receipt.parent / 'output_hashes.json').read_text())
    if args.moments.resolve() != (args.observations.parent / (args.condition + '_moments.npz')).resolve():
        raise ValueError('Column norms must belong to the same observed model')
    for path in [args.observations, args.moments]:
        expected = output_hashes.get('n38/bf16_text/' + path.name)
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Observation metadata or norms differ from audited outputs')
    name = record['array_file']
    if Path(name).name != name:
        raise ValueError('Observation array must be a local basename')
    array = args.observations.parent / name
    if hashlib.sha256(array.read_bytes()).hexdigest() != record['array_sha256']:
        raise ValueError('Observation array differs from the audited record')
    design = json.loads(args.design.read_text())
    coordinates = design['coordinates']['ordered']
    if hashlib.sha256(json.dumps(coordinates, separators=(',', ':')).encode()).hexdigest() != N38_COORDINATE_SHA256:
        raise ValueError('Coordinate labels differ from the frozen N38 measurements')
    widths = [6144] * 15 + [12288] * 20
    starts = np.cumsum([0, *widths[:-1]])
    indices = []
    for layer, channel in coordinates:
        if type(layer) is not int or type(channel) is not int or not 0 <= layer < len(widths) or not 0 <= channel < widths[layer]:
            raise ValueError('Coordinate outside the pinned Gemma E2B layout')
        indices.append(starts[layer] + channel)
    with np.load(array, allow_pickle=False) as data:
        values = data['values']
    with np.load(args.moments, allow_pickle=False) as data:
        all_norms = data['down_norm']
    if all_norms.shape != (sum(widths),):
        raise ValueError('Wrong native column-norm layout')
    label = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision='3e22461f65e89153144f8adb70e3b8c2cc9845a7', local_files_only=True, trust_remote_code=False)
        label = lambda token: tokenizer.convert_ids_to_tokens(token)
    provenance = 'Private audited N38 native observation: ' + args.condition + ', case ' + args.case_id + '. This is activity, not a semantic role.'
    trace = make_trace(record, values, coordinates, all_norms[indices], provenance, label)
    trace['audit_receipt_sha256'] = hashlib.sha256(args.audit_receipt.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, allow_nan=False) + '\n')
    print(json.dumps({'private_trace': str(args.output.resolve()), 'recorded_positions': len(trace['tokens']), 'units': len(trace['units']), 'bytes': args.output.stat().st_size, 'model_forwards': 0, 'uploads': 0}))


if __name__ == '__main__':
    main()
