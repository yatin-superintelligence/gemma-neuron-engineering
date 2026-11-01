"""One separately frozen N57 report after full scientific audit; no new grading."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evaluation.native_readout_reporting import CONDITIONS, panel, render, require, integer

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def receipt_gate(receipt, guards, aggregate, aggregate_sha):
    require(receipt['status'] == 'complete_pass_all_frozen_guards'
            and receipt['all288_condition_records'] is True
            and receipt['all192_edited_records_native_EOS'] is True,
            'Whole192 complete frozen audit required before reporting')
    require(receipt['public_aggregate_sha256'] == aggregate_sha
            and receipt['actual_counters'] == aggregate['actual_counters'],
            'Exact observed aggregate and counters required')
    require(integer(receipt['closed_candidate_regrading_calls']) and receipt['closed_candidate_regrading_calls'] == 0
            and integer(receipt['host_model_forward_calls']) and receipt['host_model_forward_calls'] == 0
            and integer(receipt['host_tokenizer_loads']) and receipt['host_tokenizer_loads'] == 1,
            'Actual audit scope includes one host tokenizer load and no model/regrading')
    require(guards['all192_new_response_records'] is True and guards['all_native_EOS'] is True
            and integer(guards['scorer_calls_so_far']) and guards['scorer_calls_so_far'] == 0
            and guards['actual_counters'] == aggregate['actual_counters']
            and len(guards['envelope_sha256']) == 192,
            'Every response passed structural guards before any new grade')


def frozen_file(root, science, name):
    require(name in science['sources'] and sha(root/name) == science['sources'][name],
            'Observed source must bind original2966 freeze: '+name)
    return root/name


def raw_file(outputs, hashes, relative):
    path = outputs/relative
    require(isinstance(relative, str) and path.resolve().is_relative_to(outputs.resolve())
            and not path.is_symlink() and path.is_file() and relative in hashes
            and sha(path) == hashes[relative], 'Exact safe actual raw output required')
    return path


def bind(root, reporting_freeze):
    frozen = read(reporting_freeze)
    expected_sources = {'evaluation/native_readout_reporting.py', 'scripts/report_n57.py',
                        'research/experiments/n57/reporting/plan.json',
                        'research/experiments/n57/reporting/host.py'}
    require(set(frozen['sources']) == expected_sources, 'Entire separate four-file reporting freeze required')
    for name, h in frozen['sources'].items():
        require(sha(root/name) == h, name)
    original = root/'runs/n57/analysis_source_freeze.json'
    require(sha(original) == frozen['original_analysis_source_freeze_sha256'], 'Original2966 freeze pointer changed')
    science = read(original)
    require(len(science['sources']) == science['actual_files'] == 2966, 'Entire original source cohort required')
    for name, h in science['sources'].items():
        require(sha(root/name) == h, name)
    receipts = root/'runs/n57/version1'
    aggregate_path = receipts/'public_aggregate.json'
    aggregate = read(aggregate_path)
    audit, guards = read(receipts/'result_receipt.json'), read(receipts/'all_guards_pass_before_grading.json')
    receipt_gate(audit, guards, aggregate, sha(aggregate_path))
    outputs = root/'outputs/n57'
    hashes = read(receipts/'output_hashes.json')
    require(hashes, 'Complete observed raw hash inventory required')
    for name in hashes:
        raw_file(outputs, hashes, name)
    raw = outputs/'n57'
    manifest_path = raw_file(outputs, hashes, 'n57/input_manifest.json')
    require(read(manifest_path) == read(frozen_file(root, science, 'runs/n57/build.json'))['manifest'],
            'Exact launched manifest required')
    rows = read(frozen_file(root, science, '.local/n57_inputs/prompts.json'))
    meta = read(frozen_file(root, science, '.local/n57_inputs/tokenization.json'))
    lengths, envelopes, traces = {}, {}, {}
    for owner in [0, 1]:
        native_path = frozen_file(root, science, f'outputs/n41_A/n41_A/worker_{owner}/predictions.jsonl')
        natives = {v['id']: v for v in [json.loads(s) for s in native_path.read_text().splitlines() if s]}
        for row in [r for r in rows if r['owner'] == owner]:
            native = natives[row['id']]
            require(native['condition'] == 'original' and native['termination'] == 'eos',
                    'Exact complete original whole-response carry required')
            lengths['original', row['id']] = len(native['generated_ids'])
            for condition in CONDITIONS[1:]:
                key = row['id']+'__'+condition
                relative = f'n57/worker_{owner}/rows/{key}.json'
                path = raw_file(outputs, hashes, relative)
                envelope = read(path)
                record = envelope['prediction']
                require(envelope['key'] == key and sha(path) == guards['envelope_sha256'][key]
                        and record['id'] == row['id'] and record['condition'] == condition
                        and integer(record['owner']) and record['owner'] == owner,
                        'Every observed response envelope must bind exact audit/row/condition')
                require(record['complete'] is True and record['termination'] == 'eos'
                        and record['interruption'] is None and record['new_response_token_ceiling'] is None
                        and integer(record['old_cell_carries']) and record['old_cell_carries'] == 0
                        and integer(record['fresh_choices'], 1)
                        and record['fresh_choices'] == record['generated_tokens'] == len(record['generated_ids'])
                        and record['support_constrained_full_vocabulary_policy'] is True,
                        'Complete actual fresh constrained-policy response required')
                require(record['trace_path'] == 'traces/'+key+'.jsonl', 'Exact trace path binding required')
                trace_relative = f'n57/worker_{owner}/'+record['trace_path']
                trace = raw_file(outputs, hashes, trace_relative)
                require(sha(trace) == record['trace_sha256'], 'Observed actual trace hash required')
                lengths[condition, row['id']] = len(record['generated_ids'])
                envelopes[key], traces[key] = sha(path), record['trace_sha256']
    require(set(envelopes) == set(guards['envelope_sha256']) and len(envelopes) == 192,
            'All192 exact edited records required')
    supplement = panel(aggregate, rows, lengths, {r['id']: meta[r['id']]['input_tokens'] for r in rows})
    binding = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
               'reporting_freeze_sha256': sha(reporting_freeze),
               'original_analysis_source_freeze_sha256': sha(original),
               'aggregate_sha256': sha(aggregate_path), 'audit_sha256': sha(receipts/'result_receipt.json'),
               'pregrading_guard_receipt_sha256': sha(receipts/'all_guards_pass_before_grading.json'),
               'actual_raw_output_hashes_sha256': sha(receipts/'output_hashes.json'), 'raw_files': len(hashes),
               'all192_envelope_sha256': envelopes, 'all192_trace_sha256': traces,
               'reporting_model_tokenizer_scorer_executor_calls': 0,
               'actual_audit_host_tokenizer_loads': 1,
               'scope': 'Separate observed binding; no preoutput source freeze overwritten. Raw bindings remain private.'}
    return supplement, binding


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reporting-freeze', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    data, binding = bind(ROOT, args.reporting_freeze)
    require(not args.destination.exists(), 'Actual report destination may be created once only')
    args.destination.mkdir(parents=True)
    save(args.destination/'actual_input_binding.json', binding)
    save(args.destination/'n57_all_condition_supplement.json', data)
    figure = args.destination/'n57_summary.png'
    render(data, figure)
    save(args.destination/'figure_receipt.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'figure_path': figure.relative_to(ROOT).as_posix(),
        'figure_sha256': sha(figure), 'actual_render_invocations': 1, 'panels': 6,
        'visual_review': 'PENDING_all6latestactualpanels_must_be_viewed',
        'supplement_sha256': sha(args.destination/'n57_all_condition_supplement.json'),
        'actual_input_binding_sha256': sha(args.destination/'actual_input_binding.json'),
        'model_tokenizer_scorer_executor_calls': 0, 'closed_gates_repeated': 0})
    print({'status': 'rendered_once_pending_visual_QA', 'actual_figure': figure.relative_to(ROOT).as_posix(),
           'all_conditions_roles_and_families': True, 'new_scoring_or_model_calls': 0})


if __name__ == '__main__':
    main()
