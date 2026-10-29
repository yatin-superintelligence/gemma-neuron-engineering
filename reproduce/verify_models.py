"""Rebuild each declared physical model; optionally compare supplied golden logits."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import torch
from safetensors.torch import load_file
from reproduce.model_edits import load_n02_base, dense_target, apply_physical_edit, projection_hashes


def verify(folder, device='cuda:0', goldens=None, progress_path=None):
    import transformers
    start = time.monotonic()
    torch.manual_seed(196)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(196)
    folder = Path(folder)
    model, manifest = load_n02_base(folder, device=device)
    layer = model.model.language_model.layers[manifest['layer']]
    original = layer.mlp
    dense = dense_target(original)
    basis = load_file(str(folder / 'basis.safetensors'))
    encoded = {}
    if goldens:
        tokenizer = transformers.AutoTokenizer.from_pretrained(manifest['base']['checkpoint'],
                    revision=manifest['base']['revision'], trust_remote_code=False)
        for row in goldens['rows']:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
            batch = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
            token_hash = hashlib.sha256(batch['input_ids'].cpu().numpy().tobytes()).hexdigest()
            if token_hash != goldens['input_token_sha256'][row['id']]:
                raise ValueError('Golden prompt tokenization differs')
            encoded[row['id']] = batch
    records = []
    predictions = []
    try:
        for edit in manifest['edits']:
            edited = apply_physical_edit(dense, edit, manifest['target'], manifest['partner'], basis)
            actual = projection_hashes(edited)
            expected = manifest['expected_edited_projection_hashes'].get(edit)
            if expected is not None and actual != expected:
                raise ValueError('Recorded edited weights mismatch: ' + edit)
            layer.mlp = edited
            replay = []
            for key, batch in encoded.items():
                with torch.inference_mode():
                    logits = model(**batch, use_cache=True, logits_to_keep=1).logits[0, -1].float().cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite reconstructed output')
                digest = hashlib.sha256(logits.numpy().tobytes()).hexdigest()
                expected_digest = goldens['expected'][edit][key]['full_logits_sha256']
                replay.append(digest == expected_digest)
                predictions.append({'edit': edit, 'id': key, 'full_logits_sha256': digest,
                                    'top1_token_id': int(logits.argmax()), 'golden_exact': digest == expected_digest})
            records.append({'edit': edit, 'projection_hashes': actual,
                            'previous_hashes_available': expected is not None,
                            'previous_hashes_exact': expected is None or actual == expected,
                            'width': layer.mlp.intermediate_size,
                            'golden_replays': len(replay), 'all_golden_replays_exact': all(replay)})
            if progress_path is not None:
                with Path(progress_path).open('a') as stream:
                    stream.write(json.dumps(records[-1]) + '\n')
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
            print(json.dumps({'edit': edit, 'complete': True, 'golden_replays_exact': all(replay)}), flush=True)
            layer.mlp = original
            del edited
    finally:
        layer.mlp = original
    if not all(not p.requires_grad and p.grad is None for p in model.parameters()):
        raise ValueError('Expected a frozen model with no parameter gradients')
    return {'status': 'complete', 'representation': manifest['representation'], 'base': manifest['base'],
            'optimizer_steps': 0, 'seed': 196, 'torch_version': torch.__version__,
            'transformers_version': transformers.__version__,
            'edits': records, 'golden_predictions': predictions,
            'all_golden_replays_exact': all(r['all_golden_replays_exact'] for r in records),
            'golden_replay_count': len(predictions), 'seconds': time.monotonic() - start,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
            'scope': 'Model reconstruction and optional supplied development-output replay;not a coding benchmark.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--goldens', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    golden = json.loads(args.goldens.read_text()) if args.goldens else None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = verify(args.bundle, device=args.device, goldens=golden,
                    progress_path=args.report.with_suffix('.progress.jsonl'))
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'edits': len(result['edits']),
                      'golden_replays': result['golden_replay_count'],
                      'all_golden_replays_exact': result['all_golden_replays_exact'], 'optimizer_steps': 0}))
    if not result['all_golden_replays_exact']:
        raise RuntimeError('Reconstruction weights passed but golden output replay failed;retain report')


if __name__ == '__main__':
    main()
