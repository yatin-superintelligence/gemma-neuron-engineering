"""Prospective random-weight Gemma causal-path fixture. No training."""
from pathlib import Path
import datetime
import hashlib
import inspect
import json
import time
import torch
import transformers
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration
from transformers.models.gemma4 import modeling_gemma4

ROOT = Path(__file__).resolve().parents[3]


def main():
    start = time.monotonic()
    protocol_path = Path(__file__).with_name('host_path_protocol.json')
    protocol = json.loads(protocol_path.read_text())
    source = Path(inspect.getfile(modeling_gemma4))
    assert transformers.__version__ == '5.8.0'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol['model_source_sha256']
    output = ROOT / 'runs/n03/host_paths'
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'result.json').exists(), 'Do not replace a completed measurement'
    torch.set_num_threads(1)
    ids = torch.tensor([protocol['tokens']])
    records = []
    calls = 0
    for seed in protocol['seeds']:
        for ple in protocol['PLE_widths']:
            for sharing in protocol['shared_KV_counts']:
                torch.manual_seed(seed)
                config = Gemma4TextConfig(vocab_size=32, vocab_size_per_layer_input=32,
                    hidden_size=16, intermediate_size=24, num_hidden_layers=6,
                    num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                    global_head_dim=8, num_global_key_value_heads=2, num_kv_shared_layers=sharing,
                    hidden_size_per_layer_input=ple, use_double_wide_mlp=False,
                    layer_types=protocol['layer_types'], max_position_embeddings=64,
                    sliding_window=16, rope_parameters={
                        'full_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
                        'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0}})
                wrapper = Gemma4Config(text_config=config.to_dict())
                wrapper.vision_config = None
                wrapper.audio_config = None
                model = Gemma4ForConditionalGeneration(wrapper).double().eval().requires_grad_(False)
                layers = model.model.language_model.layers
                def forward():
                    nonlocal calls
                    calls += 1
                    return model(ids, use_cache=False, logits_to_keep=1).logits[0, -1]
                with torch.no_grad():
                    baseline = forward()
                activity = {}
                handles = []
                def capture(index):
                    def hook(module, args):
                        args[0].retain_grad()
                        activity[index] = args[0]
                    return hook
                try:
                    handles.append(model.model.language_model.embed_tokens.register_forward_hook(
                        lambda module, args, out: out.detach().requires_grad_(True)))
                    handles.extend(layer.mlp.down_proj.register_forward_pre_hook(capture(i)) for i, layer in enumerate(layers))
                    logits = forward()
                    (logits[3] - logits[4]).backward()
                    gradients = {i: {'past_L2': float(x.grad[:, :-1].norm()),
                                     'current_L2': float(x.grad[:, -1:].norm())} for i, x in activity.items()}
                finally:
                    for handle in handles:
                        handle.remove()
                edits = []
                for index, layer in enumerate(layers):
                    for scope in ['past', 'current']:
                        def zero(module, args, scope=scope):
                            values = args[0].clone()
                            if scope == 'past':
                                values[:, :-1] = 0
                            else:
                                values[:, -1:] = 0
                            return (values, *args[1:])
                        handle = layer.mlp.down_proj.register_forward_pre_hook(zero)
                        try:
                            with torch.no_grad():
                                logits = forward()
                            edits.append({'layer': index, 'scope': scope, 'exact_baseline': torch.equal(logits, baseline),
                                          'max_abs_logit_change': float((logits - baseline).abs().max())})
                        finally:
                            handle.remove()
                with torch.no_grad():
                    restored = forward()
                assert torch.equal(restored, baseline)
                assert all(not p.requires_grad and p.grad is None for p in model.parameters())
                writers = [i for i, layer in enumerate(layers) if not layer.self_attn.is_kv_shared_layer]
                last_writer = max(writers)
                expected_zero = [r for r in edits if r['scope'] == 'past' and r['layer'] >= last_writer]
                expected_nonzero = [r for r in edits if r not in expected_zero]
                record = {'seed': seed, 'PLE_width': ple, 'shared_KV_count': sharing,
                          'last_new_KV_writer': last_writer, 'gradients': gradients,
                          'interventions': edits, 'restored_exact': True, 'optimizer_steps': 0,
                          'expected_zero_exact': all(r['exact_baseline'] for r in expected_zero),
                          'expected_zero_gradients': all(gradients[i]['past_L2'] == 0 for i in range(last_writer, len(layers))),
                          'all_other_interventions_changed_logits': all(not r['exact_baseline'] for r in expected_nonzero)}
                records.append(record)
                with (output / 'cases.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps({k: record[k] for k in ['seed','PLE_width','shared_KV_count','expected_zero_exact','expected_zero_gradients','all_other_interventions_changed_logits']}), flush=True)
    result = {'experiment': 'N03H', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model_source_sha256': protocol['model_source_sha256'],
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'cases': len(records), 'interventions': sum(len(r['interventions']) for r in records),
              'model_forward_calls': calls, 'optimizer_steps': 0, 'seconds': time.monotonic()-start,
              'all_expected_zero_paths_exact': all(r['expected_zero_exact'] and r['expected_zero_gradients'] for r in records),
              'all_other_fixture_interventions_changed_logits': all(r['all_other_interventions_changed_logits'] for r in records),
              'full_checkpoint_tested': False,
              'interpretation': 'Random-weight actual-implementation fixture only. Shared-KV path constraints require actual-checkpoint verification before using them to interpret E2B neurons.'}
    (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert calls == protocol['expected_model_forward_calls'] and len(records) == 8


if __name__ == '__main__':
    main()
