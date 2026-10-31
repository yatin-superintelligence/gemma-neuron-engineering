"""One isolated random-model CUDA worker for the two-device runtime probe."""

import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import transformers
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration
from transformers.cache_utils import DynamicCache


def digest(value):
    return hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def state(model):
    return {'parameters': {n: [p._version, p.requires_grad, p.grad is None, digest(p)] for n, p in model.named_parameters()},
            'buffers': {n: digest(b) for n, b in model.named_buffers()},
            'CPU_RNG': digest(torch.get_rng_state()),
            'CUDA_RNG': digest(torch.cuda.get_rng_state()),
            'config': stable(model.config.to_dict()),
            'training': [m.training for m in model.modules()]}


def main():
    output = Path(sys.argv[1])
    assert torch.cuda.device_count() == 1
    assert torch.__version__ == '2.11.0+cu128' and transformers.__version__ == '5.8.0'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    models = []
    for dtype in [torch.float32, torch.bfloat16]:
        torch.manual_seed(2401)
        text = Gemma4TextConfig(
            vocab_size=32, vocab_size_per_layer_input=32, hidden_size=16,
            intermediate_size=24, num_hidden_layers=35, num_attention_heads=2,
            num_key_value_heads=2, head_dim=8, global_head_dim=8,
            num_global_key_value_heads=2, num_kv_shared_layers=20,
            hidden_size_per_layer_input=4, use_double_wide_mlp=True,
            sliding_window=16, max_position_embeddings=128,
            layer_types=['full_attention' if i % 5 == 4 else 'sliding_attention' for i in range(35)],
            rope_parameters={k: {'rope_type': 'default', 'rope_theta': 10000.0}
                             for k in ['sliding_attention', 'full_attention']})
        config = Gemma4Config(text_config=text.to_dict())
        config.vision_config = config.audio_config = None
        model = Gemma4ForConditionalGeneration(config).to(device='cuda:0', dtype=dtype).eval().requires_grad_(False)
        models.append(model)
    properties = torch.cuda.get_device_properties(0)
    output.with_suffix('.ready').write_text(json.dumps({
        'physical_visible_index': os.environ['CUDA_VISIBLE_DEVICES'],
        'name': properties.name, 'total_memory_bytes': properties.total_memory}))
    deadline = time.monotonic() + 180
    while not (output.parent / 'start').exists():
        assert time.monotonic() < deadline, 'Parent start gate timed out'
        time.sleep(0.05)
    torch.cuda.synchronize()
    started = time.time_ns()
    results = []
    for model in models:
        before = state(model)
        calls = {'calls': 0, 'input_exposures': 0}

        def count(module, args, kwargs):
            calls['calls'] += 1
            calls['input_exposures'] += kwargs['input_ids'].numel()

        handle = model.register_forward_pre_hook(count, with_kwargs=True)
        cache = DynamicCache(config=model.config.text_config)
        hashes = []
        previous = None
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.no_grad():
            for step in range(4):
                ids = [[3, 4, 5, 6, 7]] if step == 0 else [[previous]]
                value = model(input_ids=torch.tensor(ids, device='cuda:0'), past_key_values=cache,
                              use_cache=True, logits_to_keep=1)
                cache = value.past_key_values
                assert len(cache.layers) == 15
                logits = value.logits[0, -1]
                previous = int(logits.argmax())
                cache_sha = stable([{k: digest(getattr(layer, k)) for k in ['keys', 'values']}
                                    for layer in cache.layers])
                hashes.append({'argmax': previous, 'FP32_logits_sha256': digest(logits.float()),
                               'raw_cache_sha256': cache_sha})
        end.record()
        torch.cuda.synchronize()
        handle.remove()
        assert calls == {'calls': 4, 'input_exposures': 8}
        assert state(model) == before
        results.append({'dtype': str(next(model.parameters()).dtype), 'decisions': hashes,
                        'counters': calls, 'state_unchanged': True,
                        'CUDA_elapsed_ms': begin.elapsed_time(end)})
    torch.cuda.synchronize()
    output.write_text(json.dumps({'status': 'passed', 'physical_visible_index': os.environ['CUDA_VISIBLE_DEVICES'],
                                 'started_ns': started, 'finished_ns': time.time_ns(),
                                 'device': properties.name, 'total_memory_bytes': properties.total_memory,
                                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                                 'measurements': results, 'pretrained_forwards': 0,
                                 'backward_calls': 0, 'optimizer_steps': 0}, indent=2) + '\n')


if __name__ == '__main__':
    main()
