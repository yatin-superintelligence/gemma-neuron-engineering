"""Forward-only generation and strict state witnesses for N35."""

import hashlib
import json
import os

import torch
from transformers.cache_utils import DynamicCache
from interpretability.cache_proofs import check_cache
from research.experiments.n19.trajectory import cache_witness
from research.experiments.n21.generation import encode


def stable(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(
        value.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()


def configure():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)


def flags():
    return {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "TF32": torch.backends.cuda.matmul.allow_tf32,
        "threads": torch.get_num_threads(),
        "workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def snapshot(model):
    assert flags() == {
        "deterministic": True,
        "benchmark": False,
        "cudnn_deterministic": True,
        "TF32": False,
        "threads": 1,
        "workspace": ":4096:8",
    }
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not module.training for module in model.modules())
    return {
        "parameters": {
            n: {
                "version": p._version,
                "requires_grad": p.requires_grad,
                "shape": list(p.shape),
                "dtype": str(p.dtype),
            }
            for n, p in model.named_parameters()
        },
        "buffers": {n: tensor_sha(p) for n, p in model.named_buffers()},
        "hooks": {
            n: {"pre": list(m._forward_pre_hooks), "post": list(m._forward_hooks)}
            for n, m in model.named_modules()
        },
        "config": stable(model.config.to_dict()),
        "generation_config": stable(model.generation_config.to_dict()),
        "flags": flags(),
        "CPU_RNG": tensor_sha(torch.get_rng_state()),
        "CUDA_RNG": [tensor_sha(v) for v in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def assert_state(model, before):
    assert snapshot(model) == before, (
        "Native parameter/buffer/config/hook/RNG state changed"
    )


def generate(model, tokenizer, encoded, cap, eos):
    assert cap > 0 and eos
    config = model.config.text_config
    count = config.num_hidden_layers - config.num_kv_shared_layers
    cache = DynamicCache(config=config)
    ids, trace = [], []
    first = None
    with torch.no_grad():
        value = model(
            **encoded, past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        while len(ids) < cap:
            logits = value.logits[0, -1].detach()
            assert logits.ndim == 1 and torch.isfinite(logits).all()
            token = int(logits.argmax())
            witness = cache_witness(value.past_key_values)
            check_cache(witness, count, str(next(model.parameters()).dtype))
            assert all(layer["keys"] is not None for layer in witness["layers"])
            if first is None:
                first = witness
            trace.append(
                {
                    "step": len(ids),
                    "argmax": token,
                    "full_logits_sha256": tensor_sha(logits.float()),
                    "cache_sha256": witness["sha256"],
                    "raw_logits_dtype": str(logits.dtype),
                }
            )
            ids.append(token)
            if token in eos or len(ids) == cap:
                break
            value = model(
                input_ids=torch.tensor([[token]], device=encoded["input_ids"].device),
                past_key_values=value.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
    n = int(encoded["input_ids"].numel())
    return {
        "generated_ids": ids,
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "generated_tokens": len(ids),
        "cap": cap,
        "at_cap": len(ids) == cap,
        "termination": "eos" if ids[-1] in eos else "cap",
        "input_tokens": n,
        "decision_trace": trace,
        "first_cache": first,
        "expected_counter": {
            "calls": len(ids),
            "input_token_exposures": n + len(ids) - 1,
        },
    }


class Counter:
    def __init__(self, model):
        self.calls = 0
        self.exposures = 0
        self.handle = model.register_forward_pre_hook(self.count, with_kwargs=True)

    def count(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        assert ids.ndim == 2 and ids.shape[0] == 1
        self.calls += 1
        self.exposures += ids.numel()

    def value(self):
        return {"calls": self.calls, "input_token_exposures": self.exposures}

    def remove(self):
        self.handle.remove()
