"""Pinned full donor loader. No adapter, quantization or fitted conversion."""

import json
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download
from reproduce.model_edits import file_sha256, projection_hashes


def validate_text(model, expected_count=4628569344):
    text = model.model.language_model
    assert model.model.vision_tower is None and model.model.audio_tower is None
    assert model.lm_head.weight is text.embed_tokens.weight
    assert sum(p.numel() for p in model.parameters()) == expected_count
    assert len(text.layers) == 35
    matrices = {}
    with torch.no_grad():
        for p in model.parameters():
            assert p.dtype == torch.bfloat16 and not p.requires_grad and p.grad is None
            assert getattr(p, "quant_state", None) is None and torch.isfinite(p).all()
        for i, layer in enumerate(text.layers):
            width = 6144 if i < 15 else 12288
            assert list(layer.mlp.gate_proj.weight.shape) == [width, 1536]
            assert list(layer.mlp.up_proj.weight.shape) == [width, 1536]
            assert list(layer.mlp.down_proj.weight.shape) == [1536, width]
            matrices[str(i)] = projection_hashes(layer.mlp)
    return {
        "unique_parameters": expected_count,
        "all_parameters_finite_BF16": True,
        "tied_head": True,
        "media_excluded": True,
        "MLP_projection_hashes": matrices,
        "KV_entries": model.config.text_config.num_hidden_layers
        - model.config.text_config.num_kv_shared_layers,
    }


def load_donor(source_path, device="cuda:0"):
    assert transformers.__version__ == "5.8.0"
    source = json.loads(Path(source_path).read_text())
    repo, rev = source["checkpoint"], source["revision"]
    files = {}
    for name, expected in source["shards"].items():
        p = Path(hf_hub_download(repo, name, revision=rev, token=False))
        assert (
            p.stat().st_size == expected["bytes"]
            and file_sha256(p) == expected["sha256"]
        )
        files[name] = expected["sha256"]
    for name, key in [
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
    ]:
        path = hf_hub_download(repo, name, revision=rev, token=False)
        assert file_sha256(path) == source[key]
    config = transformers.AutoConfig.from_pretrained(
        repo, revision=rev, trust_remote_code=False, token=False
    )
    assert getattr(config, "quantization_config", None) is None
    config.vision_config = None
    config.audio_config = None
    model, info = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        repo,
        revision=rev,
        token=False,
        config=config,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    assert (
        not info.get("missing_keys")
        and not info.get("mismatched_keys")
        and not info.get("error_msgs")
    )
    unexpected = info.get("unexpected_keys", [])
    assert all(
        n.startswith(
            (
                "model.vision_tower.",
                "model.audio_tower.",
                "model.embed_vision.",
                "model.embed_audio.",
            )
        )
        for n in unexpected
    ), unexpected
    model.eval().requires_grad_(False)
    return model, {
        "checkpoint": repo,
        "revision": rev,
        "shard_hashes": files,
        "loading_info": info,
        "validation": validate_text(model),
    }
