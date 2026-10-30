"""Validate compact current-cache witnesses against explicit architecture size."""
import hashlib
import json
import re


def check_cache(witness, expected_layers, dtype="torch.bfloat16"):
    layers = witness["layers"]
    assert len(layers) == expected_layers
    assert [layer["layer"] for layer in layers] == list(range(expected_layers))
    canonical = json.dumps(layers, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == witness["sha256"]
    for layer in layers:
        assert set(layer) == {"layer", "keys", "values"}
        assert (layer["keys"] is None) == (layer["values"] is None)
        for name in ["keys", "values"]:
            item = layer[name]
            if item is None:
                continue
            assert set(item) == {"shape", "dtype", "sha256"}
            assert item["dtype"] == dtype
            assert len(item["shape"]) == 4 and item["shape"][0] == 1
            assert all(isinstance(n, int) and n > 0 for n in item["shape"])
            assert isinstance(item["sha256"], str)
            assert re.fullmatch("[a-f0-9]{64}", item["sha256"])
    assert any(layer["keys"] is not None for layer in layers)
