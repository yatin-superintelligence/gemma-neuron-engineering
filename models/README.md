# Saved neuron edits

These bundles preserve the tested neuron modifications as a pinned base, exact small tensors and deterministic recipes. Base checkpoints are downloaded from their original publisher only when an explicit reconstruction command is run. No training is performed.

| Bundle | Base and precision | Saved modifications |
| --- | --- | --- |
| `n01` | `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`, revision `0ae102ac289cc3f17500a607ad817849778b8ba7`; mixed NF4/BF16 with FP32 vectors and new branches | Exact 8-neuron and 64-neuron branch tensors; fixed last-position steering groups and gains |
| `n02/bf16_text` | `google/gemma-4-E2B-it`, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`; unquantized BF16 text decoder | Six physical edits at layer 34, channel 4063 |
| `n02/nf4_text_same_source` | The same Google revision, quantized with the included configuration | The same six edits after explicitly dequantizing the selected MLP to BF16 |
| `n04/recipe.json` | The same pinned Google BF16 text model | Nine fixed memory policies, copied source-module hashes and the cache-aware reconstruction helper |

The N02 text model retains token embeddings, per-layer embeddings, decoder and tied output head. Audio and vision modules are excluded. It has 4,628,569,344 unique text parameters. The selected partner channel for the prescribed recombination is 7978; it is a control coordinate, not a known semantic feature.

## Inspect without downloading the base

Run from the repository root after installing the project:

```bash
python -m reproduce.model_edits --bundle models/n01
python -m reproduce.model_edits --bundle models/n02/bf16_text
python -m reproduce.model_edits --bundle models/n02/nf4_text_same_source
```

The inspector checks the bundle schema, declared file hashes and zero training steps. It does not load pretrained weights. `PUBLIC_RELEASE_MANIFEST.json` also records the bundle manifest's hash.

## Reconstruct N02

The measured environment used Python 3.12, PyTorch 2.11.0, Transformers 5.8.0, Accelerate 1.14.0 and bitsandbytes 0.50.2 on a Kaggle T4. These are recorded versions, not a promise of bitwise portability across hardware or libraries.

```bash
python -m pip install -e '.[reproduce]'
python -m reproduce.model_edits --bundle models/n02/bf16_text \
  --rebuild --edit dense_delete --device cuda:0
```

This explicit command downloads the pinned base, verifies its payload SHA-256, constructs the text model, checks the original target-layer hashes, applies the stored edit and prints the resulting hashes. It leaves the edit applied to the loaded model. To use that model in an application:

```python
from reproduce.model_edits import reconstruct_n02

model, receipt = reconstruct_n02(
    'models/n02/bf16_text', 'dense_recombined', device='cuda:0'
)
```

Six declared edit names are available:

| Edit | Stored change |
| --- | --- |
| `dense_delete` | Remove the coordinated gate/up rows and down column |
| `dense_scale2` | Double the selected outgoing vector |
| `dense_clone_unsplit` | Add a full clone, giving amplification |
| `dense_clone_split` | Add a clone and split the outgoing contribution |
| `dense_new_zero` | Add a prescribed neuron with zero outgoing vector |
| `dense_recombined` | Add the target gate, partner up vector and half the target outgoing vector |

For the NF4 bundle, the selected layer becomes dense BF16 before the edit. The packed quantized storage is never sliced as if it were a dense matrix. Other quantized layers keep their declared representation.

## Verify the reconstructed weights

```bash
python -m reproduce.verify_models --bundle models/n02/bf16_text \
  --device cuda:0 --report verification_bf16.json
```

Repeat with `models/n02/nf4_text_same_source` for the paired condition. Verification reconstructs every declared edit from an unchanged parent and records all projection hashes. The manifests contain hashes for all six edits. R01 independently reconstructed the models from the published code and matched all 24 original output hashes across the two representations and two existing development prompts. It also matched all 18 previously recorded edited projection hashes. See the [R01 verification](../results/r01_reconstruction.json). The loader, model source and environment are pinned in that record.

Exact weights do not guarantee identical outputs across environments. BF16 split cloning already produced a small bounded logit difference in the original experiment. No semantic function, unrestricted coding improvement or speedup is established by these engineering checks.

## Use N01 branches and steering

```python
from reproduce.model_edits import load_n01_base, apply_n01_branch, apply_n01_gain

model, manifest = load_n01_base('models/n01', device='cuda:0')
with apply_n01_branch(model, 'models/n01', 'added8_zero'):
    pass  # Run inference here with the saved branch installed.
with apply_n01_gain(model, 'models/n01', 'boundary_top8', 0.5):
    pass  # Run inference here with the fixed activation intervention.
```

`new_MLP64_zero` adds a normalized residual MLP branch; it is not a full transformer layer. The contexts restore the original computation on exit. Steering affects the last position of each forward pass. Family labels describe how groups were selected, not proven neuron meanings. These bundles exclude counterfactual donor examples and private task records. The N01 public loader has not yet received a separate full-checkpoint package replay.

## Reconstruct the N04 memory connections

[recipe.json](n04/recipe.json) preserves all nine fixed policies, the immutable parent, source-module hashes and the exact helper hash. The added K/V weights are copies of existing parent weights, so another tensor file would duplicate the source. The recipe installs a custom memory connection; it does not insert a full transformer layer or change native MLP widths.

```python
from contextlib import nullcontext
from pathlib import Path
import json
from reproduce.model_edits import load_n02_base
from interpretability.memory_connection import memory_connection

recipe = json.loads(Path('models/n04/recipe.json').read_text())
model, manifest = load_n02_base('models/n02/bf16_text', device='cuda:0')
for policy in recipe['policies']:
    context = nullcontext() if policy['mode'] is None else memory_connection(
        model, source_index=recipe['source_layer'],
        target_index=recipe['target_layer'], mode=policy['mode'],
        strength=policy['strength'],
    )
    with context:
        pass  # Run inference using a fresh cache for each sequence and policy.
```

The context restores the original model on exit. A cache created for one policy must never be reused with another. Reconstruct through this helper and recipe; the original `save_pretrained` configuration does not describe the added branch.

The internal real-checkpoint experiment used the same published helper source and verified copied-weight equality, 32 exact off-mode rows and nine restorations. Its full-prefix versus incremental test uses a disclosed weaker BF16 diagnostic after the original numerical bound failed on the unchanged base. This is not a separate public recipe package replay or a portability guarantee. Raw task data remain private, so the complete behavior experiment is not yet publicly reproducible.

All conditions and their limits are in the [N04 aggregate audit](../results/n04_connection_summary.json). Full replacements fail all 32 tested outputs. A small key-only development gain is inconclusive. Neither a new semantic capability nor a speedup is established.

## Storage and provenance

The exact tensor bundles occupy about 1.4 MB; the N04 recipe adds only small JSON metadata. All four Safetensors files contain the exact tensor values saved by the experiments, with added attribution metadata. Each `bundle.json` records file hashes and tensor provenance. Retain the bundle, its license and notice, the reconstruction code revision and the base revision together. Do not rely on `save_pretrained` with the original configuration to reload heterogeneous edited widths.

## N05 native activation policies

[The N05 recipe](n05/recipe.json) saves every fixed coordinate group and all thirteen policies. The supplied anchor helper applies prescribed gains or actual donor-state swaps and removes hooks on exit. These are inference policies, not new trained weight checkpoints. The original GPU audit verified identity, self/rescue and restoration controls; a separate clean-package replay is not claimed.

Donor policies require unmodified activations from the caller's donor prompt at the mapped position. Raw original prompts and activation traces remain private. Fixed edit reconstruction and exact replication of the private behavioral panel are separate scopes. No coding benefit or specific neuron role was established.

## N06 full-site donor policies

[The N06 recipe](n06/recipe.json) saves all ten fixed policies and the validated MLP/shared-read helper contract. Capture actual donor states from caller-supplied prompts at identical absolute selector positions. The original GPU study and host audit passed all no-op, restoration and cache guards. This is a conditional inference policy. A separate clean-package replay and release of the private behavioral panel are not claimed.

## N07 individual-layer policies

[The N07 recipe](n07/recipe.json) retains every layer and donor policy, including the failed rule-specificity control. Apply actual caller-supplied donor states with the same native MLP helper used by N06. All original GPU and host guards passed; a separate clean-package replay is not claimed. No individual layer met the scientific eligibility criterion.

## N08: fixed conditional block policies

[The N08 recipe](n08/recipe.json) retains all thirty-three conditions, the six native layer blocks, the common layer 0 corruption and exact helper hashes. It uses caller-supplied actual donor activations, with no fitted vector or native parameter update. The generic composition helper is `interpretability/mlp_composition.py`; it composes disjoint native MLP assignments during prompt prefill and removes the hooks afterward.

The original experiment passed all 660 audited records and its self, identity, replay and restoration guards. These records establish the original experiment's execution. A separate clean-package pretrained replay has not been performed. The recipe reconstructs the mechanisms; reproducing the exact private task table requires the unreleased original inputs. Raw task and donor traces remain private.

Layers 5 to 9 meet the predefined whole-block rescue criterion. This is a conditional inference policy, not a trained checkpoint, an individual-neuron interpretation or an improvement of the unmodified model. N09 stopped before interventions because strict matching failed. N10 stopped at a retention-baseline gate before sparse tests. N11 subsequently completed its full sparse panel and failed the primary criterion; all results and controls are retained.


## N11 native groups and conditional control policies

[The complete recipe](n11/recipe.json) contains all 40 main policies, 48 individual policies and all three 256-coordinate groups in their fixed order. Use `interpretability.native_group_policy.native_group_policy` with caller-supplied actual donor states. Slice the named group to its declared size; individual policies use the declared rank. The helper composes a layer 0 background, a full block donor and sparse native-channel changes in that order.

These are reproducible activation policies, not a standalone edited checkpoint. Raw donor traces remain private. The original pretrained experiment passed its declared guards, and the public interface has an actual tiny-Gemma fixture. Independent pretrained reconstruction of this new interface has not been run. N02's separate clean-package reconstruction remains verified.

The selected 256-neuron group rescued two corrupted programs, below the required four, and its coverage failed. Cutting it from full rescue caused four losses, a result restricted to that engineered context. All individual conditions matched controls. No semantic neuron role or coding improvement is established.

`interpretability.stored_neuron_gain.stored_neuron_gain` supports prescribed factors 0, 1 and 2 for one dense outgoing column, with exact scope and restoration checks. Packed quantized weights are refused. Host fixtures cover this utility. N12 stopped before pretrained weight edits; N13 completed those pretrained edits and its host audit. The exact compact columns are now available below.


## N13 stored native outgoing-column edits

[The N13 recipe](n13/recipe.json) retains all sixteen conditions. [The compact tensor file](n13/down_columns.safetensors) contains the original, zero and doubled outgoing columns for native layer 9 channel 4632 in the pinned BF16 parent. Its total size is 9,608 bytes. It contains no task or activation trace.

Load the original parent with `load_n02_base`, verify the source-column hash in the recipe, and evaluate a prescribed factor inside `with stored_neuron_gain(model, 9, 4632, 0):`. Use factor `2` for doubling. The utility assigns the native weight column for the entire evaluation context and restores the parent when the context ends. The saved columns preserve the exact parameter variants without storing another full base model.

The original experiment passed 640 audited records, forty carried-original replays, two hundred generated identity comparisons and thirty-two restorations. Factors zero and two matched their global-activity equivalents exactly on forty inputs each. Every fixed condition had the same correctness. These verified edits produced no coding improvement. A separate clean-package pretrained replay of this bundle has not been performed.
