# Gemma Neuron Engineering: Interpretability and Control

We investigate what individual neurons and small groups do in a language model, and whether direct changes to them can improve coding. The current study uses Gemma 4 E2B. All current interventions use zero training: we steer activations, edit connections, remove neurons, clone them and construct prescribed additions.

**Engineering feasibility is established for the tested edits. A semantic neuron role and a coding improvement are not yet established.**

## Documentation

- [Goal & Intent](docs/GOAL_AND_INTENT.html): research questions, technical design, controls and decision rules.
- [Research Logbook](docs/RESEARCH_LOG.html): every recorded experiment, failure, result and subsequent decision.
- [Model identity and precision](docs/model_identity.json).
- [Model bundles and reconstruction instructions](models/README.md).
- [Release inventory and file hashes](PUBLIC_RELEASE_MANIFEST.json).

Clone the repository and open `docs/index.html` to read the HTML documents. GitHub's file viewer shows their source.

## Research method

A gated-MLP neuron has an incoming gate vector, an incoming value vector and an outgoing vector. We observe its activity and test its causal contribution by changing these computations. Necessity, counterfactual rescue, prompt-format controls, matched random interventions, executable coding and retention tests are needed before assigning it a useful function.

There are no optimizer updates, fine-tuning, fitted steering vectors or training of new neurons in the current study. Gradients may measure influence, but are never applied as parameter updates. Development selection and fresh confirmation are kept separate. Earlier training experiments are retained as historical groundwork in the logbook and are closed.

## Results so far

| Study | Evidence | Limit |
| --- | --- | --- |
| N01: localization, steering and additions | 43 conditions and 2,320 binary code-patch decisions. An 8-neuron parallel branch and a 64-neuron residual MLP addition with zero outgoing weights each preserved all 64 baseline logit hashes. | Silent additions establish compatibility. Selected counterfactual groups caused no choice flips on 48 recipient prompts. Activity matching was weak, so specificity is unresolved. |
| N02: physical edits and matched precision | 460 measurements on text-only, unquantized BF16 and same-source NF4 models. Physical deletion matched activity zeroing; doubling an outgoing vector matched gain-2 steering; adding a zero-output neuron preserved its dense baseline. | One channel in layer 34, on 16 reused development prompts. BF16 split cloning changed logits by up to 0.125 within the fixed tolerance. No coding or speed improvement was tested. |
| N03H: token-specific causal paths | Eight small random-weight Gemma configurations and 96 interventions. Predicted past-token zero paths had exactly unchanged later logits and zero measurement gradients. | An implementation fixture, not a pretrained E2B result or a semantic neuron role. |
| N03: native-neuron coding pilot | All 896 BF16/NF4 outputs audited. Every fixed condition scored 4/16 threshold, 4/8 signed division and 8/8 JSON controls. Identity, self-patch, rescue and restoration passed. Real-checkpoint fixed-input late-past path nulls were exact. | No coding gain or semantic role. A two-token contrast missed most dominant errors, and task wording omitted integer types. These limitations and the unchanged grades are retained. |
| N04: copied memory connection | Thirty random-weight cases passed after a preserved cache-allocation failure. The first real-model job stopped at a native BF16 cache-tolerance check before applying any connection. Its 32 baseline outputs are preserved; one amended private resume tests the eight remaining conditions. | No pretrained connection result yet. The amended cache diagnostic is weaker and disclosed. A copied connection is not a complete transformer layer or a demonstrated capability. |

The N02 representations share the pinned floating-point parent. The NF4 intervention first dequantizes the selected MLP and uses its unchanged dense form as the comparison baseline. BF16 is unquantized, but is not FP32. Full transformer-layer insertion remains unvalidated.

Complete aggregates: [N01 conditions](results/n01_development_summary.json), [N01 functional controls](results/n01_function_controls.json), [N02 engineering audit](results/n02_engineering_summary.json), [N03 coding audit](results/n03_coding_summary.json), [N04 host gate](results/n04_host_gate.json). Startup failures and scientific nulls remain in the [logbook](docs/RESEARCH_LOG.html).

## Code and saved model edits

```text
interpretability/   Observation, activation steering and dense neuron surgery
reproduce/          Pinned-base loaders, reconstruction and verification
models/n01/         Two saved branch tensors and fixed steering policies
models/n02/         BF16 and same-source NF4 physical-edit bundles
tests/              Small independent fixtures, no pretrained download
docs/               Goal & Intent and sequential Research Logbook
results/            Reviewed experiment aggregates and run history
research/experiments/n03/  Shared-KV causal-path fixture and frozen protocol
research/experiments/n04/  Copied-memory fixture and cache amendment
```

The saved models are represented by immutable base revisions, compact exact tensors and deterministic edit recipes. This reconstructs the modified models without copying the full base into Git history. Each bundle records source identity, hashes, precision, coordinates and zero optimizer steps. Changed MLP widths require the supplied loader; an unchanged Transformers configuration cannot encode them on its own.

Install and run the local checks:

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python -m reproduce.model_edits --bundle models/n02/bf16_text
```

These commands inspect small fixtures and bundle files. They do not download a pretrained model. To reconstruct or verify an actual model on a compatible GPU, follow the explicit commands in [models/README.md](models/README.md). The public N02 loader also passed an independent T4 package replay: all 24 original output hashes matched, covering six edits in each representation on two existing development prompts. See the [R01 verification](results/r01_reconstruction.json). N01 loader replay is a separate pending check.

## Current connection experiment

The prototype copies an existing trained key/value projection into a new frozen branch. It asks whether past activity from later native layers can reach future outputs through that new memory route. Fixed historical, current-only and feature-permuted controls separate access to past activity from additional computation and projection compatibility. Original weights receive no updates.

The generic helper is in [memory_connection.py](interpretability/memory_connection.py). To repeat the small random-weight cache and path checks without downloading a checkpoint:

```bash
python -m research.experiments.n04.host_gate
```

The fixture writes local records under `runs/n04/host/version2` and refuses to overwrite a completed result. Its source, cache amendment and aggregate result are public. The real-model notebook, prompts and raw outputs remain private. A new internal path alone would not show coding improvement.

## Publication scope

This repository contains reviewed goals, methods, code, tests, validated edit bundles and aggregate research records. Private notebooks, datasets, raw predictions, activation traces, historical trained adapters, credentials and the unpublished paper remain outside the repository. The public log records every experiment while the full internal forensic record remains private.

Updates are reviewed before publication. A model edit that changes behavior is not automatically a useful improvement. Existing steering, pruning and function-preserving growth methods are prior work; this project does not claim to have invented them.

## License

Original code and documentation use the Apache License 2.0. Gemma 4 is also released under [Apache 2.0](https://ai.google.dev/gemma/apache_2). Model bundles include source attribution, modification notices and their license. Third-party dependencies retain their own licenses.
