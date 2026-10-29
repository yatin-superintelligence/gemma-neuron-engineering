# Gemma Neuron Engineering: Interpretability and Control

**Interpretability and control through direct neural interventions, without training.**

Current study: **Gemma Neuron Engineering: Interpretability and Control**.

We study how individual gated-MLP neurons and small groups affect coding. We test activity steering, prescribed connection edits, removal, cloning and the construction of new neurons. The current validation uses Gemma; compatible models can use the same intervention primitives.

- [Goal & Intent: detailed technical specification](docs/GOAL_AND_INTENT.html)
- [Research Logbook: sequential questions, evidence and decisions](docs/RESEARCH_LOG.html)
- [Current model and precision](docs/model_identity.json)
- [Planned matched-precision comparison](docs/precision_pair_intent.json)
- [Complete N01 condition/family aggregates](results/n01_development_summary.json)

Download the repository and open `docs/index.html` to read the HTML locally. GitHub's file viewer displays HTML source.

## Research scope

No optimizer updates, fine-tuning, learned steering vectors or training of added neurons/layers in the current sprint. Gradients may measure influence; they are never applied as parameter updates. Direct prescribed weight assignments are allowed. Discovery and development select hypotheses; fresh confirmation tests them. Correlation does not establish a neuron function. Existing steering, pruning and function-preserving growth methods are precedents, not our novelty claims.

## Current results

N01 is an engineering and development pilot, not a capability result: 32 discovery prompts, 64 development prompts, 43 fixed conditions, 2,320 decisions and zero optimizer updates. The actual quantized E2B checkpoint accepted an 8-neuron parallel branch and a 64-neuron residual MLP block with zero outgoing weights. Each preserved all 64 baseline logit hashes. A silent addition proves compatibility, not new capability. Physical dense width surgery and complete transformer-layer insertion remain unvalidated on the checkpoint. **No coding improvement is established.**

The pilot uses binary code-patch decisions. Broader executable coding and retention tests are required before an improvement claim. The current checkpoint is mixed NF4/BF16, with FP32 vectors/branches; it is not full precision. Unquantized BF16 and FP32 feasibility are distinct questions.

## Physical edits and matched precision

N02 completed 460 engineering measurements on text-only E2B in unquantized BF16 and NF4 derived from the same pinned floating-point source. The NF4 condition explicitly dequantized the selected MLP. Within each unchanged dense target-layer baseline, physical deletion matched activity zeroing and doubling an outgoing weight vector matched gain-2 steering exactly on all 16 prompts. Adding a zero-output neuron preserved that dense baseline exactly. All hard checks and ten algebra comparisons passed.

Splitting a cloned neuron was not bitwise exact in BF16: the largest logit difference was 0.125, within the fixed tolerance, with unchanged top-1 tokens. These checks cover one channel in layer 34 and custom patch replay. They do not establish a neuron function, coding improvement or a speedup. Both startup failures are retained in the log. [Audited N02 engineering summary](results/n02_engineering_summary.json).

## Code

`interpretability/gated_mlp_surgery.py` implements scoped reversible activity steering and explicit dense-matrix connection/width edits. Packed integer/quantized matrices are refused by dense surgery. Gate/up rows and down columns must be coordinated.

`interpretability/expansion.py` provides separately stored new gated neurons and a normalized residual MLP branch. A residual MLP block is not a full attention/transformer layer. `interpretability/probe.py` supplies compact observation utilities.

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
```

The tests use small random-weight fixtures, including the Transformers Gemma MLP class. They download no pretrained weights and do not train a model. Passing these fixtures does not establish full-checkpoint structural compatibility.

## Publication boundary

This repository is a reviewed public projection with an independent history. It includes goals, methods, generic code, fixture tests and aggregate progress. Private notebooks, task records, predictions, activation arrays, model weights, adapters/patches, the unpublished paper and credentials are excluded. The full internal forensic log remains private during the study. Public aggregates do not provide a complete reproduction package yet.

Updates are generated from the private question/evidence register and reviewed before publication. `PUBLIC_RELEASE_MANIFEST.json` records payload hashes. No competition entry or submission is implied.

## License

Original code/documentation: Apache License 2.0. Model checkpoints and third-party assets retain their own licenses; no model weights are distributed here.
