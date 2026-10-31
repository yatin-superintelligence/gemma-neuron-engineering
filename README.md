# Digital Surgery

A no-training study of native neurons, small groups and direct model edits in Gemma 4 E2B. We test proposed functions through causal interventions, then evaluate fixed edits on executable Python and retention tasks.

**Physical editing and reconstruction work on the tested model. No neuron edit has established a useful coding improvement or an individual neuron meaning.**

## Findings

| Intervention | Observed result | Scope |
| --- | --- | --- |
| Zero or double a native neuron's outgoing weight column | In N13, the original and every tested condition score 16/32 coding and 8/8 retention. No output reaches its limit and no correctness changes. | One fixed neuron, L9 C4632, on dependent development cases. This is a bounded negative result. |
| Patch, zero or double native groups of 16 and 64 neurons | N20 retains selected groups and two magnitude-matched random controls. All groups have zero opposite-program transfer, clean-zero loss and conditional-transfer loss. | Full-site controls work, but the sparse role criterion fails. No coding improvement follows. |
| Swap whole native MLP states at a conditional decision | N19 switches 13/13 programs to the opposite donor choice while same-rule donors preserve 13/13. Separate relation/order controls expose literal comparator copying. | Conditional sufficiency on controlled inputs, not unrestricted coding or an abstract skill. |
| Delete, clone, recombine or add native neurons | N02 audits 460 engineering measurements in pinned BF16 and same-source NF4. R01 reconstructs all six edits in each representation through the public loader. | One selected layer and a fixed panel. BF16 split cloning has a bounded nonzero logit difference. No speed or coding benefit was tested. |

Small-group interventions also change comparator scores without changing the chosen token. The [N24 result](results/n24_finite_summary.json) reports every matched control and operation. Score movement alone does not establish a function or useful behavior.

The [Research Logbook](docs/RESEARCH_LOG.html) retains all experiments, negative results, failures, amendments and decision rules. The [question register](docs/research_questions.json) connects each proposal to its evidence. These are controlled custom panels, not a comprehensive coding-agent benchmark.

## Current work

The completed [N34 block study](results/n34_summary.json) audits all 1176 records. Two late blocks pass the copying and coding-loss components, but zeroing them also damages retention. No block passes the full selective-role criterion. All 49 [runtime policies](models/n34/recipe.json) remain available.

The [adequate-output baseline](results/n37_summary.json) is complete and audited. The unmodified original scores 24/32 coding and 7/16 retention; the compatible donor scores 23/32 and 11/16. Every extended reply finishes. All 96 outcomes remain, with no model edits or training. The [N38 observation panel](results/n38_summary.json) is complete: all 96 native trajectories and all 10,467 decision replays pass, including exact full-logit and whole-cache hashes. It records 471 fixed timeline coordinates and case-balanced moments over all 337,920 native channels. No new response or neuron edit is made. Actual traces remain private; the public viewer starts with a synthetic example.

Host text annotations show a substantial context difference: comments average 20.26% of each original coding reply and 0% of donor replies. These are dependent per-response means, not neuron functions. The completed [N39 controlled comment-entry study](results/n39_summary.json) finds that full late donor states change 40/42 current choices, while same-class donors preserve 41/42. The selected 32 channels and both strictly matched random controls cause zero choice transfer, clean-entry loss or conditional-transfer loss. The sparse primary fails. The original auditor's tied-maximum metadata error and its tested, dated repair remain preserved; no inference is rerun. All [N39 runtime recipes](models/n39/recipe.json) and outcomes remain available. No edited full coding program or coding gain is established by that panel.

The [N40 complete-function utility study](methods/n40_prospective_utility_design.json) is now running privately under its frozen protocol, source and prospective auditor. It keeps the prescribed groups and factor zero, both random controls, all 32 coding and 16 retention cases, and one fixed instruction comparison. Each new reply has a 2048-token budget with native EOS. Physical versus activation-zero, identity, cache and restoration paths pass on actual random Gemma fixtures in FP32/BF16. Pretrained utility outputs are pending. The pure [paired metrics helper](evaluation/paired_generation_metrics.py) passes 21 independent synthetic checks and preserves exact ratios, every case and uncertain comment spans.

## Offline surgery workbench

- [Editing and observation toolkit](docs/SURGERY_TOOLKIT.html): actual APIs, full source and hashes.
- [Saved model registry](docs/MODEL_REGISTRY.html): exact patch tensors, pinned parents and reconstruction code.
- [Activation viewer](docs/ACTIVATION_VIEWER.html): inspect a local compact trace, with a clearly labelled synthetic example.
- [Surgery process](docs/SURGERY_PROCESS.html): a repeatable observation, intervention, evaluation and release workflow.

Open these HTML files from a clone. They work offline and do not upload a trace. Rebuild them from the repository root with `python scripts/build_surgery_assets.py --root .`.

## Read the project

- [Goal & Intent](docs/GOAL_AND_INTENT.html): objectives, mechanisms and scientific controls.
- [Research Logbook](docs/RESEARCH_LOG.html): sequential experiment and decision history.
- [Saved edits](models/README.md): compact tensors, recipes and reconstruction limits.
- [Model identity](docs/model_identity.json) and [release inventory](PUBLIC_RELEASE_MANIFEST.json).

Clone this repository and open `docs/index.html` to view the HTML documents. GitHub's file viewer displays their source.

## Reproduce an edit

The common parent is `google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The primary text-only representation is unquantized BF16. It retains decoder, token/per-layer embeddings and tied output head while excluding media modules. BF16 is distinct from FP32; no native BF16 acceleration on T4 is claimed.

```text
interpretability/   Observation, activation patches and direct weight edits
reproduce/          Pinned-base loading, bundle inspection and reconstruction
models/             Validated compact tensors, coordinates and edit recipes
methods/            Reviewed prospective methods and amendments
results/            Complete aggregate outcomes and run history
tests/              Software fixtures without a checkpoint download
docs/               Goal & Intent, logbook, figures and question register
```

Follow [the model bundle instructions](models/README.md) for the validated loader and exact edit names. Base checkpoints are referenced by immutable revision rather than duplicated. Completed studies are stored by experiment ID; local raw evidence uses verified lossless archives, and generated caches are removed. See the [storage policy](docs/storage_policy.json). Donor-based recipes require caller-supplied contexts; private donor states are excluded. R01 verifies the N02 reconstruction path. Other recipes state their replay limits explicitly. Full transformer-layer insertion remains unvalidated.

The [stored group gain helper](interpretability/stored_group_gain.py) assigns only declared dense outgoing columns and restores the original matrices on exit, including exceptions. Its fixed factor is not fitted to outcomes. Random-module fixtures are separate from pretrained behavioral evidence.

## Research and publication boundaries

There are zero optimizer updates in the current study. No fine-tuning, LoRA, reinforcement learning, fitted steering vector or training of added neurons is used. Gradients may measure influence and are never applied as updates. Earlier training work remains closed historical groundwork.

Necessity, counterfactual rescue, matched controls, executable coding and retention must support any functional interpretation. Rescue of an injected error and transfer of an opposite instruction do not improve the unmodified model. Known steering, sparse circuits, neuron-basis methods, contextual binding and function-preserving growth are prior work; this project makes no general-method novelty claim.

This public repository contains reviewed documentation, generic code, aggregate records and validated compact edit bundles. Raw tasks, answers, notebooks, predictions, activations, historical trained adapters, credentials, private history and the unpublished paper remain private. Each public update is reviewed and its remote tree verified. No competition submission has been made through this repository.

## License

Original code and documentation use Apache 2.0. Model bundles carry their source attribution, modification notices and applicable licenses. Base-model and third-party dependency terms remain applicable.
