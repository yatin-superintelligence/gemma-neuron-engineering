# Gemma Neuron Engineering: Interpretability and Control

A no-training study of native neurons, small groups and direct model edits in Gemma 4 E2B. We test whether a proposed function survives controlled interventions and whether a fixed edit helps executable coding.

**The tested physical edits are reproducible. Individual neuron meanings and a confirmed coding improvement remain unestablished.**

N24 finds a controlled comparator-score contribution: selected64 opposite-policy donors shift the logit margin by +1.574 on average, versus +0.025 and -0.051 for matched random groups. Same-policy donors average +0.022. All 578 trajectories and guards pass, with zero updates. Every nominated operator margin remains on the original side, so the prior program-level null remains unchanged. See the [complete result and limits](results/n24_finite_summary.json).

N21 is complete: all 160 records pass audit. The selected edit matches the original at 12/24 coding functions; both random controls score 14/24. Ten capped source-format failures exceed the predeclared baseline limit, making the planned benefit interpretation unassessable. There is no selected coding gain.

## Read the research

- [Goal & Intent](docs/GOAL_AND_INTENT.html): scientific objectives, mechanisms, controls and decision rules.
- [Research Logbook](docs/RESEARCH_LOG.html): sequential experiments, failures, amendments, results and next decisions.
- [Question register](docs/research_questions.json): the evidence behind each research question.
- [Saved edits](models/README.md): compact tensors, recipes and reconstruction limits.
- [Model identity](docs/model_identity.json) and [release inventory](PUBLIC_RELEASE_MANIFEST.json).

Clone this repository and open `docs/index.html` to view the HTML documents. GitHub's file viewer displays their source.

## What the evidence supports

| Finding | Evidence | Boundary |
| --- | --- | --- |
| Physical channel edits work on the tested model | N02 audits 460 measurements across the pinned unquantized BF16 and same-source NF4 representations. Deletion agrees with zeroing; doubled outgoing weights agree with activation gain two. R01 reconstructs six edits per representation using the published loader. | One selected layer and a fixed development panel. BF16 split cloning has bounded nonzero logit differences. No speed or coding benefit was tested. |
| Whole native sites can transfer a contextual coding rule | N14 transfers 14/16 programs with layers 5 to 9. N16 tests all 31 subsets and selects layers 5, 6 and 9 under a fixed smallest-site rule. At the later N19 comparator decision, all-layer opposite donors transfer 13/13 programs and same-rule donors preserve 13/13. | Reused development inputs, actual donor states and wording dependence. These are conditional sufficiency results, not individual meanings, native necessity or improved coding. N18 and N19 also differ in layers and eligible cases, so their contrast does not isolate position. |
| Small-group hypotheses have not passed their role screens | N11 fails its predefined rescue and coverage criterion. N13's rare-unit correlation produces no correctness change. N17's selected 96 channels and matched groups produce no transfer or necessity effect. N20 audits all 840 records; selected64, both random64 controls and all nested16 groups have zero transfer, clean-zero loss and conditional-transfer loss. | Full-site controls still work. Sparse nulls do not imply that neurons have no function. No winning dose or subgroup replaces a failed primary test. |
| Conversational wording can change without a clear trait shift | B01 audits 192 records and rates 160 chat responses before unblinding. Fixed edits preserve its bounded uncertainty, false-premise, remembered-name and coding checks. | Exploratory agent ratings, frequent token caps and random controls that are not activity matched. No validated personality change. |
| Channel reversal does not reverse knowledge | B02 retains all 64 generations and 128 separately derived answer readings. Original and coordinated channel reversal score 4/4 coding with normal input and raw reading; all other model/input/reading cells score 0/4. | Coordinated reversal is a known ideal symmetry and is not bit-identical in this run. Reversed text does not recover incoherent outputs. No semantic inversion or new capability. |

Every tested condition and scientific null remains in the logbook. The [N19 grid](results/n19_current_decision_summary.json), [N20 grid](results/n20_summary.json), [B01 assessment](results/b01_chat_summary.json) and [B02 reversal grid](results/b02_reversal_summary.json) provide complete aggregates.

## Fresh coding result

N21 applies the same prescribed global outgoing-column factor of two to the existing selected64 group and each fixed random64 group. Original and same-scope identity are separate arms. The edit has no answer-derived trigger, fitted strength or new coordinate selection. All 24 coding functions remain in gains/losses, with eight family-level tables and eight retention inputs.

The private GPU package contains inference code and prompt records. Hidden oracles, stress inputs and host grading code are excluded. Host gates cover paired executable oracles, strict output parsing, bounded isolated execution, exact identity and restoration, first-prefill cache preservation and native generation counters. All actual-checkpoint identity, cache, restoration and declared parameter guards pass with zero optimizer updates. The selected edit has zero coding gains or losses against the original. Every arm scores 6/8 under the frozen retention checks. Source-format failures coincide with capped unclosed code fences; no outputs are repaired or regraded. Random-arm gains remain controls, not selected winners.

Retention scores follow the frozen checks. One rotation probe checks returned values without testing fresh-list identity; this limit is recorded before output access.

See the [complete N21 result](results/n21_utility_summary.json), [all five weight recipes](models/n21/recipe.json) and [prospective protocol](methods/n21_prospective_protocol.json). The [completed stage/site control](results/n22_stage_site_summary.json) separates stage from coverage on the same inherited cohort. Both missing cells transfer 0/13; only the inherited all-35-layer current-decision cell transfers 13/13. Same-rule preservation is 13/13 throughout. All 273 records and mandatory guards pass. The [N23 original-only task gate](results/n23_task_gate_summary.json) stopped before any neuron edit. Full role requirements pass 16/32, no complete counterbalance block qualifies, and retention passes 5/8. Diagnostic denotation passes 28/32 but does not replace the required operand-order checks. All 40 records and mandatory guards pass audit; no prompts, caps or grades were changed. The [prospective protocol](methods/n23_baseline_protocol.json) remains preserved.

The [finite-score study](methods/n24_finite_score_protocol.json) is complete. [Every group and operation](docs/figures/n24_summary.png), [all 24 runtime recipes](models/n24/recipe.json), the [metric helper](interpretability/decision_scores.py) and [plot source](scripts/plot_n24_summary.py) are preserved. This is a bounded score contribution on reused development, with no individual semantic meaning or coding gain. The [next controller intent](methods/n25_runtime_controller_intent.json) considers a fixed runtime syntax trigger without a known comparator position or baseline answer. It is not implemented or launched.

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

Follow [the model bundle instructions](models/README.md) for the validated loader and exact edit names. Base checkpoints are referenced by immutable revision rather than duplicated. Donor-based recipes require caller-supplied contexts; private donor states are excluded. R01 verifies the N02 reconstruction path. Other recipes state their replay limits explicitly. Full transformer-layer insertion remains unvalidated.

The [stored group gain helper](interpretability/stored_group_gain.py) assigns only declared dense outgoing columns and restores the original matrices on exit, including exceptions. Its fixed factor is not fitted to outcomes. Random-module fixtures are separate from pretrained behavioral evidence.

## Research and publication boundaries

There are zero optimizer updates in the current study. No fine-tuning, LoRA, reinforcement learning, fitted steering vector or training of added neurons is used. Gradients may measure influence and are never applied as updates. Earlier training work remains closed historical groundwork.

Necessity, counterfactual rescue, matched controls, executable coding and retention must support any functional interpretation. Rescue of an injected error and transfer of an opposite instruction do not improve the unmodified model. Known steering, sparse circuits, neuron-basis methods, contextual binding and function-preserving growth are prior work; this project makes no general-method novelty claim.

This public repository contains reviewed documentation, generic code, aggregate records and validated compact edit bundles. Raw tasks, answers, notebooks, predictions, activations, historical trained adapters, credentials, private history and the unpublished paper remain private. Each public update is reviewed and its remote tree verified. No competition submission has been made through this repository.

## License

Original code and documentation use Apache 2.0. Model bundles carry their source attribution, modification notices and applicable licenses. Base-model and third-party dependency terms remain applicable.
