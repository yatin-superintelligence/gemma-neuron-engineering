# Gemma Neuron Engineering: Interpretability and Control

A no-training study of native neurons, small groups and direct model edits in Gemma 4 E2B. We test whether a proposed function survives controlled interventions and whether a fixed edit helps executable coding.

**The tested physical edits are reproducible. Individual neuron meanings and a confirmed coding improvement remain unestablished.**

N24 finds a controlled comparator-score contribution: selected64 opposite-policy donors shift the logit margin by +1.574 on average, versus +0.025 and -0.051 for matched random groups. Same-policy donors average +0.022. All 578 trajectories and guards pass, with zero updates. Every nominated operator margin remains on the original side, so the prior program-level null remains unchanged. See the [complete result and limits](results/n24_finite_summary.json).

N25 now supplies an assessable runtime-edit null. The earlier N21 result remains preserved: all 160 records pass audit. The selected edit matches the original at 12/24 coding functions; both random controls score 14/24. Ten capped source-format failures exceed the predeclared baseline limit, making the planned benefit interpretation unassessable. There is no selected coding gain.

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

## Complete coding evaluations

N21 applies the same prescribed global outgoing-column factor of two to the existing selected64 group and each fixed random64 group. Original and same-scope identity are separate arms. The edit has no answer-derived trigger, fitted strength or new coordinate selection. All 24 coding functions remain in gains/losses, with eight family-level tables and eight retention inputs.

The private GPU package contains inference code and prompt records. Hidden oracles, stress inputs and host grading code are excluded. Host gates cover paired executable oracles, strict output parsing, bounded isolated execution, exact identity and restoration, first-prefill cache preservation and native generation counters. All actual-checkpoint identity, cache, restoration and declared parameter guards pass with zero optimizer updates. The selected edit has zero coding gains or losses against the original. Every arm scores 6/8 under the frozen retention checks. Source-format failures coincide with capped unclosed code fences; no outputs are repaired or regraded. Random-arm gains remain controls, not selected winners.

Retention scores follow the frozen checks. One rotation probe checks returned values without testing fresh-list identity; this limit is recorded before output access.

See the [complete N21 result](results/n21_utility_summary.json), [all five weight recipes](models/n21/recipe.json) and [prospective protocol](methods/n21_prospective_protocol.json). The [completed stage/site control](results/n22_stage_site_summary.json) separates stage from coverage on the same inherited cohort. Both missing cells transfer 0/13; only the inherited all-35-layer current-decision cell transfers 13/13. Same-rule preservation is 13/13 throughout. All 273 records and mandatory guards pass. The [N23 original-only task gate](results/n23_task_gate_summary.json) stopped before any neuron edit. Full role requirements pass 16/32, no complete counterbalance block qualifies, and retention passes 5/8. Diagnostic denotation passes 28/32 but does not replace the required operand-order checks. All 40 records and mandatory guards pass audit; no prompts, caps or grades were changed. The [prospective protocol](methods/n23_baseline_protocol.json) remains preserved.

The [finite-score study](methods/n24_finite_score_protocol.json) is complete. [Every group and operation](docs/figures/n24_summary.png), [all 24 runtime recipes](models/n24/recipe.json), the [metric helper](interpretability/decision_scores.py) and [plot source](scripts/plot_n24_summary.py) are preserved. This is a bounded score contribution on reused development, with no individual semantic meaning or coding gain. 

The [runtime syntax controller](methods/n25_runtime_protocol.json) is complete and audited on a usable error-bearing baseline. All five conditions score **23/32 coding and 8/8 retention**, with **zero gains or losses**. All 200 records and mandatory guards pass. The 160 new token sequences equal the original responses, although every gain arm changes 54 immediate logit-vector hashes. The fixed utility criterion fails. See [all outcomes and diagnostics](results/n25_runtime_summary.json), the [complete figure](docs/figures/n25_summary.png) and [all five recipes](models/n25/recipe.json).

N25 uses only already-generated Python for its fixed trigger, with no baseline answers or hidden oracles in model inputs. The [original-only stage](results/n25_baseline_summary.json) passes every prospective usability requirement. The [host record](results/n25_host_summary.json) preserves the zero-forward parser failure and compatibility repair. Original grades, source failures and all dependent-family results remain fixed. The [example runner](examples/run_n25_policy.py) accepts your own prompt file; independent clean-package pretrained replay is not claimed.

The [global suppression study](results/n26_global_summary.json) is complete. All 320 records and mandatory guards pass. Its two primary screens fail: selected suppression matches random A on the same two gained cases, and no group loses an originally correct case under suppression. All retention checks stay correct. These are reused development cases, and the four variants in each family are dependent.

| Policy | Coding /32 | Gains / losses vs original | Retention /8 |
| --- | ---: | ---: | ---: |
| Original carry | 23 | 0 / 0 | 8 |
| Guarded identity reference | 23 | 0 / 0 | 8 |
| Selected 64, factor 0 | 25 | 2 / 0 | 8 |
| Random A64, factor 0 | 25 | 2 / 0 | 8 |
| Random B64, factor 0 | 24 | 1 / 0 | 8 |
| Selected 64, factor 2 | 22 | 0 / 1 | 8 |
| Random A64, factor 2 | 23 | 1 / 1 | 8 |
| Random B64, factor 2 | 22 | 0 / 1 | 8 |

See the [complete grid](docs/figures/n26_summary.png), [all eight recipes](models/n26/recipe.json) and [caller-prompt example](examples/run_n26_policy.py). The fixed global hypothesis is closed. No winning dose replaces the failed primary. The source, cache, identity, restoration and declared weight-version checks pass with zero optimizer updates. Existing exact N21 columns support these recipes without new tensor blobs or base duplication. Independent clean-package pretrained replay is not claimed.

A [generic host-only checker probe](results/return_identity_host_summary.json), completed before N26 output access, finds an [untested return-list identity property](methods/n26_return_identity_limit.json). The frozen scores check values, types and input preservation. They do not guarantee a newly allocated returned list. All original grades and cases remain unchanged; future confirmation must check the full contract. A [future-only macOS checker](evaluation/contract_execution_v2.py) now enforces fresh top-level result identity before serialization. Its [thirteen fixed host cases](results/future_return_contract_host.json) pass, including aliases of nested inputs and input mutation. It is not applied to N25 or N26.

The [contextual-binding study](results/n27_baseline_summary.json) stops before every edit. All forty records and mandatory guards pass: 21/32 coding, 8/8 retention and no complete counterbalanced block. Every role source satisfies the required operand order, but eleven compute the wrong relation. The [prospective protocol](methods/n27_prospective_protocol.json) and [host preparation](results/n27_preparation.json) remain preserved. An audit interpreter import failure occurred before grading; the unchanged frozen auditor passes under the validated research Python. No prompts, caps or grades change.

The [direct-relation study](results/n28_summary.json) is complete. Its original baseline passes 16/16 coding and 8/8 retention, but the donor-stage primary fails: **0/8 complete cross-order pairs**. Same-order opposite donors transfer 16/16 programs and same-relation donors preserve 16/16. Both reversed-order role arms have zero source-valid programs. All 168 records and mandatory guards pass, with retention 8/8 throughout and no caps. Saved-output comparison finds that all 64 foreign role continuations are token-exact to their actual donor suffixes. Invalid programs keep failed grades; this observation does not establish abstract relation control.

See the [complete grid](docs/figures/n28_summary.png), [all seven inference recipes](models/n28/recipe.json), [prospective protocol](methods/n28_prospective_protocol.json) and [interpretation limits](methods/n28_interpretation.json). No new weight tensors or full bases are stored. The [instruction-state control](results/n29_summary.json) is also complete. All 120 records and mandatory guards pass. Opposite donors transfer 0/16 programs and 0/8 complete operand-order pairs; every condition retains 16/16 original correctness and 8/8 retention. Both foreign arms change all 24 first-logit and whole-cache witnesses, but every response remains token-exact to its original. The [five recipes](models/n29/recipe.json), [prospective protocol](methods/n29_prospective_protocol.json) and [interpretation limits](methods/n29_interpretation.json) preserve the complete null. This closes the panel anchor search. N29 remains closed.

The [fixed native-unit study](methods/n30_prospective_protocol.json) is complete. All **399 records and mandatory guards pass**, but its finite-bias conjunction fails **one of 27 components**: a conditional-cut cell ties control B. The other 26 components pass; they do not replace the failed primary. The target has a small measured score effect, while all three units produce zero opposite-program transfers, zero clean ablation losses and zero conditional-transfer losses. All new zero-arm retention checks pass 8/8. No coding gain or individual semantic role is established.

The target remains L34 C3271, with strict controls C9170 and C4059. The original duplicate control draw and separately frozen distinct draw are preserved. See the [complete results](results/n30_summary.json), [all nineteen recipes](models/n30/recipe.json), [full figure](docs/figures/n30_summary.png), [source freeze](methods/n30_execution_freeze.json), [preparation record](results/n30_preparation.json) and [current-unit helper](interpretability/final_unit.py). No new weight blobs are needed for these runtime edits. Independent clean-package pretrained replay is not claimed.

A [conditional fresh coding intent](methods/n31_conditional_intent.json) fixed both permanent column factors, zero and two, before N30 outputs. Its parent prerequisite fails, so it remains unexecuted. There is no N31 dataset or model result.

The [fresh group study](methods/n32_prospective_protocol.json) stopped at its original-only baseline gate. All 40 outputs and audit guards pass, but only 19/32 role programs satisfy the frozen contract: predicate 14/16 and filter 5/16. Retention is 8/8, with no caps. Ten sources violate names or operand order; three source-valid programs are wrong. Only two of eight blocks qualify, and none is selected. B and C never run. See the [complete outcomes](results/n32_summary.json) and [all-block figure](docs/figures/n32_summary.png). All grades remain fixed; no neuron intervention or group effect was tested.

The first N32 attempt stopped before inference because its tokenization metadata used the wrong schema. The preserved failure has zero forwards or predictions. A [versioned supplement](methods/n32_startup_metadata_repair_v2.json) verifies all 40 unchanged input tensors and supplies the complete metadata expected by the same inference code. Both source freezes and the path-only auditor amendment remain traceable.

The next [scaffolded development study](methods/n33_prospective_protocol.json) is frozen before implementation. It keeps the same groups and crosses relation with operand order at an input position fixed before any answer. Supplied unfinished syntax makes this conditional completion. It requires original/full-site controls, new-input magnitude adequacy and a separate executable-binding test. This format is chosen after earlier feedback and is not fresh confirmation. New private data and the actual encoder pass 211 host fixture outcomes and 11,291 independent-reference comparisons, with zero model forwards. Twelve random-weight Gemma policy fixtures in FP32/BF16 also pass, with independent future-token checks for guarded references and eight corrupted-state refusals. These 48 forwards use no pretrained weights. Runners, auditors and the execution freeze remain pending. No N33 pretrained result exists, and no GPU job is active.

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
