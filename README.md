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

The [N40 complete-function utility study](results/n40_summary.json) is complete and audited. Original coding is 24/32; selected deletion is 26/32 with two gains and no losses; random controls are 25/32 and 24/32. Both selected repairs occur in one reused task family, with one repair also seen in each random. The output-reduction primary fails: selected mean token ratio is 1.027 across all 32 cases. The simple instruction reaches 22/32 while shortening replies. All conditions retain 7/16 strict retention cases, and all 192 new replies reach native EOS without hitting the old endpoint. This is an exploratory accuracy lead, not confirmed useful coding improvement or a semantic neuron role. The [all-condition figure](docs/figures/n40_summary.png), [exact compact model recipes](models/n40/recipe.json) and all failures remain available.

The [current inference policy](methods/inference_budget_policy.json) records native EOS stopping and actual resource/context interruptions. The [native-EOS engine](interpretability/native_eos_generation.py) has no response-token ceiling, streams exact decisions and resumes only after matching its own saved history. Its random Gemma FP32/BF16 checks pass, and its isolated public imports require no model, pretrained-load or tokenizer calls. Callers must bind the actual model, prompt and edit, persist streamed tokens, and provide measured resource checks. [Parallel-compute review](methods/compute_parallelism.json) confirms the allocation is two separate 16 GB T4 GPUs. The completed N40 worker used GPU 0. The private two-device random-model probe passes on both actual T4 devices, with overlapping workers and all eight cross-device logit/cache comparisons exact. It ran alongside the utility study, confirming notebook concurrency. The distinct [full-replica prerequisite](results/full_replica_readiness_summary.json) now passes: two complete BF16 checkpoints reside concurrently, all 288 native decision/logit/cache comparisons and 96 first caches match, with peak 9.333 GB allocated per GPU. No new responses or grades. Two full replicas also complete concurrent native-EOS generation in N41. The edited workers overlap for 4570.71 seconds with peak 9.333 GB allocated per GPU. No speed multiplier or full-context capacity is inferred from these measurements.

The [fresh N41 comparison](results/n41_summary.json) is complete, audited and closed. The same selected deletion falls from 55/64 original coding to 51/64, with no repairs and four regressions. Both random controls score 53/64; the simple instruction scores 48/64. All 320 coding sources are valid and every coding failure is an executable wrong answer. The fresh primary fails, closing this fixed selected-group utility lead. The earlier two N40 development repairs did not establish a generalizable benefit.

| Fixed condition | Coding /64 | Coding gains / losses | Retention /32 | Retention gains / losses |
| --- | --- | --- | --- | --- |
| Original | 55 | 0 / 0 | 17 | 0 / 0 |
| Selected zero | 51 | 0 / 4 | 17 | 0 / 0 |
| Random A zero | 53 | 1 / 3 | 16 | 0 / 1 |
| Random B zero | 53 | 2 / 4 | 16 | 0 / 1 |
| Instruction only | 48 | 4 / 11 | 17 | 0 / 0 |

All 480 comparison records remain: 352 new replies, 96 original carries and 32 exact instruction-control retention reuses. Across both phases there are 448 unique generations. [Every new reply reaches native EOS](results/n41_termination.json) without an arbitrary response-token ceiling; the longest is 2134 tokens. All identity, physical/activation equivalence, state, cache, restoration and counter guards pass. These are dependent families and bindings, with weak baseline retention, not broad coding or knowledge claims. The [all-condition figure](docs/figures/n41_summary.png), [interpretation](results/n41_interpretation.json) and [exact recipes](models/n41/recipe.json) retain every condition. The recipes share the existing compact N40 tensors and add no weight bytes.

The [N42 source-structure review](results/n42_summary.json) is complete. It describes all 320 saved coding sources and 256 original/condition pairs without generating, executing or grading a program. The protocol froze after N41 outcomes, so these are posthoc descriptions rather than fresh confirmation.

| Condition | Identical tokens | Changed tokens, same extracted source | Changed source, same AST | Changed AST |
| --- | --- | --- | --- | --- |
| Selected zero | 37 | 0 | 6 | 21 |
| Random A zero | 36 | 0 | 4 | 24 |
| Random B zero | 40 | 0 | 5 | 19 |
| Instruction only | 5 | 0 | 5 | 54 |

The four selected regressions involve changed syntax trees. Of its other structural changes, twelve remain correct and five remain wrong. AST difference does not prove a behavioral difference. These edits affect actual returned program structure as well as surface spelling, with similar structural-change counts under the matched randoms. Every saved grade and failed primary remains unchanged. The [generic source-comparison helper](evaluation/program_structure.py) preserves names, literals, docstrings and operators; its independent literal checks and malformed-input refusals pass.

The [N43 normalization measurement](methods/n43_normalization_design.json) is now running privately with a frozen 220-file source/data/parent record. It replays all 96 original native histories and crosses native versus removed-column raw values with their actual reciprocal-RMS factors at every one of nine fixed late sites. The same selected group and both random controls remain. All four cells, exact native/physical diagonals, same-input caches and restorations are mandatory. Two full replicas measure disjoint cases; no new replies or scores are produced. The [random-model host gates](results/n43_host_preparation.json) pass after a preserved scalar-cache failure and minimal repair. Actual pretrained measurements remain pending the terminal audit. This is a layer-local mechanism control with [direct normalization priors](methods/n43_source_and_prior_review.json), not a replacement utility test or an established explanation of the fresh coding regressions.

A separate [all-condition figure plan](methods/n43_figure_plan.json) was registered after launch and before output access. The [saved-aggregate renderer](scripts/plot_n43_summary_v2.py) and [complete-grid validator](evaluation/normalization_figure_contract.py) pass synthetic software/layout checks. They require a complete passing audit, retain every group/cell/site/family and display undefined cosine separately from zero. The first legend overlap and layout-only repair are preserved. Synthetic previews were deleted after visual review and hashing. No actual N43 figure or result has been produced.

The [pre-output evidence map](methods/n43_evidence_decision_map.json) states thirteen claim boundaries. It distinguishes numerical zero from byte identity, mean effect norms from a shared direction, and scoped restoration witnesses from independent full-checkpoint replay. It also confirms that preserved historical cap definitions are uninvoked by the active measurement worker. No audit rule or scientific threshold changes.

## Offline surgery workbench

The [saved-history comparison helper](evaluation/common_history_trace.py) compares whole-cache hashes while consumed inputs match. It includes the first differing next-token prediction and excludes decisions after that differing token is consumed. Its literal software guards pass. The separately registered [N41 path prediction](results/n41_cache_path_summary.json) also passes: all 29,672 whole-cache comparisons across 288 original/group pairs are exact within this same-input range. This bounded consequence of the late edit site is not quantitative mediation, semantic meaning or a measured runtime speedup.

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
