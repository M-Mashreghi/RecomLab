# RecomLab Human-AI Study Extension

This phase is optional and is intended to make RecomLab useful for Human-AI / recommender-explanation research.

## Example Research Questions

- Do contextualized LLM explanations increase recommendation acceptance compared with generic explanations?
- Does explanation fidelity predict trust independently of recommendation quality?
- Do effects differ by prior user engagement or familiarity with AI/recommenders?

## Suggested Conditions

1. Recommendation only
2. Template explanation
3. Generic LLM explanation
4. Contextualized LLM explanation grounded in past interactions

## Outcomes

Keep constructs separate:
- recommendation acceptance / intention
- trust
- perceived usefulness
- satisfaction
- transparency/understanding if measured
- decision time
- explanation fidelity (system-side)

## Fidelity Principle

A persuasive explanation is not necessarily a faithful explanation. Every LLM condition should be checked against the recommender evidence payload.

## Study Governance

The repository may implement randomization, study configuration, survey schemas, synthetic fixtures, and analysis code. It must **not** assume authorization to recruit human participants. Real data collection should follow the relevant institutional ethics/privacy process.

## Synthetic Data

Synthetic responses are for software validation only and must be labeled simulated in every output/report.

## Grounding Evidence (`recomlab.explanations.evidence`, schema `evidence-1`)

`build_evidence(user_id, item_id, history, items, sources, contributions, as_of, history_window)`
produces an `EvidencePayload` for one recommended item:

| Evidence id | Kind | Role | Source of truth |
|---|---|---|---|
| `item:<item>:title`, `item:<item>:categories` | `item_fact` | presentation | item catalog |
| `src:<source>` (+ `seed_items`) | `retrieval` | ranking | fused candidate provenance |
| `feat:<feature>` | `ranker_feature` | ranking | ranker contributions |
| `hist:<event_id>` | `history` | ranking if a retrieval seed item, else presentation | stored interactions before `as_of` |

- History is bounded to the `history_window` most recent events (default 5); seed items used
  for ranking are always included because they are ranking evidence.
- Missing facts (no title, no categories, no history, no retrieval source, no contributions)
  are listed in `unavailable`; nothing is synthesized.
- `validate_payload` checks that every history entry is a stored interaction of that user for
  the stated item and that item facts equal the catalog.

## Explanation Conditions (`recomlab.explanations.conditions`)

| Condition id | Strategy | Evidence scope |
|---|---|---|
| `none-v1` | no explanation | none |
| `template-v1` | deterministic template, no LLM dependency | all payload evidence |
| `generic_llm-v1` | LLM | item facts + non-personal ranker features (`popularity`, `recency`) |
| `contextual_llm-v1` | LLM | item-level evidence + bounded user history, retrieval provenance, all ranker features |

`restrict_evidence` enforces each scope before generation. `ExplanationRouter.explain(payload,
condition_id)` dispatches to the strategy; LLM conditions without a configured generator fall
back to the template (recorded in `fallback_from`). The template cites the evidence ids it
uses and abstains (`text = None`) when no usable evidence exists. Conditions never touch
ranking.

## Structured LLM Generator (`recomlab.explanations.llm`, prompt `expl-prompt-v1`)

- Providers implement `generate(system, prompt, schema) -> str`. `MockProvider` is used in all
  tests; `AnthropicProvider` (optional extra `.[llm]`, e.g. model `claude-opus-5-5`) uses the
  Messages API with structured outputs (`output_config.format` JSON schema) and low effort.
  No live API is called in CI.
- Output schema: `explanation`, `cited_evidence_ids`, `abstain`, `uncertainty`
  (`low|medium|high`). `parse_output` rejects malformed JSON, wrong types, citations not in the
  condition-scoped payload, and non-abstaining answers without text or citations.
- Retries are bounded (`max_attempts`) and only for transient provider errors and schema
  failures; non-transient errors stop immediately. The final fallback is the template
  (default) or no explanation — never an LLM ranking.
- Every explanation records `prompt_version` and `model`; generation and fallbacks are logged
  with the prompt/model version. The explanation path runs after ranking and never alters it
  (integration-tested with a failing provider).

## Fidelity Validation (`recomlab.explanations.fidelity`)

- `extract_claims` finds deterministic claims in explanation text: cited evidence ids
  (`citation`), catalog titles (`item`), catalog categories (`category`), "you
  watched/rated/liked/bought…" statements (`history`), and popularity/trending statements
  (`popularity`). Each is supported only if it traces to the condition's evidence payload.
- `validate_explanation(..., threshold)` reports the unsupported-claim rate and rejects
  explanations above the threshold (default 0: any unsupported claim rejects).
- `feature_ablation` is the perturbation check for additive rankers (heuristic, LTR): it
  removes each cited feature's contribution, reports score delta and rank before/after, and
  flags cited features with no effect (which also rejects the explanation).
- `FidelityLog` stores results as `record_type = "fidelity"` JSONL, separate from any
  perception outcome; mixing record types is an error.

## Study Protocol & Randomization (`recomlab.research.study`)

- `StudyConfig` (versioned by `config_version`, a content hash) holds research questions,
  hypotheses (outcome, treatment vs control condition, direction), explanation conditions,
  primary/secondary outcomes, exclusion rules (attention check, minimum completion time),
  block size, seed, `data_mode`, and an `EthicsRecord` (approval status/reference, consent
  version, retention days).
- **Ethics boundary**: `data_mode="real"` is rejected unless an approved ethics/IRB decision
  with a reference, a consent version and a retention period are recorded. Even then the
  code only validates configuration; recruiting participants is outside the engineering
  workflow.
- `assign_conditions`: permuted-block randomization in enrollment order, deterministic for
  the seed; every complete block is perfectly balanced.
- `sample_size_per_group(d, alpha, power, n_comparisons)`: two-sided two-sample normal
  approximation with Bonferroni adjustment (d = 0.5 → 63 per group).
- `simulate_responses` produces **simulated** records only (`simulated=True`, `SIM-` ids);
  `StudyDataStore` refuses to mix simulated and real records or records from another config
  version, and reports exclusions with reasons.
