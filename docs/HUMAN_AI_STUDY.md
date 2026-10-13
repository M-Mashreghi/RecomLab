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
