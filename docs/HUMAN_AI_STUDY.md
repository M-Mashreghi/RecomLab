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
