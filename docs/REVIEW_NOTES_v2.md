# RecomLab v2 Review Notes

The first planning package had a solid retrieval/ranking backbone but several methodological and workflow gaps. v2 intentionally fixes them rather than simply adding more models.

## Added / Strengthened

- impression/exposure logging contract
- explicit-vs-implicit relevance semantics
- training negative sampling as a dedicated task
- point-in-time feature construction
- candidate fusion + eligibility filters
- full-catalog vs sampled evaluation separation
- user-level uncertainty / paired significance tests
- optional exposure-aware off-policy evaluation only when valid propensities exist
- multi-metric champion selection and promotion guardrails
- model registry + champion/challenger workflow
- drift and train-serving skew monitoring
- contextualized vs generic LLM explanation conditions
- explanation fidelity as separate from persuasion/trust
- human-study randomization, power planning, and ethics boundary

## Removed / Avoided

- A mandatory external vector database: unnecessary for a portfolio-scale v1.
- A mandatory distributed queue / microservice architecture: adds operational complexity without research value at this stage.
- Automatic model promotion after retraining: unsafe and methodologically weak.
- Treating sampled-negative metrics as equivalent to full-catalog evaluation.
- Treating LLM explanations as evidence of recommender correctness.

## Post-v1 Backlog (not required)

- session/sequential recommendation
- contextual bandits / online learning
- slate optimization
- graph neural recommenders
- multimodal item encoders
- large-scale ANN benchmarking
- causal recommendation / advanced counterfactual estimators
- federated recommendation

These are valuable research directions but would dilute the v1 portfolio story if added before the core pipeline is complete.
