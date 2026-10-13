# Architecture Rules

Keep ingestion, dataset protocol, features, retrieval, ranking, evaluation, registry, serving, and explanation modules separate. Retrieval optimizes candidate recall; ranking scores only candidates. UI/API layers do not train models or compute research metrics.
