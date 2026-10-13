# RecomLab Data Contract

Schema version: **1.0** (`recomlab.domain.schemas.SCHEMA_VERSION`).

## Canonical Tables

Identifiers (`*_id`) are pseudonymous strings matching `^[A-Za-z0-9_.:\-]{1,64}$`.

### users (`recomlab.domain.schemas.User`)

| Field | Required | Notes |
|---|---|---|
| `user_id` | yes | pseudonymous stable identifier |
| other columns | no | non-sensitive research/demo attributes (`attributes` map) |

Attribute keys that look like direct identifiers (`name`, `email`, `phone`, `address`,
`ip`, `device_id`, ...) are rejected with code `pii_attribute`.

### items (`Item`)

| Field | Required | Notes |
|---|---|---|
| `item_id` | yes | |
| `title` | no | only when license/privacy permits |
| `categories` | no | `|`-separated in CSV; genres/tags |
| `available` | no | availability metadata, default `true` |
| other columns | no | `attributes` map |

### interactions (`Interaction`)

| Field | Required | Notes |
|---|---|---|
| `event_id` | yes | duplicate key |
| `user_id`, `item_id` | yes | |
| `event_type` | yes | see below |
| `value` | for `rating` | finite number; meaning depends on `event_type` |
| `event_time` | yes | ISO-8601 with timezone; normalized to UTC |
| `session_id` | no | |

Event types and feedback semantics. Explicit and implicit feedback are never conflated:
converting either into relevance labels is a separate, versioned label policy (Task 2.2).

| `event_type` | Kind | `value` |
|---|---|---|
| `rating` | explicit | required, range `[0, 10]` |
| `watch` | implicit | optional fraction watched, `[0, 1]` |
| `purchase` | implicit | optional quantity, `>= 0` |
| `click`, `view`, `like`, `add_to_cart` | implicit | optional, unused by default |

### Validation errors

Every violation raises `SchemaValidationError` with a stable `code` and `field_name`:
`missing_id`, `invalid_id`, `invalid_event_type`, `missing_timestamp`, `invalid_timestamp`,
`naive_timestamp`, `missing_value`, `invalid_value`, `value_out_of_range`, `pii_attribute`.

### Timestamp policy

- Timestamps must be timezone-aware; naive timestamps are rejected (`naive_timestamp`).
- All timestamps are converted to UTC on load.
- Canonical event order is `(event_time, event_id)` (`schemas.sort_key`), which makes ties
  deterministic.

### Duplicate-event policy

- `event_id` is the duplicate key.
- The first occurrence is canonical. Later byte-identical records are **exact duplicates**
  and are dropped with a reason code during cleaning (Task 2.1).
- Later records with the same `event_id` but different content are **conflicting
  duplicates**; they are contract violations and are reported, never silently merged.
- `schemas.find_duplicate_events` returns `(exact, conflicting)` lists.

## Exposure Contract

Exposure schema version: **1.0** (`recomlab.domain.exposure.EXPOSURE_SCHEMA_VERSION`).

### impressions (`Impression`)

| Field | Required | Notes |
|---|---|---|
| `impression_id` | yes | idempotency key |
| `request_id` | yes | one recommendation request / slate |
| `user_id`, `item_id` | yes | |
| `position` | yes | 1-based rank shown |
| `model_id` | yes | model/bundle that produced the slate |
| `shown_at` | yes | timezone-aware, UTC |
| `propensity` | no | logging-policy probability in `(0, 1]`; **never fabricated** |
| `session_id` | no | |

### feedback_events (`FeedbackEvent`)

| Field | Required | Notes |
|---|---|---|
| `feedback_id` | yes | |
| `user_id`, `item_id` | yes | |
| `feedback_type` | yes | `click`, `like`, `dislike`, `watch`, `purchase`, `dismiss`, `rating` |
| `event_time` | yes | timezone-aware, UTC |
| `value` | no | numeric only |
| `impression_id` | no | link to the exposure when known |

Free-text and identifier fields (`comment`, `text`, `free_text`, `message`, `review`,
`email`, `name`, `ip_address`) are rejected with `free_text_not_allowed`.

### Linkage (`exposure.link_feedback`)

- `linked`: `impression_id` given and consistent (same user/item, feedback not before exposure).
- `invalid`: `impression_id` given but unknown (`unknown_impression`), for another user/item
  (`user_item_mismatch`), or earlier than the exposure (`feedback_before_exposure`).
- `inferred`: no `impression_id`; matched to the latest prior impression of the same user/item
  only when the caller passes an explicit `infer_window` (off by default).
- `unlinked`: no matching exposure.

### Datasets without exposure logs

`exposure.assess_exposure_log` classifies a log as `unavailable` (no impressions),
`impressions_only` (some propensities missing), or `propensity_available`. Only the last
allows exposure-aware / off-policy evaluation. Missing propensities are counted, never imputed.

## Data Semantics

- Raw feedback is preserved.
- Binary/graded relevance is a derived, versioned policy.
- Duplicate handling is explicit.
- Event timestamps are required for time-aware experiments.
- Missing exposure logs mean exposure-aware evaluation is unavailable, not approximated silently.

## Fixtures

`data/fixtures/tiny/` is a hand-designed, **simulated** dataset used by CI (see its README).

## Privacy

Use pseudonymous identifiers. Do not store names, emails, device identifiers, or raw free-text unless a specific research task requires them and the applicable privacy/ethics process permits it.

## Dataset Manifest

Every dataset snapshot records source, version, checksum, schema version, retrieval date, license/terms note, cleaning policy, and label policy.

## Dataset Registry & Data Cards

- Adapters (`recomlab.datasets.adapters`) turn an immutable raw snapshot directory into a
  `CanonicalDataset`: `FixtureAdapter` (in-repo simulated `tiny` fixture) and
  `MovieLensAdapter` (recommended public demo path; `ratings.csv` + `movies.csv`, IDs
  prefixed `ml_u` / `ml_i`). MovieLens is fetched locally with
  `python scripts/fetch_movielens.py` into `data/raw/` (git-ignored) and is never committed.
- `materialize()` writes processed canonical CSVs to
  `data/processed/<dataset_id>/<version>/` with `manifest.json` and `DATA_CARD.md`.
  Raw directories are never modified.
- `DatasetManifest` records id, version, source, schema version, license/terms note,
  retrieval date, per-file sha256, combined checksum, synthetic flag, stats, and the
  cleaning/label policy identifiers once applied.
- `DatasetRegistry` (JSON index) registers manifests idempotently; re-registering the same
  `id@version` with a different checksum fails (`DatasetChecksumMismatch`) so data changes
  require a new version. `verify_snapshot()` detects modified files. `resolve(id)` without a
  version returns the highest version in natural order; experiment runs pin versions.

## Cleaning Pipeline (`recomlab.ingestion.cleaning`)

`clean_interactions(raw_rows, users, items, CleaningPolicy)` applies, in order:

| Step | Reason code(s) |
|---|---|
| `schema` | `schema:<validation code>` (e.g. `schema:naive_timestamp`) |
| `duplicates` | `duplicate_exact` (later identical copies), `duplicate_conflict` (every record of an `event_id` with conflicting content) |
| `known_users` | `unknown_user` |
| `known_items` | `unknown_item` |
| `time_window` | `out_of_time_window` (`min_time` inclusive, `max_time` exclusive) |

The `CleaningReport` lists before/removed/after counts per step and every rejection with
its row index and reason. `report.reconciles()` asserts
`n_input == n_output + removed` and that removals equal recorded rejections. Output is
sorted canonically, independent of input order, and identified by a content checksum.
`CleaningPolicy.policy_id` is stored in dataset manifests (`cleaning_policy`).
