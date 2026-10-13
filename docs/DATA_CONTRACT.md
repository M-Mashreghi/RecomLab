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

### impressions

- `impression_id`
- `request_id`
- `user_id`
- `item_id`
- `position`
- `model_id`
- `shown_at`
- `propensity` (optional; never fabricated)

### feedback_events

- `feedback_id`
- optional `impression_id`
- `user_id`
- `item_id`
- `feedback_type`
- `value`
- `event_time`

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
