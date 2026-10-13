# RecomLab Data Contract

## Canonical Tables

### users
- `user_id` — pseudonymous stable identifier
- optional non-sensitive research/demo attributes

### items
- `item_id`
- title/name only when license/privacy permits
- categories/genres/tags
- availability metadata

### interactions
- `event_id`
- `user_id`
- `item_id`
- `event_type`
- `value` (optional)
- `event_time` (timezone-aware)
- `session_id` (optional)

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

## Privacy

Use pseudonymous identifiers. Do not store names, emails, device identifiers, or raw free-text unless a specific research task requires them and the applicable privacy/ethics process permits it.

## Dataset Manifest

Every dataset snapshot records source, version, checksum, schema version, retrieval date, license/terms note, cleaning policy, and label policy.
