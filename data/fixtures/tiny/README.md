# Tiny synthetic fixture (`tiny`)

**Simulated data for software tests only.** Not derived from any real users.
Regenerate with `python scripts/make_tiny_fixture.py`.

- 7 pseudonymous users: `u7` has no interactions (fully cold user); `u6` has one (sparse).
- 12 items in two taste clusters (action/scifi `i01`–`i05`, drama/romance `i06`–`i10`);
  `i11` is a cold item with no interactions and `i12` is unavailable.
- 33 interactions in January 2024 (UTC): explicit `rating` events (1–5) plus implicit
  `click`, `watch` (fraction watched) and `purchase` events.
