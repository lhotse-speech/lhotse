# Development

- Keep diffs focused. Reuse helpers; minimize options, duplication, and inheritance.
- Prefer descriptive names, keyword arguments, accurate types, and `pathlib`.
- Extract cohesive modules; keep public methods and CLI wrappers thin. Update exports and documentation, including environment variables. Use configured formatters.

## Manifests and dependencies

- Preserve callers, IDs, old manifests, round trips, and stable evaluation splits.
- Return copies from transforms. Propagate timing through supervisions, alignments, and custom arrays. Reuse sample/frame helpers; bound tolerances.
- Import optional dependencies locally; initialize on demand. Give installation hints. Keep imports free of side effects.
- Preserve lazy/eager support and bounded memory. Stream large manifests as compressed JSONL. Keep training choices downstream.

## Tests

- Add small core regression tests that fail before the fix. Assert exact values, full equality, shapes, and dtypes.
- Reuse dummies, fixtures, temporary paths, and parametrization. Test one behavior; cover affected variants and serialization.
- Seed randomness; extend existing timing/composition invariant tests.
- Smoke-test recipes on small real subsets; skip recipe-specific unit-test files.
- Skip unavailable integrations cleanly. Test imports without extras; enable optional coverage in suitable CI. Avoid duplicate tests, unnecessary multiprocessing, and large downloads.

## Verification

- Run affected tests and formatting/lint checks. Check real audio/backends; plot or listen when useful.
- Benchmark against a baseline; record workload and environment.
- Diagnose unrelated CI flakes separately. Codecov is advisory. Report checks and remaining gaps.

## Review

- Trace callers; reproduce downstream regressions.
- Report bugs first: location, trigger, consequence, fix. Separate optional suggestions.
