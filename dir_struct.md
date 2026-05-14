# Directory Structure

The production code now lives under the `aimusic/` package and the test suite lives under `tests/`.

## Current Layout

```text
aimusic/
  app/
    main.py
  core/
    config.py
    core_types.py
    rng.py
    vocab.py
  planning/
    candidates.py
    graph.py
    sb.py
  scoring/
    gttm_features.py
    priors.py
    rhythm_features.py
  theory/
    edo.py
    tonal.py

tests/
  test_candidates.py
  test_config.py
  test_core_types.py
  test_graph.py
  test_gttm.py
  test_gttm_beatstate.py
  test_priors.py
  test_rhythm.py
  test_rng.py
  test_sb.py
  test_tonal.py
  test_vocab.py
```

## Responsibility Split

- `aimusic.core`
  - shared config, immutable domain types, vocabularies, RNG helpers
- `aimusic.theory`
  - EDO math and tonal-system utilities
- `aimusic.scoring`
  - GTTM energy features, prior interfaces, rhythm compatibility facade
- `aimusic.planning`
  - candidate generation, sparse graph construction, SB inference
- `aimusic.app`
  - entrypoints and demo wiring

## Dependency Direction

The intended dependency direction remains:

1. `aimusic.core`
2. `aimusic.theory`
3. `aimusic.scoring`
4. `aimusic.planning`
5. `aimusic.app`

`aimusic.core` should not depend on higher layers. Planning and app code can depend downward, not sideways back up the stack.

## Next Cleanup Passes

1. Update external callers and scripts to import from `aimusic.*` directly.
2. Add packaging metadata if this should be installable as a library.
3. Split oversized modules only if maintenance pain justifies it.
