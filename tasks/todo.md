# Todo: M1 PR2 — Native audio entry spine

- [x] Create `feature/m1-audio-spine` from fork `main`
- [ ] Add `aimusic/audio/config.py` (YAML loader)
- [ ] Add `aimusic/audio/from_score.py` + `aimusic/audio/entry.py`
- [ ] Wire lazy `render-audio` CLI in `aimusic/app/cli.py`
- [ ] Declare `[audio-bridge]` in `pyproject.toml`
- [ ] Add `tests/test_audio_entry.py`; CI tests job uses `.[audio]`
- [ ] Update `docs/audio-next-steps.md`, `docs/audio-pipeline.md`
- [ ] Run lint/typecheck/tests; push; open PR to fork `main`
