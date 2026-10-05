# SpeechLens Roadmap

## Phase 1: Environment & Baseline Verification [COMPLETED]
- [x] Configure Python 3.12 virtual environment (`.venv`) with all core dependencies.
- [x] Install `eSpeak-NG` system dependency for synthetic baseline rendering.
- [x] Validate synthetic baseline generation (`make_synthetic_baseline.py`).
- [x] Verify full pipeline and unit test suite passing 10/10 (`pytest -v`).
- [x] Evaluate flaw injection benchmark (100% precision, 0 false positives on controls).

## Phase 2: Edge Case Hardening & Bug Fixes [ACTIVE]
- [x] Guard `long_pause` insertion and wrap-around at index 0 (EC-3.4 / EC-4.1).
- [x] Severity range validation `1 <= severity <= 4` (EC-4.5).
- [ ] Text normalisation improvements for numbers and smart quotes (EC-7.1 / EC-7.3).
- [ ] Sample rate mismatch detection / telephone audio handling (EC-2.5).
- [ ] Division guard for zero speech duration `max(T, 0.1)` in `score()` (EC-3.7).

## Phase 3: Real Speech Data Collection & Alignment
- [ ] Ingest public-domain speeches into `data/baseline/` and `data/transcripts/`.
- [ ] Run forced alignment via `torchaudio` MMS_FA.
- [ ] Generate real-speech contrastive dataset.
- [ ] Re-run evaluation suite on real data benchmarks.

## Phase 4: CLI & Reporting Polish
- [ ] Verify `analyze_cli.py` output formatting and JSON export.
- [ ] Complete final submission report and performance charts.
