# SpeechLens — Edge Cases

> Derived from [`ARCHITECTURE.md`](ARCHITECTURE.md), [`problem_statement.md`](problem_statement.md), and direct code inspection of every module.

---

## How to read this document

Each edge case has:

| Field | Meaning |
|---|---|
| **Module** | Source file where the edge case lives |
| **Trigger** | What input or state causes it |
| **Current behaviour** | What the code does today |
| **Risk** | `crash`, `silent wrong answer`, `degraded accuracy`, `cosmetic` |
| **Status** | `handled`, `unhandled`, `partially handled` |
| **Suggested fix** | Concrete recommendation (if unhandled) |

---

## 1. Forced Alignment — `align.py`

### EC-1.1 Empty or punctuation-only transcript

| | |
|---|---|
| **Trigger** | Transcript is `""`, `"!!!"`, or all non-alphabetic characters |
| **Current behaviour** | `normalize_words()` returns `[]` → `ValueError("Transcript has no alignable words")` |
| **Risk** | Crash (expected) |
| **Status** | ✅ Handled |

### EC-1.2 Transcript–audio mismatch (wrong text for the audio)

| | |
|---|---|
| **Trigger** | User uploads the correct audio but a transcript from a different speech |
| **Current behaviour** | MMS_FA CTC aligner runs without error but produces garbage timestamps — words get squeezed into silence or stretched across unrelated speech |
| **Risk** | **Silent wrong answer** — downstream regions will be meaningless |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | After alignment, check that the mean word confidence (if available from the aligner) exceeds a floor, or that median word duration is in a sane range (e.g. 0.05–2.0 s). Reject with HTTP 422 / CLI error if not. |

### EC-1.3 Audio contains no speech (silence / noise only)

| | |
|---|---|
| **Trigger** | Blank recording, ambient noise, or a music-only file |
| **Current behaviour** | Aligner either crashes or places all words at t ≈ 0 with near-zero duration |
| **Risk** | Crash or silent wrong answer |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Pre-check: if `librosa.feature.rms` max < −40 dB, reject early. |

### EC-1.4 Audio much longer or shorter than the transcript implies

| | |
|---|---|
| **Trigger** | 10-word transcript on a 5-minute file, or 500-word transcript on a 10-second file |
| **Current behaviour** | Aligner silently produces extreme durations |
| **Risk** | Degraded accuracy |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Sanity-check: expected duration ≈ `word_count / 2.5` (words per second). Warn if actual duration is > 3× or < 0.3× expected. |

### EC-1.5 Non-English or mixed-language transcript

| | |
|---|---|
| **Trigger** | Hindi words, accented characters, CJK |
| **Current behaviour** | `normalize_words()` strips everything except `a-z` and `'` → foreign words vanish → word count drops or becomes 0 |
| **Risk** | Crash or silent word loss |
| **Status** | ⚠️ Partially handled (crashes cleanly if all words removed, but silently drops valid foreign words otherwise) |

---

## 2. Feature Extraction — `features.py`

### EC-2.1 All-silence audio (no voiced frames)

| | |
|---|---|
| **Trigger** | Audio is digital silence or extremely quiet |
| **Current behaviour** | `voiced.any()` is `False` → `f0_median_hz` defaults to `120.0`, `f0_st` is all-NaN, `active` mask is all-False, `energy_db` reference = `np.median(e_db)` (noise floor) |
| **Risk** | **Silent wrong answer** — pitch features are meaningless; energy baseline is noise |
| **Status** | ⚠️ Partially handled (doesn't crash, but downstream deltas are garbage) |

### EC-2.2 Very short audio (< 1 second / fewer than ~5 words)

| | |
|---|---|
| **Trigger** | User uploads a 0.5-second clip |
| **Current behaviour** | Feature arrays have very few frames. `_threshold()` returns `floor` when `len(d) < 5`. `word_table` pitch pooling over 3-word window may have < 8 voiced frames for all words → `pitch_range` is all-NaN |
| **Risk** | Degraded accuracy (all pitch detectors produce NaN → no pitch regions detected even if pitch is wildly wrong) |
| **Status** | ⚠️ Partially handled |

### EC-2.3 Clipped / distorted audio

| | |
|---|---|
| **Trigger** | Recording with gain set too high; samples pegged at ±1.0 |
| **Current behaviour** | RMS energy is artificially high, harmonics are distorted → F0 tracker may produce octave errors, HF share inflated |
| **Risk** | Degraded accuracy — may flag clipping as `erratic_pitch` or `volume_spike` |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Detect clipping: if > 1% of samples are at ±(1.0 − ε), add a warning to the output. |

### EC-2.4 Audio with background music

| | |
|---|---|
| **Trigger** | Speech over a music bed (common in TED talks, competition recordings) |
| **Current behaviour** | Music inflates HF share and energy; pitch tracker may lock onto melody |
| **Risk** | False positives for `muffled` (or false negatives — music raises HF) and `volume_spike` |
| **Status** | ⚠️ Unhandled |

### EC-2.5 Extreme sample rate mismatch

| | |
|---|---|
| **Trigger** | Input audio is 8 kHz (telephone) or 48 kHz |
| **Current behaviour** | `speechlens/io.py:load_audio` probes `sf.info(str(path))` before resampling and emits a `UserWarning` when `info.samplerate < sr` (16000 Hz) warning that high-frequency clarity features may be affected (verified by `tests/test_text_and_io.py:test_load_audio_warns_on_low_sample_rate`), then resamples to `SR=16000`. |
| **Risk** | 8 kHz input lacks content above 4 kHz, so `muffled` may still trigger if the warning is ignored (the detector itself is not auto-disabled). |
| **Status** | ✅ **Resolved (warning implemented)** — `speechlens/io.py:load_audio` warns on `< 16 kHz`; auto-disabling `muffled` remains optional future work |

### EC-2.6 Octave-jump filter edge case (all frames removed)

| | |
|---|---|
| **Trigger** | Speaker with very unusual F0 (e.g. child at 350 Hz, or vocal fry at 50 Hz) where the Praat tracker oscillates |
| **Current behaviour** | `_drop_octave_jumps` may NaN out most frames → `pitch_range` all-NaN → no pitch regions |
| **Risk** | Degraded accuracy (misses real monotone/erratic pitch) |
| **Status** | ⚠️ Partially handled (doesn't crash, but silently loses coverage) |

### EC-2.7 Low-SNR (10 dB) creaky/sub-harmonic pitch dropout

| | |
|---|---|
| **Trigger** | Additive noise (`10 dB` SNR, e.g. `speech1__noise_10dB`) masks quiet downward creaky pitch tails under autocorrelation (`to_pitch_ac`), collapsing `pitch_range` vs clean baseline |
| **Current behaviour** | `speechlens/features.py:extract` switches Pass-2 F0 estimation to Praat cross-correlation (`snd.to_pitch_cc(..., silence_threshold=0.005)`), tracking the pitch tail consistently across clean and noisy audio (`0` FPs across all `18` active runs in `scripts/controls.py`). |
| **Risk** | None at `10 dB` SNR |
| **Status** | ✅ **Resolved** — `speechlens/features.py:extract` |

---

## 3. Comparison Engine — `analyze.py`

### EC-3.1 Word count mismatch between participant and baseline

| | |
|---|---|
| **Trigger** | Different transcripts, or one alignment dropped a word |
| **Current behaviour** | `analyze()` raises `ValueError` |
| **Risk** | Crash (expected, correct) |
| **Status** | ✅ Handled |

### EC-3.2 Zero-duration word (start == end)

| | |
|---|---|
| **Trigger** | Aligner collapses a very short word to a zero-length interval |
| **Current behaviour** | `dur[i] = 0` → `ln(dur_P / dur_B)` → `ln(0 / x)` = `-inf` or `ln(1e-3 / x)` because of the `np.maximum(..., 1e-3)` clamp |
| **Risk** | Degraded accuracy — the 1e-3 clamp prevents `-inf` but produces a very large negative delta that almost certainly triggers `rushed` |
| **Status** | ⚠️ Partially handled (no crash, but may produce a false positive) |
| **Suggested fix** | If a word duration < 10 ms, mark that word's rate delta as NaN and exclude from region detection. |

### EC-3.3 Fewer than 5 words total

| | |
|---|---|
| **Trigger** | Very short transcript ("Thank you very much") |
| **Current behaviour** | `_threshold()` returns `floor` for all features. `_runs(min_len=2)` needs at least 2 consecutive flagged words, and `min_len=3` for pitch/volume needs 3 — unlikely with only 4-5 words |
| **Risk** | Degraded accuracy — almost nothing will be detected regardless of how bad the delivery is |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Warn the user: "Transcript too short for reliable analysis (minimum ~20 words recommended)." |

### EC-3.4 `long_pause` at word index 0

| | |
|---|---|
| **Trigger** | Baseline has a pause ≥ 0.35 s before the first word (e.g. leading silence) and participant does not, or participant pauses before word 0 |
| **Current behaviour** | `speechlens/features.py:word_table` sets `gap[0] = 0.0` (`for i in range(1, n):`). In `speechlens/analyze.py:detect_regions`, both `long_pause` and `missing_pause` explicitly guard `i > 0`: `start=words_p[i - 1]["end"] if i > 0 else words_p[i]["start"]` (and identically for `words_b`), preventing `words[-1]` index wrap-around. |
| **Risk** | None — wrap-around bug fixed |
| **Status** | ✅ **Resolved** — `speechlens/analyze.py:detect_regions` |

### EC-3.5 All deltas are identical (MAD = 0)

| | |
|---|---|
| **Trigger** | Identical participant and baseline (already tested) or a pathological case where all words have the same delta |
| **Current behaviour** | `MAD = 0` → `K_MAD * 1.4826 * 0 = 0` → `threshold = max(floor, 0) = floor`. Falls back to the perceptual floor correctly |
| **Risk** | None |
| **Status** | ✅ Handled |

### EC-3.6 NaN propagation in pitch deltas

| | |
|---|---|
| **Trigger** | One speaker is fully unvoiced over a passage (whispering, vocal fry) |
| **Current behaviour** | `pitch_range` is NaN → `ln((NaN + 0.5) / ...)` = NaN → `dcc` is NaN → `nan_to_num(dcc)` converts to 0.0 → no pitch flag for that word |
| **Risk** | Degraded accuracy — a whispered passage should probably flag as `monotone` but won't |
| **Status** | ⚠️ Partially handled (no crash, but misses the flaw) |

### EC-3.7 Speech duration ≈ 0 (division in `score()`)

| | |
|---|---|
| **Trigger** | `words_p[-1]["end"] - words_p[0]["start"]` or `words_b[-1]["end"] - words_b[0]["start"]` ≈ 0 (all words collapsed) |
| **Current behaviour** | `speechlens/analyze.py:score` clamps `ref_duration = max(ref_duration, 0.1)`, and `speechlens/analyze.py:analyze` clamps `T = max(..., 0.1)` and `ref_duration = max(..., 0.1)` before calling `score()`. |
| **Risk** | None (guarded against division by zero) |
| **Status** | ✅ **Resolved** — `speechlens/analyze.py:score`, `speechlens/analyze.py:analyze` |

---

## 4. Flaw Injection — `inject.py`

### EC-4.1 `long_pause` injected at word 0

| | |
|---|---|
| **Trigger** | `choose_boundary()` returns `i=0` or caller passes `i=0` directly |
| **Current behaviour** | `speechlens/inject.py:apply_flaw` explicitly checks `if i < 1: raise ValueError("Cannot insert long_pause before the first word (i=0)")`, and `speechlens/inject.py:choose_boundary` uses `margin=4`. |
| **Risk** | Crash (expected fail-fast guard) |
| **Status** | ✅ **Resolved** — `speechlens/inject.py:apply_flaw` |

### EC-4.2 Overlapping flaw spans in a multi-flaw plan

| | |
|---|---|
| **Trigger** | `build_dataset.py` creates a plan with two flaws whose `[i, j]` ranges overlap |
| **Current behaviour** | `make_flawed` sorts by `-i` and applies backwards, so the time-shift logic assumes spans are disjoint. Overlapping spans corrupt word timings — earlier flaw's timestamps get double-shifted |
| **Risk** | **Silent wrong answer** — ground-truth labels are incorrect |
| **Status** | ⚠️ Partially handled (`build_dataset.py` probably avoids overlap, but `make_flawed` doesn't validate) |
| **Suggested fix** | Add an assertion in `make_flawed`: verify no two plan items have overlapping `[i, j]` ranges. |

### EC-4.3 `_monotone` with < 3 pitch points

| | |
|---|---|
| **Trigger** | Very short segment or unvoiced segment passed to Praat monotone injection |
| **Current behaviour** | `if n < 3: return seg` — returns the original segment unmodified |
| **Risk** | **Silent wrong answer** — label says `monotone` was injected but audio is unchanged. Eval will count it as a false negative |
| **Status** | ⚠️ Partially handled (no crash, but label is wrong) |

### EC-4.4 Time-stretch empty or ultra-short segment

| | |
|---|---|
| **Trigger** | Empty or sub-frame word span `[i, j]` passed to `rushed`/`dragging` (`ffmpeg` `atempo`) |
| **Current behaviour** | `speechlens/inject.py:apply_flaw` raises `ValueError` on `len(seg) == 0`, and `speechlens/inject.py:_atempo` validates `0.5 <= rate <= 2.0` and raises `RuntimeError` if `len(out) == 0`. No minimum `WIN` sample guard exists before invoking `ffmpeg`. |
| **Risk** | Crash on degenerate ultra-short spans (fails fast instead of silent misalignment) |
| **Status** | ⚠️ Partially handled (`apply_flaw` + `_atempo` fail fast; minimum `WIN` guard not added) |
| **Suggested fix** | Guard: if `len(seg) < WIN`, skip or raise explicitly before invoking `ffmpeg`. |

### EC-4.5 Severity value out of range

| | |
|---|---|
| **Trigger** | `severity=0` or `severity=5` passed to `apply_flaw` |
| **Current behaviour** | `speechlens/inject.py:apply_flaw` validates `if not (1 <= severity <= 4): raise ValueError(f"severity must be 1-4, got {severity}")` before indexing `SEVERITY[flaw]`. |
| **Risk** | Crash (expected fail-fast guard) |
| **Status** | ✅ **Resolved** — `speechlens/inject.py:apply_flaw` |

---

## 5. Evaluation — `evaluate.py`

### EC-5.1 Zero-length (degenerate) intervals

| | |
|---|---|
| **Trigger** | A region with `start == end` (e.g. collapsed pause detection) |
| **Current behaviour** | `union <= 0` → fallback: `1.0 if abs(a[0] - b[0]) < 0.3 else 0.0`. This is a proximity heuristic |
| **Risk** | Cosmetic — IoU metric is not truly IoU in this case |
| **Status** | ✅ Handled (graceful fallback) |

### EC-5.2 No predictions or no ground-truth regions

| | |
|---|---|
| **Trigger** | Clean file has 0 predicted regions; a file where all flaws are severity 1 (below detection threshold) |
| **Current behaviour** | `match()` returns `([], [], list(range(len(truth))))`. Both `scripts/run_eval.py:main` and `scripts/eval_real.py:run_real_eval` guard precision/recall/F1 denominators with `max(..., 1)` and `max(..., 1e-9)` (and return `recall=None` when no ground truth exists for a class). |
| **Risk** | None (division-by-zero guarded) |
| **Status** | ✅ **Resolved** — `scripts/run_eval.py:main`, `scripts/eval_real.py:run_real_eval` |

### EC-5.3 Type mismatch: injected `rushed` detected as `dragging`

| | |
|---|---|
| **Trigger** | Severity 1 `rushed` where the time-stretch is so slight that robust-centring flips the sign |
| **Current behaviour** | `require_type=True` → no match → counted as both a false positive and a false negative |
| **Risk** | Degraded accuracy metric (correct detection, wrong label) |
| **Status** | ⚠️ Partially handled (by design — type matching is strict) |

---

## 6. Audio I/O — `io.py`

### EC-6.1 Unsupported audio format

| | |
|---|---|
| **Trigger** | `.ogg`, `.aac`, `.m4a` without ffmpeg installed |
| **Current behaviour** | `librosa.load` raises `audioread.NoBackendError` or similar |
| **Risk** | Crash with an unhelpful error message |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Catch and re-raise with a user-friendly message listing supported formats. |

### EC-6.2 Corrupted or truncated audio file

| | |
|---|---|
| **Trigger** | Partial download, disk error |
| **Current behaviour** | librosa/soundfile raise an opaque C-level error |
| **Risk** | Crash |
| **Status** | ⚠️ Unhandled |

### EC-6.3 Stereo or multi-channel audio

| | |
|---|---|
| **Trigger** | Stereo WAV file |
| **Current behaviour** | `librosa.load(..., mono=True)` downmixes to mono automatically |
| **Risk** | None |
| **Status** | ✅ Handled |

### EC-6.4 Broad exception swallow around sample-rate probe

| | |
|---|---|
| **Trigger** | Unexpected non-I/O error inside `sf.info()` probe in `load_audio` |
| **Current behaviour** | `speechlens/io.py:load_audio` narrows the probe fallback catch to `except (OSError, RuntimeError): pass`, avoiding masking programming/logic errors while allowing unsupported container probes to fall through to `librosa.load`. |
| **Risk** | None |
| **Status** | ✅ **Resolved** — `speechlens/io.py:load_audio` |

---

## 7. Text Utilities — `text.py`

### EC-7.1 Transcript with numbers or abbreviations

| | |
|---|---|
| **Trigger** | `"We scored 100 points in the NBA finals"` |
| **Current behaviour** | `speechlens/text.py:normalize_words` runs `re.findall(r"\d+", text)` and fails fast with `ValueError` instructing that numbers must be spelled out (verified by `tests/test_text_and_io.py:test_normalize_words_rejects_numeric_digits`). Alphabetic abbreviations (e.g. `"NBA"`) are lowercased (`"nba"`) and retained by `[a-z']+`. |
| **Risk** | Crash on unexpanded digits (expected fail-fast guard); spoken letter-by-letter acronyms remain dependent on `MMS_FA` lexicon alignment |
| **Status** | ✅ **Resolved (digit validation)** — `speechlens/text.py:normalize_words` |

### EC-7.2 Apostrophes and contractions

| | |
|---|---|
| **Trigger** | "don't", "it's", "I'll" |
| **Current behaviour** | Apostrophe is kept → `["don't", "it's"]`. MMS_FA should handle these correctly |
| **Risk** | None expected |
| **Status** | ✅ Handled |

### EC-7.3 Unicode smart quotes / curly apostrophes

| | |
|---|---|
| **Trigger** | Text pasted from Word/Google Docs: `"It’s wonderful, don‘t you think? \`SpeechLens\` is live."` |
| **Current behaviour** | `speechlens/text.py:normalize_words` replaces `\u2019`, `\u2018`, and `` ` `` with ASCII `'`, extracts `[a-z']+` tokens, and strips boundary quotes (`t.strip("'")`) while preserving internal contractions (verified by `tests/test_text_and_io.py:test_normalize_words_handles_smart_quotes`). |
| **Risk** | None |
| **Status** | ✅ **Resolved** — `speechlens/text.py:normalize_words` |

---

## 8. Scripts

### EC-8.1 `build_dataset.py` — Missing baseline audio files

| | |
|---|---|
| **Trigger** | `data/baseline/` is empty (as it currently is) |
| **Current behaviour** | Script finds no files → loop body never executes → empty `manifest.csv` |
| **Risk** | Cosmetic (no crash, but no output either, with no warning) |
| **Status** | ⚠️ Partially handled |

### EC-8.2 `run_eval.py` — Missing label files

| | |
|---|---|
| **Trigger** | Flawed audio exists but its label JSON is missing or malformed |
| **Current behaviour** | `json.load` raises `FileNotFoundError` or `JSONDecodeError` |
| **Risk** | Crash |
| **Status** | ⚠️ Unhandled |

### EC-8.3 `analyze_cli.py` — Baseline and participant from different texts

| | |
|---|---|
| **Trigger** | User provides `--participant` and `--baseline` that were recorded from different scripts |
| **Current behaviour** | If alignments have different word counts → `ValueError` in `analyze()`. If word counts happen to match but words differ → **silent wrong answer** |
| **Risk** | Silent wrong answer |
| **Status** | ⚠️ Partially handled (crashes on count mismatch, but not on content mismatch) |

---

## 9. Scoring & Rubric — `analyze.py`

### EC-9.1 No regions detected → perfect score

| | |
|---|---|
| **Trigger** | Participant delivery is genuinely identical to baseline, or all flaws are below threshold |
| **Current behaviour** | All `impact = 0` → all dimensions = 100.0 → overall = 100.0 |
| **Risk** | None (correct) |
| **Status** | ✅ Handled |

### EC-9.2 Single catastrophic flaw tanks overall score via `min()`

| | |
|---|---|
| **Trigger** | One dimension (e.g. `clarity`) = 10, all others = 95 |
| **Current behaviour** | `overall = 0.5 * weighted_mean + 0.5 * min(dims)` = `0.5 * ~91 + 0.5 * 10` ≈ 50.5 |
| **Risk** | Cosmetic — this is by design ("judges penalise the weakest aspect"), but a single muffled region in an otherwise flawless speech drops the score by ~50 points. May surprise users |
| **Status** | ✅ Handled (by design, but worth documenting in the dashboard) |

### EC-9.3 Overlapping regions of different types inflate penalty

| | |
|---|---|
| **Trigger** | A passage flagged as both `rushed` and `volume_drop` (common: speaking fast + trailing off) |
| **Current behaviour** | Both regions' `duration_fraction * severity` are summed into their respective dimensions independently. The same time span counts against two dimensions |
| **Risk** | Degraded accuracy — total penalty can exceed what a single passage "deserves" |
| **Status** | ⚠️ Partially handled (penalty is per-dimension, so no double-counting within a single dimension, but the `min()` in overall may amplify) |

---

## 10. System-Level / Deployment

### EC-10.1 espeak-ng not installed

| | |
|---|---|
| **Trigger** | Running tests or `make_synthetic_baseline.py` on a machine without espeak-ng |
| **Current behaviour** | Tests skip via `pytest.importorskip` or a PATH check. `make_synthetic_baseline.py` crashes with `FileNotFoundError` |
| **Risk** | Crash (expected) |
| **Status** | ✅ Handled (tests skip; script should be run only with espeak-ng) |

### EC-10.2 torch/torchaudio not installed

| | |
|---|---|
| **Trigger** | Running alignment without the optional torch dependency |
| **Current behaviour** | `align.py` does a lazy `import torch; import torchaudio` inside `align_words()` → `ModuleNotFoundError` |
| **Risk** | Crash |
| **Status** | ⚠️ Partially handled (clear error, but no user-friendly message) |

### EC-10.3 Determinism broken by library updates

| | |
|---|---|
| **Trigger** | librosa or parselmouth minor version changes algorithm internals |
| **Current behaviour** | `test_injection_is_deterministic` checks MD5 of output bytes — will fail if any dependency changes output |
| **Risk** | Test failure (expected; that's the point of the test) |
| **Status** | ✅ Handled |

### EC-10.4 Windows path issues

| | |
|---|---|
| **Trigger** | Paths with spaces or unicode characters on Windows |
| **Current behaviour** | `io.py` uses `str(path)` → should work. Some subprocess calls in scripts may fail with spaces |
| **Risk** | Crash on some Windows setups |
| **Status** | ⚠️ Partially handled |

---

## 11. Web Server & Upload Handling — `viz/server.py`

### EC-11.1 Invalid transcript digits or word-count mismatch on `POST /api/analyze`

| | |
|---|---|
| **Trigger** | Client submits digits/empty transcript (`normalize_words()` `ValueError`) or alignments whose word counts differ (`analyze()` `ValueError`) |
| **Current behaviour** | `viz/server.py:SpeechLensHandler.handle_analyze` catches `ValueError` separately and responds with HTTP `400` JSON `{"error": str(err)}` instead of HTTP `500` (verified by `tests/test_pipeline.py:test_server_returns_400_on_value_error_and_413_on_oversized_upload`). |
| **Risk** | None |
| **Status** | ✅ **Resolved** — `viz/server.py:handle_analyze` |

### EC-11.2 Oversized upload or stale `%TEMP%` WAV accumulation

| | |
|---|---|
| **Trigger** | Large base64 upload on `POST /api/analyze` or repeated interactive uploads leaving `tmp_*_*.wav` files in `%TEMP%` |
| **Current behaviour** | `viz/server.py:SpeechLensHandler.handle_analyze` enforces `MAX_UPLOAD_BYTES = 25 * 1024 * 1024` (returning HTTP `413` JSON before reading `rfile`) and prunes `tmp_*_*.wav` files older than `1800` s via `_cleanup_old_tmp_wavs()`. |
| **Risk** | None |
| **Status** | ✅ **Resolved** — `viz/server.py:handle_analyze`, `viz/server.py:_cleanup_old_tmp_wavs` |

---

## Summary

| Risk level / Status | Count | IDs |
|---|---|---|
| 🔴 **Bug** (known unfixed wrong results) | 0 | — |
| ⚠️ **Silent wrong answer (open / partial)** | 5 | EC-1.2, EC-2.1, EC-4.2, EC-4.3, EC-8.3 |
| ⚠️ **Degraded accuracy / cosmetic (open / partial)** | 12 | EC-1.4, EC-1.5, EC-2.2, EC-2.3, EC-2.4, EC-2.6, EC-3.2, EC-3.3, EC-3.6, EC-5.3, EC-8.1, EC-9.3 |
| ⚠️ **Crash (unhandled / partial)** | 7 | EC-1.3, EC-4.4, EC-6.1, EC-6.2, EC-8.2, EC-10.2, EC-10.4 |
| ✅ **Handled / Resolved** | 22 | EC-1.1, EC-2.5, EC-2.7, EC-3.1, EC-3.4, EC-3.5, EC-3.7, EC-4.1, EC-4.5, EC-5.1, EC-5.2, EC-6.3, EC-6.4, EC-7.1, EC-7.2, EC-7.3, EC-9.1, EC-9.2, EC-10.1, EC-10.3, EC-11.1, EC-11.2 |

### Remaining Open Priorities (audited against current code)

1. **EC-1.2 / EC-1.3 / EC-8.3** — Validate transcript–audio match (word confidence/duration sanity), reject silent/no-speech recordings early, and verify word-text alignment equality in `analyze()`.
2. **EC-4.2** — Validate disjoint `[i, j]` spans inside `speechlens/inject.py:make_flawed`.
3. **EC-2.3 / EC-2.4** — Clipping and background-music warnings in `speechlens/features.py:extract`.
4. **EC-3.2 / EC-3.6** — Exclude `< 10 ms` collapsed words from rate deltas and handle unvoiced/whispered passages explicitly in `speechlens/analyze.py:detect_regions`.
5. **EC-6.1 / EC-6.2** — Surface friendly decoder errors in `speechlens/io.py:load_audio`.

*Resolved items previously listed here (now implemented and verified): EC-2.5 & EC-6.4 (`speechlens/io.py:load_audio`), EC-2.7 (`speechlens/features.py:extract`), EC-3.4 (`speechlens/analyze.py:detect_regions`), EC-3.7 (`speechlens/analyze.py:score`, `analyze`), EC-4.1 & EC-4.5 (`speechlens/inject.py:apply_flaw`), EC-5.2 (`scripts/run_eval.py:main`, `scripts/eval_real.py:run_real_eval`), EC-7.1 & EC-7.3 (`speechlens/text.py:normalize_words`), EC-11.1 & EC-11.2 (`viz/server.py:handle_analyze`).*
