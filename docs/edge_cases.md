# SpeechLens — Edge Cases

> Derived from [`ARCHITECTURE.md`](file:///c:/Personal/IIT%20MANDI/docs/ARCHITECTURE.md), [`problem_statement.md`](file:///c:/Personal/IIT%20MANDI/docs/problem_statement.md), and direct code inspection of every module.

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
| **Current behaviour** | `load_audio()` resamples to `SR=16000` via librosa. 8 kHz → no content above 4 kHz; HF share will always be very low |
| **Risk** | 8 kHz input: `muffled` triggered on every word because HF energy is genuinely absent. This is a property of the recording, not the speaker |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Check original sample rate before resampling. If < 16 kHz, disable or discount the `muffled` detector and document the limitation. |

### EC-2.6 Octave-jump filter edge case (all frames removed)

| | |
|---|---|
| **Trigger** | Speaker with very unusual F0 (e.g. child at 350 Hz, or vocal fry at 50 Hz) where the Praat tracker oscillates |
| **Current behaviour** | `_drop_octave_jumps` may NaN out most frames → `pitch_range` all-NaN → no pitch regions |
| **Risk** | Degraded accuracy (misses real monotone/erratic pitch) |
| **Status** | ⚠️ Partially handled (doesn't crash, but silently loses coverage) |

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
| **Trigger** | Baseline has a pause ≥ 0.35 s before the first word (e.g. leading silence) and participant does not |
| **Current behaviour** | `gap[0] = 0` always (line 97 of features.py: loop starts at `i=1`). The `missing_pause` check `(wt_b["gap"] >= 0.35) & (wt_p["gap"] < 0.1)` will never fire for `i=0`. The `long_pause` detector accesses `words_p[i - 1]["end"]` when `i=0` → `words_p[-1]["end"]` (last word!) — **wrong start time** |
| **Risk** | **Silent wrong answer** — `long_pause` region at word 0 would have `start` = end of the last word |
| **Status** | 🔴 **Bug** |
| **Suggested fix** | Guard: `start = words_p[i - 1]["end"] if i > 0 else words_p[0]["start"]` in both the `long_pause` and `missing_pause` branches (lines 103 and 107 of analyze.py). Same issue exists on line 95 of `inject.py`. |

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
| **Trigger** | `words_p[-1]["end"] - words_p[0]["start"]` ≈ 0 (all words collapsed) |
| **Current behaviour** | `T ≈ 0` → `(r["end"] - r["start"]) / T` → division by zero / inf → `exp(-15 * inf) = 0` → scores collapse to 0 |
| **Risk** | Crash or nonsensical score |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | `T = max(T, 0.1)` or reject the analysis if `T < 1.0`. |

---

## 4. Flaw Injection — `inject.py`

### EC-4.1 `long_pause` injected at word 0

| | |
|---|---|
| **Trigger** | `choose_boundary()` returns `i=0` (margin not respected, or margin overridden) |
| **Current behaviour** | `apply_flaw` → `words[i - 1]["end"]` → `words[-1]["end"]` (wraps to last word). Insertion point `b` is near the end of the file |
| **Risk** | **Silent wrong answer** — silence inserted at the wrong location, ground-truth region is wrong |
| **Status** | 🔴 **Bug** (same pattern as EC-3.4) |
| **Suggested fix** | `choose_boundary` has `margin=4`, which prevents `i=0` in practice. But `apply_flaw` should still guard `i > 0`. |

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

### EC-4.4 `time_stretch` produces empty array

| | |
|---|---|
| **Trigger** | Very short segment (< 1 frame) stretched by a large factor |
| **Current behaviour** | `librosa.effects.time_stretch` may return a 0-length array → `_fade` division, `np.concatenate` produces misaligned audio |
| **Risk** | Crash or corrupted audio |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Guard: if `len(seg) < WIN`, skip injection and return original segment. |

### EC-4.5 Severity value out of range

| | |
|---|---|
| **Trigger** | `severity=0` or `severity=5` passed to `apply_flaw` |
| **Current behaviour** | `SEVERITY[flaw][severity - 1]` → index `-1` (last element) or index `4` (`IndexError`) |
| **Risk** | Crash or silent wrong answer (severity 0 maps to severity 4) |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Validate `1 <= severity <= 4` at the top of `apply_flaw`. |

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
| **Current behaviour** | `match()` returns `([], [], list(range(len(truth))))`. Precision = 0/0 undefined; recall = 0 |
| **Risk** | Division by zero in `run_eval.py` when computing precision/recall |
| **Status** | ⚠️ Partially handled (depends on how `run_eval.py` computes the metric — needs a `tp / max(tp + fp, 1)` guard) |

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

---

## 7. Text Utilities — `text.py`

### EC-7.1 Transcript with numbers or abbreviations

| | |
|---|---|
| **Trigger** | "We scored 100 points in the NBA finals" |
| **Current behaviour** | `normalize_words()` strips `100` and `NBA` entirely (only keeps `a-z` + `'`) → word count drops → alignment mismatch if the speaker says those words |
| **Risk** | **Silent wrong answer** — alignment breaks or `analyze()` raises `ValueError` due to word count mismatch |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Either (a) expand numbers to words (`100` → `one hundred`) before normalisation, or (b) document that transcripts must be pre-normalised with numbers spelled out. |

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
| **Trigger** | Text pasted from Word/Google Docs: `"don't"` (U+2019 right single quotation mark) |
| **Current behaviour** | `normalize_words()` regex `[a-z']` only matches ASCII apostrophe `'` (U+0027). The curly `'` is stripped → `"dont"` → MMS_FA may or may not align it |
| **Risk** | Degraded accuracy |
| **Status** | ⚠️ Unhandled |
| **Suggested fix** | Normalise `'`, `'`, `` ` `` → `'` before the regex filter. |

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

## Summary

| Risk level | Count | IDs |
|---|---|---|
| 🔴 **Bug** (produces wrong results today) | 2 | EC-3.4, EC-4.1 |
| ⚠️ **Silent wrong answer** | 5 | EC-1.2, EC-2.1, EC-4.2, EC-4.3, EC-8.3 |
| ⚠️ **Degraded accuracy** | 8 | EC-1.4, EC-2.3, EC-2.4, EC-2.5, EC-2.6, EC-3.2, EC-3.3, EC-3.6 |
| ⚠️ **Crash (unhandled)** | 5 | EC-1.3, EC-4.4, EC-4.5, EC-6.1, EC-6.2 |
| ✅ **Handled** | 10 | EC-1.1, EC-3.1, EC-3.5, EC-5.1, EC-6.3, EC-7.2, EC-9.1, EC-9.2, EC-10.1, EC-10.3 |

### Priority fixes before submission

1. **EC-3.4 / EC-4.1** — `words[i - 1]` wrap-around bug at index 0. Quick one-line guard.
2. **EC-4.5** — Severity validation. One-line assertion.
3. **EC-7.1 / EC-7.3** — Text normalisation losing numbers and smart quotes. Moderate effort.
4. **EC-2.5** — 8 kHz input false-triggers `muffled`. Check original sample rate.
5. **EC-3.7** — Division by ≈ 0 in `score()`. One-line `max(T, 0.1)`.
