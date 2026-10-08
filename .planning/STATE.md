# SpeechLens Execution State

## Current Phase: Phase 3 (Real Speech Data Collection & Alignment)
- **Status**: Active / In Progress
- **Active Directives**: GSD, Ponytail, CodeRabbit, Ralph Loop.

## Completed Tasks
0. **Ground-Truth Rebuild and Evaluation Audit (2026-10-08)**:
   - Corrected `run_eval.py` so severity recall is explicitly reported at strict IoU >= 0.5; severity 1 is 1/6 (0.17), not the prior IoU 0.3 value of 0.33.
   - Added an auditable IoU 0.3 monotone-FP dump with word indices/text and per-recording two-pass F0 bounds. Canonical indices 99-101 are `a promise we`; 102-104 are `keep every single`.
   - Tested pair-shared F0 bounds and rejected them: they neither reduced the five monotone FPs nor preserved clean +/-3-semitone controls. Per-recording two-pass bounds remain in use.
   - Replaced phase-vocoder rushed/dragging injection with ffmpeg `atempo`, rebuilt all 30 files, refreshed MMS_FA participant alignments, and regenerated matched benchmark/control artifacts.
   - Final matched result: strict IoU >= 0.5 recall 26/36 (0.72), precision 26/31 (0.84), F1 0.78; no IoU 0.3 monotone or volume-drop FPs. Full controls are clean except the documented `speech1` 10 dB SNR monotone FP.
   - Verification: `pytest --collect-only -q` collects 18 tests; `pytest -v` passes 18/18.
1. **Synthetic Data Audit**:
   - Verified why metrics shifted (files 28 -> 30 due to `--mixed 4` in `build_dataset.py`, adding 3 severity-2 flaws that were all recalled).
   - Saved full evaluation output to [`eval/synthetic_2026-10-05.txt`](file:///c:/Personal/IIT%20MANDI/speechlens/eval/synthetic_2026-10-05.txt).
   - Analyzed non-monotonicity in `rushed` (-0.80) and `muffled` (-0.74):
     - `rushed`: higher speed reduces physical duration, reducing `(dur / T)` penalty.
     - `muffled`: `r["severity"]` saturates at 1.0, leaving penalty dependent on random word length variations.
2. **Real Speech Alignment Pipeline**:
   - Installed `torch 2.5.1` and `torchaudio 2.5.1` with CPU wheels.
   - Fixed Windows memory allocation / access violation in MMS_FA by implementing meta-device state-dict assignment in [`speechlens/align.py`](file:///c:/Personal/IIT%20MANDI/speechlens/speechlens/align.py).
   - Downloaded and cut 45 s continuous reading from LibriVox (*The Tell-Tale Heart*) to [`data/baseline/speech1.wav`](file:///c:/Personal/IIT%20MANDI/speechlens/data/baseline/speech1.wav).
   - Prepared exact normalized transcript in [`data/transcripts/speech1.txt`](file:///c:/Personal/IIT%20MANDI/speechlens/data/transcripts/speech1.txt).
   - Implemented and verified [`scripts/check_alignment.py`](file:///c:/Personal/IIT%20MANDI/speechlens/scripts/check_alignment.py) (78 words aligned, 0 zero-duration words).
   - Implemented [`scripts/align_to_audacity.py`](file:///c:/Personal/IIT%20MANDI/speechlens/scripts/align_to_audacity.py) and exported labels to [`data/baseline/speech1_labels.txt`](file:///c:/Personal/IIT%20MANDI/speechlens/data/baseline/speech1_labels.txt).
   - Full test suite verified passing: 10/10 tests green.
