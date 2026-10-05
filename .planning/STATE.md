# SpeechLens Execution State

## Current Phase: Phase 3 (Real Speech Data Collection & Alignment)
- **Status**: Active / In Progress
- **Active Directives**: GSD, Ponytail, CodeRabbit, Ralph Loop.

## Completed Tasks
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
