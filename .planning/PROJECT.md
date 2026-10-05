# Project: SpeechLens (Track C)

## Core Mission
SpeechLens performs contrastive speech analytics with temporal flaw grounding. It compares a participant's speech delivery against a baseline recording of the same text, flags exact time regions where delivery deviates, provides causal and plain-language explanations, and outputs a calibrated rubric score (0–100).

## Architecture Stack
1. **Forced Alignment**: MMS_FA (torchaudio) anchors words on both participant and baseline timelines.
2. **Speaker-Normalised Features**: F0 in semitones re speaker median, energy in dB re median speech level, STFT high-frequency power share, CMN-MFCCs, and pause durations.
3. **Temporal Grounding**: Robust-centred per-word deltas flagged against perceptual floors and MAD thresholds (`max(floor, 3 x 1.4826 x MAD)`). Consecutive words merge into grounded time regions.
4. **Causal Explanation**: Observed vs reference values, delta, threshold, and human-readable explanation (`explain()`).
5. **Calibrated Rubric**: 5 dimensions (pace, pausing, expressiveness, volume, clarity) + overall score combining weighted mean and worst dimension.

## Key Project Documents
- Architecture: [`docs/ARCHITECTURE.md`](file:///c:/Personal/IIT%20MANDI/docs/ARCHITECTURE.md)
- Problem Statement: [`docs/problem_statement.md`](file:///c:/Personal/IIT%20MANDI/docs/problem_statement.md)
- Edge Cases: [`docs/edge_cases.md`](file:///c:/Personal/IIT%20MANDI/docs/edge_cases.md)
