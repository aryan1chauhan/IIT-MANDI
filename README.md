# SpeechLens: contrastive speech analytics with temporal flaw grounding (Track C)

Compares a participant's delivery with a baseline recording of the **same text**, finds the exact
time regions where delivery deviates, explains each deviation numerically, and scores it against a rubric.

## How it works
1. **Forced alignment** (`speechlens/align.py`, torchaudio MMS_FA): transcript -> word start/end times.
   Words are exact anchors between baseline and participant timelines (no DTW needed).
2. **Speaker-normalised features** (`speechlens/features.py`): F0 in semitones re the speaker's median,
   energy in dB re the speaker's median speech level, STFT high-band (>3 kHz) power share, CMN-MFCCs, pauses.
3. **Temporal grounding** (`speechlens/analyze.py`): per-word deltas (rate, pause, pitch range, level,
   clarity) are robust-centred (median removed -> speaker/gain/tempo agnostic) and flagged when they exceed
   `max(perceptual floor, 3 x 1.4826 x MAD)`. Consecutive words merge into regions with start/end timestamps.
4. **Causal explanation**: each region carries observed vs reference value, delta, threshold, and a
   plain-language reason (`explain()`).
5. **Rubric score** (0-100): pace, pausing, expressiveness, volume, clarity;
   `dim = 100 * exp(-15 * sum(duration_fraction * (0.3 + 0.7 * severity)))`;
   `overall = 0.5 * weighted mean + 0.5 * worst dimension`.

## Dataset construction (`scripts/build_dataset.py`)
Baselines (one human public-domain reading from LibriVox, one synthetic speech rendered via eSpeak TTS; aligned with `align.py`) are mirrored with
injected flaws applied to whole-word spans, so labels are exact and re-timed word alignments come for free.

- 6 flaws (rushed, dragging, monotone, long pause, volume drop, muffled) x 4 severities
  (1 = almost perfect ... 4 = botched), mixed multi-flaw files, and negative controls (gain change, faint noise).
- Rushed and dragging spans use ffmpeg's pitch-preserving `atempo` filter. The prior librosa phase-vocoder
  stretch created an energy-envelope artifact that could be mistaken for `volume_drop`; this is a ground-truth
  correction, not threshold tuning.
- Seeds come from CRC32 of (speech, flaw, severity): fully reproducible across machines.
- Licensing: avoid CC BY-NC-ND material (TED) for the published dataset, since modified copies are derivatives.

## Prerequisites
- **Python**: 3.11 or 3.12
- **FFmpeg**: Required for audio processing (`atempo` filter). Must be on system `PATH`.
- **eSpeak-NG**: Required for synthetic baseline fixtures and automated unit tests (`espeak-ng.exe`).

## Setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu   # alignment only
```

## Run
```powershell
# 1. put speech audio in data\baseline\<id>.wav and the transcript in data\transcripts\<id>.txt
#    (spell out numbers). Align once:
python -c "from speechlens.align import align_words; from speechlens.text import save_alignment; import pathlib; i='<id>'; save_alignment(f'data/alignments/{i}.json', align_words(f'data/baseline/{i}.wav', pathlib.Path(f'data/transcripts/{i}.txt').read_text()))"
# 2. build the contrastive dataset
python scripts/build_dataset.py
# 3. evaluate detection against injected labels (refresh participant MMS_FA sidecars after rebuilding)
python scripts/run_eval.py --realign --realign-baseline --refresh-realignments --out eval\results_matched.json
# 4. analyse any participant recording
python scripts/analyze_cli.py --participant p.wav --baseline data\baseline\<id>.wav --transcript data\transcripts\<id>.txt
pytest -q
```
Synthetic test fixture (needs `espeak-ng`): `python scripts/make_synthetic_baseline.py --out data`.

## Dataset Download
Because `.wav`, `.flac`, and large audio binaries are excluded from Git repository tracking, the full audio dataset (including baseline and flawed recordings) is hosted on Google Drive:
- **Audio Dataset (Google Drive)**: [speechlens_data.zip](https://drive.google.com/file/d/1xV9NBX-7LPChUpnVPBt-Y0-VsS1yiho-/view?usp=sharing)
- **SHA-256 Checksum**: `370605D00D82A0C5113B7968822AC3D95B82D787F1B78C5D546FD1266F379F78`
*(To run tests locally without downloading the audio package, `pytest` generates synthetic fixtures automatically via eSpeak-NG).*

## Results on the matched-provenance synthetic benchmark (pipeline validation)
Full rebuilt 30-file evaluation ([`eval/results_matched.json`](eval/results_matched.json)):
- At strict IoU >= 0.5: recall 0.72 (26/36), precision 0.84 (26/31), F1 0.78.
- At IoU >= 0.3: recall 0.81 (29/36), precision 0.94 (29/31), F1 0.87. (The 2 false positives at IoU 0.3 are both `long_pause`; the 3 additional FPs at IoU 0.5 are boundary offsets, not spurious detections.)
- Strict recall by injected severity 1-4: 0.17, 0.60, 0.89, 1.00. The severity table uses the same IoU >= 0.5 criterion as the headline.
- No monotone false positives at IoU >= 0.3 and no volume-drop false positives after replacing the phase-vocoder injector.
- Score severity correlations remain negative for all six flaw types (rho -0.95 or -1.00).

## Limitations (be upfront in the report)
- Small benchmark scale: n=2 clips, one human voice, synthetic01 is TTS.
- Dataset rebuilt post-evaluation: The synthetic dataset was rebuilt once after the initial evaluation to fix an injector artifact (replacing librosa phase-vocoder pitch stretching with ffmpeg `atempo`, which eliminated spurious energy-envelope flaws). The flaw detector thresholds themselves remained strictly frozen.
- Injected flaws are generated by the same signal-processing family the detector measures, so the
  synthetic score is optimistic. Add real self-recorded flawed readings as an independent test set.
- Pitch range uses a 3-word window, so boundaries of pitch regions can leak by about one word.
- Accuracy depends on alignment quality (MMS_FA word spans are about 20 ms resolution, CTC-tight).
- At 10 dB SNR, the real-speech `speech1` control has a monotone false positive; treat low-SNR pitch findings as a limitation.
- One baseline per text; delivery that is different but equally good (alternative phrasing/emphasis) may be flagged.
- Dashboard: not included yet (`analyze()` returns JSON with series, regions, scores for it).
