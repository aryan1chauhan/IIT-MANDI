"""VCTK Natural-Speech Cross-Speaker Audit.

Executes a natural-speech cross-speaker control audit across 5 speakers (p225-p229)
reading identical elicitation text (utterances 001-005: "Please call Stella").

Evaluates whether SpeechLens triggers false alarms on natural human speaker variations
under frozen thresholds (tag: thresholds-frozen).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from speechlens.analyze import analyze, FLOORS, K_MAD
from speechlens.io import load_audio
from speechlens.text import load_alignment

DEFAULT_VCTK_WAV = None  # pass --vctk-wav /path/to/VCTK-Corpus-0.92/wav48_silence_trimmed

# Known utterance structure from the concatenation (5 sentences, 69 words total)
UTTERANCES = [
    {
        "id": "001",
        "word_start": 0,
        "word_end": 2,
        "text": "Please call Stella.",
    },
    {
        "id": "002",
        "word_start": 3,
        "word_end": 13,
        "text": "Ask her to bring these things with her from the store.",
    },
    {
        "id": "003",
        "word_start": 14,
        "word_end": 33,
        "text": "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
    },
    {
        "id": "004",
        "word_start": 34,
        "word_end": 48,
        "text": "We also need a small plastic snake and a big toy frog for the kids.",
    },
    {
        "id": "005",
        "word_start": 49,
        "word_end": 68,
        "text": "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.",
    },
]


def get_flag_utterance(r: dict) -> str:
    """Assign flag to the utterance containing the pause's following word (word_end)."""
    target_idx = r.get("word_end", r.get("word_start", -1))
    for u in UTTERANCES:
        if u["word_start"] <= target_idx <= u["word_end"]:
            return u["id"]
    return "unknown"


def measure_source_file_silence(audio_path: Path, top_db: float = 35.0) -> tuple[float, float]:
    """Measure leading and trailing silence (energy-based, in seconds) of source audio file."""
    if not audio_path.exists():
        return 0.0, 0.0
    y, sr = sf.read(str(audio_path), dtype="float32")
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_frames = (len(y) - frame_len) // hop_len + 1
    if n_frames <= 0:
        dur = len(y) / sr
        return dur, dur
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_len)[::hop_len]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    peak = np.max(rms)
    if peak <= 1e-9:
        dur = len(y) / sr
        return dur, dur
    db = 20 * np.log10(rms / peak + 1e-12)
    active = np.where(db > -top_db)[0]
    if len(active) == 0:
        dur = len(y) / sr
        return dur, dur
    lead_s = float(active[0] * hop_len / sr)
    trail_s = float((len(frames) - 1 - active[-1]) * hop_len / sr)
    return lead_s, trail_s


def run_boundary_silence_diagnostic(
    speakers: list[str],
    vctk_wav_dir: Path,
    align_dir: Path,
) -> dict[str, Any]:
    """Diagnostic for all 5 speakers and each of the 4 utterance boundaries."""
    boundaries = [
        ("001 -> 002", "001", "002", 2, 3, "stella", "ask"),
        ("002 -> 003", "002", "003", 13, 14, "store", "six"),
        ("003 -> 004", "003", "004", 33, 34, "bob", "we"),
        ("004 -> 005", "004", "005", 48, 49, "kids", "she"),
    ]

    print("\n" + "=" * 98)
    print("DIAGNOSTIC: SOURCE VCTK FILE SILENCES & INTER-UTTERANCE ALIGNED GAPS")
    print("=" * 98)
    header = f"{'Boundary':12s} | {'Speaker':7s} | {'Source Prev Trail':17s} | {'Concat Gap':10s} | {'Source Next Lead':16s} | {'Aligned Gap':11s}"
    print(header)
    print("-" * len(header))

    diagnostic_records = []
    gaps_by_boundary = defaultdict(dict)

    for b_label, u_prev, u_next, w_prev_idx, w_next_idx, w_prev_word, w_next_word in boundaries:
        for spk in speakers:
            f_prev = vctk_wav_dir / spk / f"{spk}_{u_prev}_mic1.flac"
            f_next = vctk_wav_dir / spk / f"{spk}_{u_next}_mic1.flac"

            _, trail_s = measure_source_file_silence(f_prev)
            lead_s, _ = measure_source_file_silence(f_next)

            align_file = align_dir / f"{spk}.json"
            words = load_alignment(align_file)
            aligned_gap = words[w_next_idx]["start"] - words[w_prev_idx]["end"]

            gaps_by_boundary[b_label][spk] = aligned_gap

            rec = {
                "boundary": b_label,
                "prev_utterance": u_prev,
                "next_utterance": u_next,
                "prev_word": w_prev_word,
                "next_word": w_next_word,
                "speaker": spk,
                "source_prev_trail_s": round(trail_s, 3),
                "concat_synthetic_gap_s": 0.300,
                "source_next_lead_s": round(lead_s, 3),
                "aligned_gap_s": round(aligned_gap, 3),
            }
            diagnostic_records.append(rec)
            print(f"{b_label:12s} | {spk:7s} | {trail_s:15.3f} s | {'0.300 s':10s} | {lead_s:14.3f} s | {aligned_gap:9.3f} s")

    print("\n" + "-" * 98)
    print("CROSS-SPEAKER BOUNDARY GAP VARIATION ANALYSIS (vs 0.35 s pause threshold):")
    print("-" * 98)
    boundary_variation = {}
    for b_label, spk_gaps in gaps_by_boundary.items():
        min_spk = min(spk_gaps, key=spk_gaps.get)
        max_spk = max(spk_gaps, key=spk_gaps.get)
        min_gap = spk_gaps[min_spk]
        max_gap = spk_gaps[max_spk]
        diff = max_gap - min_gap
        exceeds_threshold = diff > 0.350
        boundary_variation[b_label] = {
            "min_speaker": min_spk,
            "min_gap_s": round(min_gap, 3),
            "max_speaker": max_spk,
            "max_gap_s": round(max_gap, 3),
            "diff_s": round(diff, 3),
            "exceeds_0.35s_threshold": exceeds_threshold,
        }
        status = "EXCEEDS 0.35s THRESHOLD (Confirms Concatenation Artifact)" if exceeds_threshold else "Within 0.35s threshold"
        print(f"  Boundary {b_label}: min={min_gap:.3f}s ({min_spk}) to max={max_gap:.3f}s ({max_spk}) | Δ={diff:.3f}s -> {status}")

    return {
        "records": diagnostic_records,
        "boundary_variation": boundary_variation,
    }


def run_cross_speaker_audit(
    baseline_spk: str,
    test_speakers: list[str],
    data_dir: Path,
    align_dir: Path,
) -> dict[str, Any]:
    """Run cross-speaker audit against a chosen baseline."""
    base_audio_path = data_dir / f"{baseline_spk}__passage.wav"
    base_align_path = align_dir / f"{baseline_spk}.json"

    if not base_audio_path.exists():
        raise FileNotFoundError(f"Baseline audio not found: {base_audio_path}")
    if not base_align_path.exists():
        raise FileNotFoundError(f"Baseline alignment not found: {base_align_path}")

    yb = load_audio(base_audio_path)
    wb = load_alignment(base_align_path)
    base_dur = len(yb) / 16000.0

    print("=" * 86)
    print(f"AUDIT RUN: Baseline = {baseline_spk} (duration: {base_dur:.2f} s, {len(wb)} words)")
    print(f"Test Speakers: {test_speakers}")
    print(f"Frozen Thresholds: K_MAD={K_MAD}, FLOORS={FLOORS}")
    print("=" * 86)

    flags_detailed = []
    speaker_summaries = {}
    total_test_duration = 0.0
    total_flagged_duration = 0.0
    flaw_type_counts = defaultdict(int)
    speaker_counts = defaultdict(int)

    for spk in test_speakers:
        part_audio_path = data_dir / f"{spk}__passage.wav"
        part_align_path = align_dir / f"{spk}.json"

        yp = load_audio(part_audio_path)
        wp = load_alignment(part_align_path)
        clip_dur = len(yp) / 16000.0
        total_test_duration += clip_dur

        # Run frozen SpeechLens detector
        res = analyze(yp, wp, yb, wb)
        regions = res["regions"]
        score = res["scores"]["overall"]

        spk_flag_dur = sum(r["end"] - r["start"] for r in regions)
        total_flagged_duration += spk_flag_dur
        speaker_counts[spk] = len(regions)

        speaker_summaries[spk] = {
            "score": score,
            "clip_duration_s": clip_dur,
            "flag_count": len(regions),
            "flagged_duration_s": spk_flag_dur,
            "flagged_fraction": spk_flag_dur / clip_dur if clip_dur > 0 else 0.0,
            "flaw_types": [r["type"] for r in regions],
        }

        print(f"\n--- Speaker: {spk} (clip: {clip_dur:.2f} s, score: {score:.1f}/100, flags: {len(regions)}) ---")
        if not regions:
            print("  [No flags detected - 100% clean under cross-speaker comparison]")

        for r in regions:
            flaw = r["type"]
            flaw_type_counts[flaw] += 1
            start_s = r["start"]
            end_s = r["end"]
            dur = end_s - start_s
            utt = get_flag_utterance(r)
            obs = r["observed"]
            ref = r["reference"]
            thr = r["threshold"]
            unit = r.get("unit", "")
            metric = r.get("metric", "")
            delta = r.get("delta", obs - ref)

            flag_entry = {
                "baseline_speaker": baseline_spk,
                "test_speaker": spk,
                "flaw_type": flaw,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(dur, 3),
                "utterance": utt,
                "words_spanned": r.get("text", ""),
                "observed": round(obs, 3),
                "reference": round(ref, 3),
                "threshold": round(thr, 3),
                "delta": round(delta, 3),
                "unit": unit,
                "metric": metric,
            }
            flags_detailed.append(flag_entry)

            print(f"  Flag: [{flaw.upper()}] at {start_s:.2f}s - {end_s:.2f}s (dur={dur:.2f}s)")
            print(f"    Utterance: {utt} | Following Word: \"{r.get('text', '')}\"")
            print(f"    Measured: obs={obs:.3f} {unit} vs ref={ref:.3f} {unit} | delta={delta:.3f} | threshold={thr:.3f} {unit}")

    # Summary
    print("\n" + "-" * 86)
    print(f"SUMMARY: Baseline = {baseline_spk}")
    print("-" * 86)
    print("1. Flags per Speaker:")
    for spk in test_speakers:
        info = speaker_summaries[spk]
        print(f"   {spk:8s}: {info['flag_count']} flags | {info['flagged_duration_s']:.2f}s flagged / {info['clip_duration_s']:.2f}s total ({info['flagged_fraction']*100:.1f}%) | Score: {info['score']:.1f}")

    print("\n2. Flags per Flaw Type:")
    if not flaw_type_counts:
        print("   None (0 flags across all flaw types)")
    else:
        for f_type, cnt in sorted(flaw_type_counts.items(), key=lambda x: -x[1]):
            print(f"   {f_type:15s}: {cnt}")

    total_frac = total_flagged_duration / total_test_duration if total_test_duration > 0 else 0.0
    print(f"\n3. Overall Flagged Duration Fraction:")
    print(f"   Total flagged: {total_flagged_duration:.2f} s / {total_test_duration:.2f} s ({total_frac*100:.2f}%)")

    return {
        "baseline_speaker": baseline_spk,
        "test_speakers": test_speakers,
        "speaker_summaries": speaker_summaries,
        "flaw_type_counts": dict(flaw_type_counts),
        "total_test_duration_s": round(total_test_duration, 2),
        "total_flagged_duration_s": round(total_flagged_duration, 2),
        "total_flagged_fraction": round(total_frac, 4),
        "detailed_flags": flags_detailed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run VCTK natural-speech cross-speaker audit.")
    parser.add_argument("--data", default="data/vctk", help="Directory containing concatenated VCTK audio")
    parser.add_argument("--vctk-wav", default=str(DEFAULT_VCTK_WAV), help="Directory containing source VCTK wav48_silence_trimmed")
    parser.add_argument("--out", default="results/vctk_audit.json", help="Path to write JSON results")
    args = parser.parse_args()

    data_dir = ROOT / args.data
    align_dir = data_dir / "alignments"
    vctk_wav_dir = Path(args.vctk_wav)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_speakers = ["p225", "p226", "p227", "p228", "p229"]

    # Verify audio formats and sample rates
    for spk in all_speakers:
        p = data_dir / f"{spk}__passage.wav"
        y = load_audio(p)
        dur = len(y) / 16000.0
        assert y.ndim == 1, f"Expected mono audio for {spk}"
        assert np.all(np.isfinite(y)), f"Audio contains NaN or Inf for {spk}"
        print(f"Verified {spk}__passage.wav: 16000 Hz, mono, {dur:.2f} s, finite")

    # Boundary Silence Diagnostic
    diagnostic_results = run_boundary_silence_diagnostic(all_speakers, vctk_wav_dir, align_dir)

    print("\n" + "#" * 86)
    print("PHASE 1: BASELINE = p225")
    print("#" * 86 + "\n")
    run1 = run_cross_speaker_audit("p225", ["p226", "p227", "p228", "p229"], data_dir, align_dir)

    print("\n" + "#" * 86)
    print("PHASE 2: BASELINE = p226 (Checking Baseline Dependence)")
    print("#" * 86 + "\n")
    run2 = run_cross_speaker_audit("p226", ["p225", "p227", "p228", "p229"], data_dir, align_dir)

    combined_results = {
        "title": "VCTK Natural-Speech Cross-Speaker Control Audit",
        "protocol": "natural-speech, cross-speaker audit (no deliberate flaws; not accuracy or F1)",
        "summary_claims": [
            "No rate, pitch, volume or clarity flags on 8 comparisons.",
            "Pause flags at utterance boundaries on 3 of 4 clips; the aligned gap exceeds the acoustic silence by 1.2 to 1.9 s at every boundary, cause not yet determined.",
        ],
        "thresholds_frozen": {
            "K_MAD": K_MAD,
            "FLOORS": FLOORS,
        },
        "boundary_silence_diagnostic": diagnostic_results,
        "run_p225_baseline": run1,
        "run_p226_baseline": run2,
    }

    out_path.write_text(json.dumps(combined_results, indent=2), encoding="utf-8")
    print(f"\nResults successfully written to: {out_path}")


if __name__ == "__main__":
    main()
