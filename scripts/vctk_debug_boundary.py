"""VCTK boundary diagnostic: RMS energy between aligned word edges.

For p229 and p228 at boundary 003->004, prints the aligned end time of the
last word of utterance 003 ("bob"), the aligned start time of "we", and
RMS energy in 50 ms frames between them. Exports the gap ±0.5 s as
data/vctk/debug_<speaker>_b34.wav.

Does NOT modify speechlens/ or any threshold.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "vctk"
ALIGN_DIR = DATA_DIR / "alignments"
OUT_DIR = DATA_DIR  # exports go next to the passage WAVs

# Utterance 003 ends at word index 33 ("bob"), utterance 004 starts at word 34 ("we")
LAST_WORD_IDX = 33   # "bob"
FIRST_WORD_IDX = 34  # "we"
SPEAKERS = ["p229", "p228"]
FRAME_MS = 50


def main() -> None:
    rows: list[dict] = []

    for spk in SPEAKERS:
        align_path = ALIGN_DIR / f"{spk}.json"
        wav_path = DATA_DIR / f"{spk}__passage.wav"

        alignment = json.loads(align_path.read_text(encoding="utf-8"))
        words = alignment["words"]
        audio, sr = sf.read(str(wav_path), dtype="float32")

        bob = words[LAST_WORD_IDX]
        we = words[FIRST_WORD_IDX]
        assert bob["word"].lower() == "bob", f"Expected 'bob' at idx {LAST_WORD_IDX}, got {bob['word']}"
        assert we["word"].lower() == "we", f"Expected 'we' at idx {FIRST_WORD_IDX}, got {we['word']}"

        bob_end = bob["end"]
        we_start = we["start"]
        gap_s = we_start - bob_end

        # RMS in 50 ms frames within the gap
        frame_samples = int(sr * FRAME_MS / 1000)
        gap_start_sample = int(bob_end * sr)
        gap_end_sample = int(we_start * sr)
        gap_audio = audio[gap_start_sample:gap_end_sample]

        rms_frames: list[dict] = []
        for i in range(0, len(gap_audio) - frame_samples + 1, frame_samples):
            chunk = gap_audio[i : i + frame_samples]
            rms = float(np.sqrt(np.mean(chunk**2)))
            rms_db = 20 * np.log10(rms + 1e-10)
            t = bob_end + i / sr
            rms_frames.append({"t": round(t, 3), "rms": round(rms, 6), "rms_dB": round(rms_db, 1)})

        rows.append({
            "speaker": spk,
            "bob_end_s": round(bob_end, 3),
            "we_start_s": round(we_start, 3),
            "aligned_gap_s": round(gap_s, 3),
            "n_frames": len(rms_frames),
            "rms_frames": rms_frames,
        })

        # Export gap ±0.5 s
        margin = 0.5
        export_start = max(0, int((bob_end - margin) * sr))
        export_end = min(len(audio), int((we_start + margin) * sr))
        export_audio = audio[export_start:export_end]
        export_path = OUT_DIR / f"debug_{spk}_b34.wav"
        sf.write(str(export_path), export_audio, sr)
        print(f"Exported: {export_path} ({len(export_audio)/sr:.2f} s)")

    # Print table
    print(f"\n{'Speaker':<8} {'bob end (s)':>12} {'we start (s)':>13} {'aligned gap (s)':>16}")
    print("-" * 52)
    for r in rows:
        print(f"{r['speaker']:<8} {r['bob_end_s']:>12.3f} {r['we_start_s']:>13.3f} {r['aligned_gap_s']:>16.3f}")

    print(f"\nRMS energy in 50 ms frames between 'bob' end and 'we' start:")
    print(f"{'Speaker':<8} {'Frame t (s)':>12} {'RMS':>10} {'RMS (dB)':>10}")
    print("-" * 44)
    for r in rows:
        for f in r["rms_frames"]:
            print(f"{r['speaker']:<8} {f['t']:>12.3f} {f['rms']:>10.6f} {f['rms_dB']:>10.1f}")
        print()


if __name__ == "__main__":
    main()
