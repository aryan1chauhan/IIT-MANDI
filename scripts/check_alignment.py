"""Check forced alignment between audio and transcript using MMS_FA."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speechlens.align import align_words
from speechlens.text import save_alignment, normalize_words


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/check_alignment.py <audio_path> <transcript_path> [out_json]")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    transcript_path = Path(sys.argv[2])
    transcript = transcript_path.read_text(encoding="utf-8")

    print(f"Audio file: {audio_path}")
    print(f"Transcript: {transcript_path}")

    words_in_transcript = normalize_words(transcript)
    print(f"Normalized word count: {len(words_in_transcript)}")

    print("\nRunning MMS_FA forced alignment...")
    aligned_words = align_words(str(audio_path), transcript)

    # Save to requested location and standard alignments directory
    stem = audio_path.stem
    out1 = Path(sys.argv[3]) if len(sys.argv) > 3 else audio_path.with_suffix(".wav.align.json")
    out2 = ROOT / "data" / "alignments" / f"{stem}.json"
    out2.parent.mkdir(parents=True, exist_ok=True)

    save_alignment(out1, aligned_words)
    save_alignment(out2, aligned_words)
    print(f"Saved alignment to: {out1}")
    print(f"Saved alignment to: {out2}")

    # Summary statistics
    durations = [w["end"] - w["start"] for w in aligned_words]
    total_time = aligned_words[-1]["end"] - aligned_words[0]["start"] if aligned_words else 0
    mean_dur = sum(durations) / len(durations) if durations else 0
    wps = len(aligned_words) / total_time if total_time > 0 else 0

    print(f"\n--- Alignment Diagnostics ---")
    print(f"Aligned words: {len(aligned_words)}")
    print(f"Span: {aligned_words[0]['start']:.3f} s -> {aligned_words[-1]['end']:.3f} s (total: {total_time:.2f} s)")
    print(f"Mean word duration: {mean_dur:.3f} s (min: {min(durations):.3f} s, max: {max(durations):.3f} s)")
    print(f"Speaking rate: {wps:.2f} words/s")

    # Edge-case checks
    zero_dur = [w for w in aligned_words if w["end"] <= w["start"]]
    if zero_dur:
        print(f"WARNING: {len(zero_dur)} zero/negative duration words detected!")
    else:
        print("Edge check: No zero-duration words (clean).")

    print("\nFirst 10 words:")
    for w in aligned_words[:10]:
        print(f"  {w['start']:6.3f} - {w['end']:6.3f} s : {w['word']}")

    if len(aligned_words) > 10:
        print("...")
        print("Last 5 words:")
        for w in aligned_words[-5:]:
            print(f"  {w['start']:6.3f} - {w['end']:6.3f} s : {w['word']}")


if __name__ == "__main__":
    main()
