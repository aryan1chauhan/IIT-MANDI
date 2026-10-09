"""Import Audacity label tracks and generate SpeechLens label JSONs and MMS_FA alignments.

Usage:
    python scripts/import_audacity_labels.py [--data data/real] [--align]

This script:
1. Discovers all audio takes (*.wav, *.flac, *.mp3) and Audacity label files (*.labels.txt).
2. Generates MMS_FA forced alignments for any take or baseline missing an alignment.
3. Converts each <take>.labels.txt (tab-separated: start s, end s, label) into the canonical
   SpeechLens ground-truth label schema (matching data/labels/*.json).
4. Handles pause flaws (long_pause, missing_pause) and segment/acoustic flaws (rushed,
   dragging, monotone, erratic_pitch, volume_drop, volume_spike, muffled) per folder convention.
5. Emits clean ground truth (empty regions) for baseline and clean2 takes (or missing label files).
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speechlens.align import align_words
from speechlens.text import load_alignment, normalize_words, save_alignment

# Supported flaw types in SpeechLens rubric
FLAW_TYPES = {
    "rushed", "dragging", "long_pause", "missing_pause",
    "monotone", "erratic_pitch", "volume_drop", "volume_spike", "muffled"
}


def find_audio_files(directory: Path) -> list[Path]:
    """Find all wav/flac/mp3 files in directory recursively."""
    exts = {".wav", ".flac", ".mp3"}
    return sorted([p for p in directory.rglob("*") if p.suffix.lower() in exts])


def get_speaker_id(stem: str) -> str:
    """Extract speaker ID from file stem (e.g., 'real01__rushed_s4' -> 'real01')."""
    return stem.split("__")[0] if "__" in stem else stem.split("_")[0]


def is_baseline_take(stem: str) -> bool:
    """Check if stem represents a baseline take."""
    return stem.endswith("__baseline") or stem.endswith("_baseline") or ("__" not in stem and not stem.endswith("clean2"))


def is_clean2_take(stem: str) -> bool:
    """Check if stem represents a clean2 take."""
    return "__clean2" in stem or "_clean2" in stem


def parse_audacity_labels(label_file: Path) -> list[dict]:
    """Parse Audacity label file: start_s <tab> end_s <tab> label."""
    if not label_file.exists():
        return []
    records = []
    for line_idx, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\t+|\s{2,}", line)
        if len(parts) < 3:
            # Fallback for single space delimiter if tabs were converted
            parts = line.split(maxsplit=2)
        if len(parts) < 3:
            print(f"Warning: Skipping malformed line {line_idx} in {label_file}: {line!r}")
            continue
        try:
            start_s = float(parts[0])
            end_s = float(parts[1])
        except ValueError:
            print(f"Warning: Invalid timestamps on line {line_idx} in {label_file}: {line!r}")
            continue
        raw_label = parts[2].strip().lower().replace(" ", "_")
        # Match against known flaw types or strip suffixes if present
        matched_flaw = raw_label
        for ft in FLAW_TYPES:
            if raw_label == ft or raw_label.startswith(ft):
                matched_flaw = ft
                break
        records.append({
            "start": start_s,
            "end": end_s,
            "type": matched_flaw,
        })
    return records


def find_overlapping_words(words: list[dict], start_s: float, end_s: float) -> list[int]:
    """Find indices of words overlapping with [start_s, end_s]."""
    indices = [
        k for k, w in enumerate(words)
        if max(w["start"], start_s) < min(w["end"], end_s)
    ]
    if indices:
        return indices
    # Fallback: check midpoint overlap
    mid_indices = [
        k for k, w in enumerate(words)
        if start_s <= (w["start"] + w["end"]) / 2.0 <= end_s
    ]
    if mid_indices:
        return mid_indices
    # Fallback: nearest word to interval center
    midpoint = (start_s + end_s) / 2.0
    nearest = min(range(len(words)), key=lambda k: abs((words[k]["start"] + words[k]["end"]) / 2.0 - midpoint))
    return [nearest]


def find_pause_boundary(words: list[dict], start_s: float, end_s: float) -> int:
    """Find word index i (i >= 1) where pause occurs between word i-1 and word i."""
    midpoint = (start_s + end_s) / 2.0
    # Search for gap containing or closest to midpoint
    best_i = 1
    best_dist = float("inf")
    for i in range(1, len(words)):
        prev_end = words[i - 1]["end"]
        next_start = words[i]["start"]
        # If interval falls nicely between words
        if prev_end <= midpoint <= next_start:
            return i
        gap_center = (prev_end + next_start) / 2.0
        dist = abs(gap_center - midpoint)
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def convert_labels_to_schema(
    take_id: str,
    baseline_id: str,
    raw_labels: list[dict],
    words_p: list[dict],
    words_b: Optional[list[dict]] = None,
) -> dict:
    """Convert parsed Audacity labels into canonical SpeechLens label JSON schema."""
    kind = "clean2" if is_clean2_take(take_id) else ("baseline" if is_baseline_take(take_id) else "single")
    regions = []

    for item in raw_labels:
        flaw_type = item["type"]
        s_raw, e_raw = item["start"], item["end"]

        if flaw_type == "long_pause":
            # Folder convention for long_pause:
            # word_start = i, word_end = i, start = prev_end, end = next_start
            i = find_pause_boundary(words_p, s_raw, e_raw)
            p_start = words_p[i - 1]["end"] if i > 0 else words_p[i]["start"]
            p_end = words_p[i]["start"]
            b_start = words_b[i - 1]["end"] if (words_b and i > 0) else (words_b[i]["start"] if words_b else p_start)
            b_end = words_b[i]["start"] if words_b else p_end

            regions.append({
                "type": "long_pause",
                "severity": 1,
                "start": round(p_start, 4),
                "end": round(p_end, 4),
                "word_start": i,
                "word_end": i,
                "base_start": round(b_start, 4),
                "base_end": round(b_end, 4),
                "raw_start": round(s_raw, 4),
                "raw_end": round(e_raw, 4),
            })

        elif flaw_type == "missing_pause":
            # Missing pause: participant rushed through pause present in baseline.
            # In SpeechLens convention, detected at word boundary i:
            # start = words_p[i-1]["end"], end = words_p[i]["start"] + 0.05
            cand_words = find_overlapping_words(words_p, s_raw, e_raw)
            w0, w1 = cand_words[0], cand_words[-1]

            # Find boundaries in [w0+1, w1] where baseline had a pause (gap >= 0.25s)
            flagged_boundaries = []
            if words_b and len(words_b) == len(words_p):
                for i in range(max(1, w0), min(len(words_p), w1 + 1)):
                    gap_b = words_b[i]["start"] - words_b[i - 1]["end"]
                    if gap_b >= 0.25:
                        flagged_boundaries.append(i)

            if not flagged_boundaries:
                # Fallback to closest boundary to label center
                flagged_boundaries = [find_pause_boundary(words_p, s_raw, e_raw)]

            for i in flagged_boundaries:
                # We add + 0.05s to the end timestamp to give the zero-length boundary a 
                # non-zero temporal footprint, matching analyze.py's convention for IoU math.
                p_start = words_p[i - 1]["end"] if i > 0 else words_p[i]["start"]
                p_end = max(words_p[i]["start"], p_start + 0.05)
                b_start = words_b[i - 1]["end"] if (words_b and i > 0) else p_start
                b_end = words_b[i]["start"] if words_b else p_end

                regions.append({
                    "type": "missing_pause",
                    "severity": 1,
                    "start": round(p_start, 4),
                    "end": round(p_end, 4),
                    "word_start": i,
                    "word_end": i,
                    "base_start": round(b_start, 4),
                    "base_end": round(b_end, 4),
                    # For raw timestamps, we synthesize a point label at the predicted boundary,
                    # since the raw Audacity label spans the entire phrase and would break IoU.
                    "raw_start": round(p_start, 4),
                    "raw_end": round(p_end, 4),
                })

        else:
            # Segment/acoustic flaws: rushed, dragging, monotone, erratic_pitch, volume_drop, volume_spike, muffled
            overlap = find_overlapping_words(words_p, s_raw, e_raw)
            w_start = overlap[0]
            w_end = overlap[-1]
            p_start = words_p[w_start]["start"]
            p_end = words_p[w_end]["end"]
            b_start = words_b[w_start]["start"] if words_b else p_start
            b_end = words_b[w_end]["end"] if words_b else p_end

            regions.append({
                "type": flaw_type,
                "severity": 1,
                "start": round(p_start, 4),
                "end": round(p_end, 4),
                "word_start": w_start,
                "word_end": w_end,
                "base_start": round(b_start, 4),
                "base_end": round(b_end, 4),
                "raw_start": round(s_raw, 4),
                "raw_end": round(e_raw, 4),
            })

    # Sort regions by start timestamp
    regions.sort(key=lambda r: r["start"])

    return {
        "id": take_id,
        "baseline": baseline_id,
        "kind": kind,
        "regions": regions,
        "words": [
            {"word": w["word"], "start": round(float(w["start"]), 4), "end": round(float(w["end"]), 4)}
            for w in words_p
        ],
    }


def ensure_alignment(
    audio_path: Path,
    transcript: str,
    alignment_path: Path,
    force_align: bool = False,
    device: str = "cpu",
) -> list[dict]:
    """Load alignment from cache or compute via MMS_FA."""
    if alignment_path.exists() and not force_align:
        return load_alignment(alignment_path)
    print(f"Aligning {audio_path.name} with MMS_FA...", flush=True)
    words = align_words(str(audio_path), transcript, device=device)
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    save_alignment(alignment_path, words)
    return words


def main():
    parser = argparse.ArgumentParser(description="Import Audacity labels and generate SpeechLens ground truth JSONs.")
    parser.add_argument("--data", default="data/real", help="Directory containing real voice recordings")
    parser.add_argument("--transcript", default="data/transcripts/real_passage.txt",
                        help="Path to reference transcript text")
    parser.add_argument("--out-labels", default=None, help="Output directory for label JSONs (defaults to <data>/labels)")
    parser.add_argument("--align", action="store_true", help="Run MMS_FA forced alignment on missing files")
    parser.add_argument("--refresh-align", action="store_true", help="Recompute all MMS_FA forced alignments")
    parser.add_argument("--device", default="cpu", help="Device for MMS_FA (cpu or cuda)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        # Fall back to checking data if data/real doesn't exist
        alt_data = Path("data")
        if alt_data.exists():
            data_dir = alt_data
        else:
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    out_labels_dir = Path(args.out_labels) if args.out_labels else (data_dir / "labels")
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    alignments_dir = data_dir / "alignments"
    alignments_dir.mkdir(parents=True, exist_ok=True)

    transcript_file = Path(args.transcript)
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript file {transcript_file} not found.")
    transcript = transcript_file.read_text(encoding="utf-8")

    audio_files = find_audio_files(data_dir)
    if not audio_files:
        print(f"No audio files found in {data_dir}.")
        return

    # Categorize into speakers and find baselines
    speaker_baselines = {}
    takes = []

    for af in audio_files:
        stem = af.stem
        spk = get_speaker_id(stem)
        if is_baseline_take(stem):
            speaker_baselines[spk] = af
        else:
            takes.append(af)

    print(f"Found {len(audio_files)} audio files across {len(set(get_speaker_id(f.stem) for f in audio_files))} speakers.")
    print(f"Baselines: {list(speaker_baselines.keys())}")
    print(f"Takes to process: {len(takes)}")

    # 1. Align all baselines
    baseline_words_cache = {}
    for spk, base_path in speaker_baselines.items():
        base_align_path = alignments_dir / f"{base_path.stem}.json"
        wb = ensure_alignment(base_path, transcript, base_align_path,
                              force_align=args.refresh_align, device=args.device)
        baseline_words_cache[spk] = (base_path.stem, wb)

    # 2. Process all takes
    processed_count = 0
    for take_path in takes:
        stem = take_path.stem
        spk = get_speaker_id(stem)
        if spk not in baseline_words_cache:
            print(f"Warning: No baseline found for speaker '{spk}' (take: {stem}). Skipping.")
            continue
        base_stem, words_b = baseline_words_cache[spk]

        # Forced alignment for take
        take_align_path = alignments_dir / f"{stem}.json"
        words_p = ensure_alignment(take_path, transcript, take_align_path,
                                   force_align=args.refresh_align, device=args.device)

        # Look for Audacity label file
        # Check <take>.labels.txt in same directory, or data_dir/<take>.labels.txt
        label_file = take_path.with_name(f"{stem}.labels.txt")
        if not label_file.exists():
            label_file = data_dir / f"{stem}.labels.txt"
        if not label_file.exists():
            label_file = data_dir / "labels" / f"{stem}.labels.txt"

        raw_labels = parse_audacity_labels(label_file) if label_file.exists() else []
        if not label_file.exists() and not is_clean2_take(stem):
            print(f"Note: No label file for {stem}.labels.txt (assuming 0 flaws).")

        label_data = convert_labels_to_schema(
            take_id=stem,
            baseline_id=base_stem,
            raw_labels=raw_labels,
            words_p=words_p,
            words_b=words_b,
        )

        out_json_path = out_labels_dir / f"{stem}.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=1)

        processed_count += 1
        n_regions = len(label_data["regions"])
        print(f"Wrote {out_json_path.name:36s} (kind={label_data['kind']}, regions={n_regions})")

    print(f"\nSuccessfully converted and saved {processed_count} label files to {out_labels_dir}.")


if __name__ == "__main__":
    main()
