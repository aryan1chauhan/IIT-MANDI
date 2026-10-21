"""Prepare VCTK cross-speaker natural speech audit dataset.

(1) Identifies shared utterances read with identical text across speakers.
(2) Concatenates shared utterances for 5 speakers with 0.3 s inter-sentence gap into data/vctk/<speaker>__passage.wav.
(3) Writes the shared transcript to data/vctk/transcript.txt and data/transcripts/vctk_passage.txt.
(4) Prints each clip's duration and the shared text.
"""
import argparse
import sys
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VCTK = None  # pass --vctk /path/to/VCTK-Corpus-0.92
TARGET_SR = 16000
GAP_SEC = 0.30


def find_shared_utterances(txt_dir: Path, speakers: list[str]) -> list[tuple[str, str]]:
    """Find utterances present across all given speakers with identical text."""
    speaker_utts = {}
    for spk in speakers:
        spk_dir = txt_dir / spk
        if not spk_dir.is_dir():
            raise FileNotFoundError(f"Speaker directory not found: {spk_dir}")
        spk_utts = {}
        for p in spk_dir.glob("*.txt"):
            utt_id = p.stem.split("_")[-1]
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            spk_utts[utt_id] = text
        speaker_utts[spk] = spk_utts

    # Intersect keys across all selected speakers
    common_keys = sorted(set.intersection(*[set(u.keys()) for u in speaker_utts.values()]))

    # Filter for identical text
    identical = []
    for k in common_keys:
        texts = [speaker_utts[spk][k] for spk in speakers]
        if len(set(texts)) == 1:
            identical.append((k, texts[0]))

    return identical


def get_audio_file(wav_dir: Path, speaker: str, utt_id: str) -> Path:
    """Find the flac/wav file for a given speaker and utterance (preferring mic1)."""
    spk_wav = wav_dir / speaker
    # Look for mic1 first, then any matching
    mic1 = spk_wav / f"{speaker}_{utt_id}_mic1.flac"
    if mic1.exists():
        return mic1
    candidates = list(spk_wav.glob(f"{speaker}_{utt_id}*.flac")) + list(spk_wav.glob(f"{speaker}_{utt_id}*.wav"))
    if not candidates:
        raise FileNotFoundError(f"Audio for {speaker} utterance {utt_id} not found in {spk_wav}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Prepare VCTK cross-speaker dataset.")
    parser.add_argument("--vctk", type=Path, default=DEFAULT_VCTK, help="Path to unzipped VCTK-Corpus-0.92")
    parser.add_argument("--num-speakers", type=int, default=5, help="Number of speakers to prepare")
    args = parser.parse_args()

    vctk_dir = args.vctk
    txt_dir = vctk_dir / "txt"
    wav_dir = vctk_dir / "wav48_silence_trimmed"

    if not txt_dir.exists() or not wav_dir.exists():
        sys.exit(f"Invalid VCTK directory: {vctk_dir} (expected txt/ and wav48_silence_trimmed/)")

    # Pick 5 standard speakers
    all_speakers = sorted([p.name for p in txt_dir.iterdir() if p.is_dir()])
    selected_speakers = all_speakers[: args.num_speakers]
    print(f"Selected {len(selected_speakers)} speakers: {selected_speakers}")

    # (1) Find shared identical utterances
    # Utterances 001-005 form the complete, cohesive 'Please call Stella' passage read by all speakers
    stella_keys = [f"{i:03d}" for i in range(1, 6)]
    shared = find_shared_utterances(txt_dir, selected_speakers)
    shared_stella = [(k, text) for k, text in shared if k in stella_keys]

    print("\nShared identical utterances found across selected speakers:")
    for k, text in shared_stella:
        print(f"  Utterance {k}: {text}")

    full_text = " ".join(text for _, text in shared_stella)

    # Output directories
    out_dir = ROOT / "data" / "vctk"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = ROOT / "data" / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Write transcript
    transcript_file = out_dir / "transcript.txt"
    transcript_file.write_text(full_text + "\n", encoding="utf-8")
    (transcripts_dir / "vctk_passage.txt").write_text(full_text + "\n", encoding="utf-8")

    gap = np.zeros(int(GAP_SEC * TARGET_SR), dtype=np.float32)

    # (2) Concatenate for each speaker
    print("\nConcatenating audio clips with 0.3s gap (resampled to 16 kHz):")
    durations = {}
    for spk in selected_speakers:
        pieces = []
        for i, (k, _) in enumerate(shared_stella):
            audio_path = get_audio_file(wav_dir, spk, k)
            y, sr = sf.read(str(audio_path), dtype="float32")
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            pieces.append(y)
            if i < len(shared_stella) - 1:
                pieces.append(gap)
        concat = np.concatenate(pieces, axis=0)
        out_wav = out_dir / f"{spk}__passage.wav"
        sf.write(str(out_wav), np.clip(concat, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
        dur = len(concat) / TARGET_SR
        durations[spk] = dur
        print(f"  {spk}__passage.wav: {dur:.2f} s -> {out_wav}")

    # (3) Print summary
    print("\n" + "=" * 70)
    print("VCTK PREPARE SUMMARY")
    print("=" * 70)
    print(f"Shared Utterance Numbers: {[k for k, _ in shared_stella]}")
    print("\nShared Text:")
    print(f"\"{full_text}\"")
    print("\nGenerated Audio Durations:")
    for spk, dur in durations.items():
        print(f"  {spk}: {dur:.2f} s")
    print(f"\nFiles written to: {out_dir}")


if __name__ == "__main__":
    main()
