"""Analyse one participant recording against a baseline of the same text.

  python scripts/analyze_cli.py --participant p.wav --baseline b.wav --transcript t.txt --out report.json

Alignments are computed with MMS_FA unless you pass --participant-align / --baseline-align JSON files.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechlens.analyze import analyze  # noqa: E402
from speechlens.io import load_audio  # noqa: E402
from speechlens.text import load_alignment, save_alignment  # noqa: E402


def get_words(audio, transcript, align_json):
    if align_json:
        return load_alignment(align_json)
    from speechlens.align import align_words
    return align_words(audio, transcript)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--participant-align")
    ap.add_argument("--baseline-align")
    ap.add_argument("--out", default="report.json")
    a = ap.parse_args()
    text = Path(a.transcript).read_text(encoding="utf-8")
    wp = get_words(a.participant, text, a.participant_align)
    wb = get_words(a.baseline, text, a.baseline_align)
    if not a.participant_align:
        save_alignment(Path(a.participant).with_suffix(".align.json"), wp)
    if not a.baseline_align:
        save_alignment(Path(a.baseline).with_suffix(".align.json"), wb)
    res = analyze(load_audio(a.participant), wp, load_audio(a.baseline), wb)
    json.dump(res, open(a.out, "w"))
    print("Scores:", res["scores"])
    for r in res["regions"]:
        print(f"  [{r['start']:6.2f}-{r['end']:6.2f}s] {r['type']}: {r['explanation']}")


if __name__ == "__main__":
    main()
