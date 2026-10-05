import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
words = d["words"] if isinstance(d, dict) else d
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for w in words:
        f.write(f"{w['start']:.3f}\t{w['end']:.3f}\t{w['word']}\n")
