import json
import re


def normalize_words(text: str) -> list[str]:
    """Lowercase a-z/apostrophe tokens. Numbers must be spelled out in the transcript."""
    text = text.lower().replace("\u2019", "'")
    return re.findall(r"[a-z']+", text)


def save_alignment(path, words):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"words": [{"word": w["word"], "start": round(float(w["start"]), 4),
                              "end": round(float(w["end"]), 4)} for w in words]}, f, indent=1)


def load_alignment(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["words"]
