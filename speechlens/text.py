import json
import re


def normalize_words(text: str) -> list[str]:
    """Lowercase a-z/apostrophe tokens. Numbers must be spelled out in the transcript."""
    digits = re.findall(r"\d+", text)
    if digits:
        raise ValueError(
            f"Transcript contains digits ({digits[:3]}): numbers must be spelled out (e.g. 'twenty' not '20') for forced alignment."
        )
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    tokens = re.findall(r"[a-z']+", text)
    cleaned = [t.strip("'") for t in tokens]
    return [t for t in cleaned if t]


def save_alignment(path, words):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"words": [{"word": w["word"], "start": round(float(w["start"]), 4),
                              "end": round(float(w["end"]), 4)} for w in words]}, f, indent=1)


def load_alignment(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["words"]
