"""Tests for Audacity label import and real-set evaluation path."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from import_audacity_labels import (
    convert_labels_to_schema,
    find_overlapping_words,
    find_pause_boundary,
    get_speaker_id,
    is_baseline_take,
    is_clean2_take,
    parse_audacity_labels,
)
from eval_real import run_real_eval


@pytest.fixture
def dummy_words():
    # 5 dummy words spaced out
    return [
        {"word": "last", "start": 0.5, "end": 0.8},
        {"word": "spring", "start": 0.9, "end": 1.4},
        {"word": "our", "start": 1.5, "end": 1.7},
        {"word": "college", "start": 1.8, "end": 2.3},
        {"word": "library", "start": 2.4, "end": 2.9},
    ]


@pytest.fixture
def dummy_words_baseline():
    # Baseline with a pause between 'spring' and 'our' (gap >= 0.35s)
    return [
        {"word": "last", "start": 0.5, "end": 0.8},
        {"word": "spring", "start": 0.9, "end": 1.4},
        {"word": "our", "start": 1.9, "end": 2.1},  # gap: 1.9 - 1.4 = 0.5s pause
        {"word": "college", "start": 2.2, "end": 2.7},
        {"word": "library", "start": 2.8, "end": 3.3},
    ]


def test_speaker_id_extraction():
    assert get_speaker_id("real01__rushed_s4") == "real01"
    assert get_speaker_id("real01__baseline") == "real01"
    assert get_speaker_id("real02__clean2") == "real02"
    assert get_speaker_id("real03_dragging_s4") == "real03"


def test_take_type_helpers():
    assert is_baseline_take("real01__baseline") is True
    assert is_baseline_take("real01__rushed_s4") is False
    assert is_clean2_take("real01__clean2") is True
    assert is_clean2_take("real01__baseline") is False


def test_parse_audacity_labels(tmp_path):
    label_file = tmp_path / "take.labels.txt"
    label_file.write_text(
        "1.200\t2.500\trushed\n"
        "3.100\t4.200\tlong_pause\n"
        "# comment line\n"
        "\n"
        "5.000\t6.500\tvolume drop\n",
        encoding="utf-8"
    )
    records = parse_audacity_labels(label_file)
    assert len(records) == 3
    assert records[0] == {"start": 1.2, "end": 2.5, "type": "rushed"}
    assert records[1] == {"start": 3.1, "end": 4.2, "type": "long_pause"}
    assert records[2] == {"start": 5.0, "end": 6.5, "type": "volume_drop"}


def test_segment_flaw_conversion(dummy_words, dummy_words_baseline):
    # Span covers 'spring' (0.9-1.4) to 'college' (1.8-2.3)
    raw_labels = [{"start": 1.0, "end": 2.2, "type": "rushed"}]
    res = convert_labels_to_schema(
        take_id="real01__rushed_s4",
        baseline_id="real01__baseline",
        raw_labels=raw_labels,
        words_p=dummy_words,
        words_b=dummy_words_baseline,
    )
    assert res["id"] == "real01__rushed_s4"
    assert res["baseline"] == "real01__baseline"
    assert res["kind"] == "single"
    assert len(res["regions"]) == 1

    reg = res["regions"][0]
    assert reg["type"] == "rushed"
    assert reg["word_start"] == 1  # 'spring'
    assert reg["word_end"] == 3    # 'college'
    assert reg["start"] == 0.9     # start of 'spring'
    assert reg["end"] == 2.3       # end of 'college'
    assert reg["base_start"] == 0.9
    assert reg["base_end"] == 2.7


def test_long_pause_conversion(dummy_words, dummy_words_baseline):
    # Pause between 'college' (end 2.3) and 'library' (start 2.4)
    # Suppose participant inserted a long silence:
    words_p = [
        {"word": "last", "start": 0.5, "end": 0.8},
        {"word": "spring", "start": 0.9, "end": 1.4},
        {"word": "our", "start": 1.5, "end": 1.7},
        {"word": "college", "start": 1.8, "end": 2.3},
        {"word": "library", "start": 4.5, "end": 5.0},  # long pause 2.3 - 4.5
    ]
    raw_labels = [{"start": 2.35, "end": 4.45, "type": "long_pause"}]
    res = convert_labels_to_schema(
        take_id="real01__long_pause_s3",
        baseline_id="real01__baseline",
        raw_labels=raw_labels,
        words_p=words_p,
        words_b=dummy_words_baseline,
    )
    assert len(res["regions"]) == 1
    reg = res["regions"][0]
    assert reg["type"] == "long_pause"
    assert reg["word_start"] == 4  # index of 'library'
    assert reg["word_end"] == 4
    assert reg["start"] == 2.3     # end of previous word 'college'
    assert reg["end"] == 4.5       # start of next word 'library'


def test_missing_pause_conversion(dummy_words, dummy_words_baseline):
    # Baseline had a pause at word index 2 ('our' started at 1.9 after 'spring' ended at 1.4)
    # Participant spoke straight through (dummy_words has 'spring' end 1.4, 'our' start 1.5)
    raw_labels = [{"start": 1.2, "end": 1.6, "type": "missing_pause"}]
    res = convert_labels_to_schema(
        take_id="real01__missing_pause_s7",
        baseline_id="real01__baseline",
        raw_labels=raw_labels,
        words_p=dummy_words,
        words_b=dummy_words_baseline,
    )
    assert len(res["regions"]) == 1
    reg = res["regions"][0]
    assert reg["type"] == "missing_pause"
    assert reg["word_start"] == 2  # 'our'
    assert reg["word_end"] == 2
    assert reg["start"] == 1.4     # end of 'spring'
    assert reg["end"] == 1.5       # start of 'our'


def test_clean2_take_has_empty_regions(dummy_words):
    res = convert_labels_to_schema(
        take_id="real01__clean2",
        baseline_id="real01__baseline",
        raw_labels=[],
        words_p=dummy_words,
        words_b=dummy_words,
    )
    assert res["kind"] == "clean2"
    assert res["regions"] == []


def test_end_to_end_real_eval_pipeline(tmp_path):
    import numpy as np
    from speechlens.io import save_audio
    from speechlens.text import save_alignment

    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    y_base = 0.2 * np.sin(2 * np.pi * 220 * t)
    y_clean2 = 0.2 * np.sin(2 * np.pi * 220 * t)
    y_rushed = 0.2 * np.sin(2 * np.pi * 220 * t)

    data_dir = tmp_path / "data" / "real"
    data_dir.mkdir(parents=True)
    (data_dir / "alignments").mkdir()
    (data_dir / "labels").mkdir()

    save_audio(data_dir / "real01__baseline.wav", y_base, sr=sr)
    save_audio(data_dir / "real01__clean2.wav", y_clean2, sr=sr)
    save_audio(data_dir / "real01__rushed_s4.wav", y_rushed, sr=sr)

    words = [
        {"word": "last", "start": 0.2, "end": 0.8},
        {"word": "spring", "start": 0.9, "end": 1.5},
        {"word": "our", "start": 1.6, "end": 2.2},
        {"word": "college", "start": 2.3, "end": 3.0},
    ]
    save_alignment(data_dir / "alignments" / "real01__baseline.json", words)
    save_alignment(data_dir / "alignments" / "real01__clean2.json", words)
    save_alignment(data_dir / "alignments" / "real01__rushed_s4.json", words)

    clean2_label = {
        "id": "real01__clean2",
        "baseline": "real01__baseline",
        "kind": "clean2",
        "regions": [],
        "words": words,
    }
    with open(data_dir / "labels" / "real01__clean2.json", "w") as f:
        json.dump(clean2_label, f)

    rushed_label = {
        "id": "real01__rushed_s4",
        "baseline": "real01__baseline",
        "kind": "single",
        "regions": [
            {
                "type": "rushed",
                "severity": 1,
                "start": 0.9,
                "end": 2.2,
                "word_start": 1,
                "word_end": 2,
                "base_start": 0.9,
                "base_end": 2.2,
            }
        ],
        "words": words,
    }
    with open(data_dir / "labels" / "real01__rushed_s4.json", "w") as f:
        json.dump(rushed_label, f)

    out_json = tmp_path / "eval" / "real_results.json"
    run_real_eval(
        data_dir=data_dir,
        labels_dir=data_dir / "labels",
        out_path=out_json,
        iou_thrs=(0.5, 0.3),
    )

    assert out_json.exists()
    results = json.load(open(out_json))
    assert results["dataset"] == "real-voice"
    assert results["total_files"] == 2
    assert "clean2_summary" in results
    assert results["clean2_summary"]["takes_count"] == 1
    assert "iou_results" in results
    assert "iou_0.5" in results["iou_results"]
    assert "iou_0.3" in results["iou_results"]
    assert "per_speaker" in results
    assert "real01" in results["per_speaker"]["iou_0.5"]

