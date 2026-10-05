import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from speechlens.analyze import analyze  # noqa: E402
from speechlens.evaluate import match  # noqa: E402
from speechlens.inject import FLAWS, make_flawed, seed_for  # noqa: E402
from speechlens.io import load_audio  # noqa: E402
from speechlens.text import load_alignment  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng needed for fixture")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_baseline.py"), "--out", str(d)],
                   check=True, capture_output=True)
    return load_audio(d / "baseline" / "synthetic01.wav"), load_alignment(d / "alignments" / "synthetic01.json")


def _single(words, flaw):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_dataset
    return build_dataset.single_plan(words, flaw, 4, "test")


def test_identical_audio_is_clean(base):
    y, w = base
    res = analyze(y, w, y, w)
    assert res["regions"] == [] and res["scores"]["overall"] == 100.0


def test_gain_change_is_not_a_flaw(base):
    y, w = base
    assert analyze(y * 0.5, w, y, w)["regions"] == []


@pytest.mark.parametrize("flaw", FLAWS)
def test_severe_flaw_is_localised(base, flaw):
    y, w = base
    y2, w2, truth = make_flawed(y, w, _single(w, flaw))
    res = analyze(y2, w2, y, w)
    pairs, _, missed = match(res["regions"], truth, iou_thr=0.3)
    assert not missed, f"{flaw} not localised: truth={truth} pred={res['regions']}"
    assert res["scores"]["overall"] < 90


def test_injection_is_deterministic(base):
    y, w = base
    plan = _single(w, "monotone")
    a = make_flawed(y, w, plan)[0]
    b = make_flawed(y, w, plan)[0]
    assert hashlib.md5(a.tobytes()).hexdigest() == hashlib.md5(b.tobytes()).hexdigest()
    assert seed_for("a", 1) == seed_for("a", 1)


def test_mismatched_transcripts_rejected(base):
    y, w = base
    with pytest.raises(ValueError):
        analyze(y, w[:-1], y, w)


@pytest.mark.parametrize("flaw", FLAWS)
def test_overall_score_monotone_in_severity(base, flaw):
    y, w = base
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_dataset
    scores = []
    for sev in (1, 2, 3, 4):
        plan = build_dataset.single_plan(w, flaw, sev, "mono_test")
        y_flawed, w_flawed, _ = make_flawed(y, w, plan)
        res = analyze(y_flawed, w_flawed, y, w)
        scores.append(res["scores"]["overall"])
    for s_prev, s_curr in zip(scores[:-1], scores[1:]):
        assert s_prev >= s_curr, f"{flaw} overall score not non-increasing: {scores}"
