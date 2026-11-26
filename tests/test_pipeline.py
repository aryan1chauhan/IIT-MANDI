import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from speechlens.analyze import _expand_runs, analyze, detect_regions  # noqa: E402
from speechlens.evaluate import match  # noqa: E402
from speechlens.features import HOP, extract  # noqa: E402
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


def test_pitch_bounds_are_reported_for_both_recordings(base):
    y, w = base
    bounds = analyze(y, w, y, w)["global"]["f0_bounds_hz"]
    assert bounds["participant"] == bounds["baseline"]
    assert 0 < bounds["participant"][0] < bounds["participant"][1]


def test_duration_injections_change_audio_length_in_the_expected_direction(base):
    y, w = base
    rushed, _, _ = make_flawed(y, w, _single(w, "rushed"))
    dragging, _, _ = make_flawed(y, w, _single(w, "dragging"))
    assert len(rushed) < len(y) < len(dragging)


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


def test_expand_runs_grows_confirmed_edges_without_seeding():
    thr = 1.0
    # Unconfirmed values below thr must never seed a region
    weak_dcc = np.array([0.8, 0.9, 0.85, 0.9, 0.0])
    assert _expand_runs([], weak_dcc, thr, direction=1, edge_ratio=0.65) == []

    # Confirmed seed [2, 3] expands outward across words >= 0.65 * thr and stops below 0.65
    dcc = np.array([0.5, 0.70, 1.2, 1.1, 0.68, -0.9, 0.9])
    assert _expand_runs([(2, 3)], dcc, thr, direction=1, edge_ratio=0.65) == [(1, 4)]

    # Separate confirmed runs whose expanded edges touch/overlap merge into one span
    merge_dcc = np.array([0.1, 1.2, 1.1, 0.70, 0.75, 1.3, 1.2, 0.2])
    assert _expand_runs([(1, 2), (5, 6)], merge_dcc, thr, direction=1, edge_ratio=0.65) == [(1, 6)]


def test_dragging_suppresses_stretched_pause_but_keeps_genuine_pause():
    n = 12
    words_b, words_p = [], []
    dur_b = np.full(n, 0.30)
    dur_p = np.full(n, 0.30)
    gap_b = np.full(n, 0.05)
    gap_p = np.full(n, 0.05)

    # Words 3..7 are dragged 2.0x; word 5 has a purely stretched gap (0.45 -> 0.90),
    # while word 6 has a genuine extra pause (0.20 -> 1.20)
    dur_p[3:8] = 0.60
    gap_b[5], gap_p[5] = 0.45, 0.90
    gap_b[6], gap_p[6] = 0.20, 1.20

    tb, tp = 0.5, 0.5
    for k in range(n):
        tb += gap_b[k]
        tp += gap_p[k]
        words_b.append({"word": f"w{k}", "start": tb, "end": tb + dur_b[k]})
        words_p.append({"word": f"w{k}", "start": tp, "end": tp + dur_p[k]})
        tb += dur_b[k]
        tp += dur_p[k]

    wt_b = {
        "dur": dur_b, "gap": gap_b, "energy": np.zeros(n), "hf": np.zeros(n),
        "pitch_range": np.full(n, 4.0), "pitch_med": np.zeros(n),
    }
    wt_p = {
        "dur": dur_p, "gap": gap_p, "energy": np.zeros(n), "hf": np.zeros(n),
        "pitch_range": np.full(n, 4.0), "pitch_med": np.zeros(n),
    }
    regs = detect_regions(wt_p, wt_b, words_p, words_b)
    pauses = [r for r in regs if r["type"] == "long_pause"]
    assert any(r["type"] == "dragging" and r["word_start"] <= 3 and r["word_end"] >= 7 for r in regs)
    assert [r["word_start"] for r in pauses] == [6]


def test_extract_uses_pitch_cc_and_preserves_bounds_and_identity(base):
    import parselmouth

    y, w = base
    feat = extract(y)
    assert 0 < feat.f0_floor_hz < feat.f0_median_hz < feat.f0_ceiling_hz

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=feat.sr)
    pitch_cc = snd.to_pitch_cc(
        time_step=HOP / feat.sr,
        pitch_floor=feat.f0_floor_hz,
        pitch_ceiling=feat.f0_ceiling_hz,
        silence_threshold=0.005,
    )
    f0 = pitch_cc.selected_array["frequency"]
    idx = np.clip(np.searchsorted(pitch_cc.xs(), feat.t), 0, len(pitch_cc.xs()) - 1)
    f0g = f0[idx]
    voiced = f0g > 0
    assert np.isclose(feat.f0_median_hz, float(np.median(f0g[voiced])), atol=1e-6)

    res = analyze(y, w, y, w)
    assert res["regions"] == [] and res["scores"]["overall"] == 100.0


def test_windowed_rate_catches_sustained_rushed_and_ignores_uniform_tempo():
    n = 20
    dur_b = np.full(n, 0.30)
    dur_p = dur_b.copy()
    # Add realistic per-word jitter outside the rushed span so single-word thr_word > 0.35
    jitter = np.array([1.15, 0.88, 1.12, 0.90, 1.14, 0.87, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.13, 0.89, 1.12, 0.88, 1.10, 0.91, 1.08, 0.92])
    dur_p *= jitter
    # Sustained 4-word rushed span (words 8..11 at 0.68x -> ln(0.68) = -0.386):
    # each word stays below inflated per-word thr_word, but the 3-word rolling window crosses thr_win
    dur_p[8:12] = 0.30 * 0.68

    def make_words(durs):
        out, t = [], 0.2
        for k, d in enumerate(durs):
            out.append({"word": f"w{k}", "start": t, "end": t + float(d)})
            t += float(d) + 0.05
        return out

    words_b, words_p = make_words(dur_b), make_words(dur_p)
    base_wt = lambda d: {
        "dur": d, "gap": np.full(n, 0.05), "energy": np.zeros(n), "hf": np.zeros(n),
        "pitch_range": np.full(n, 4.0), "pitch_med": np.zeros(n),
    }
    regs = detect_regions(base_wt(dur_p), base_wt(dur_b), words_p, words_b)
    rushed = [r for r in regs if r["type"] == "rushed"]
    assert len(rushed) == 1
    assert rushed[0]["word_start"] <= 9 and rushed[0]["word_end"] >= 10

    # Uniform 0.9x and 1.1x tempo scaling must produce 0 rate detections
    for scale in (1.0 / 0.9, 1.0 / 1.1):
        d_uni = dur_b * scale
        regs_uni = detect_regions(base_wt(d_uni), base_wt(dur_b), make_words(d_uni), words_b)
        assert [r for r in regs_uni if r["type"] in ("rushed", "dragging")] == []


def test_server_returns_400_on_value_error_and_413_on_oversized_upload(tmp_path, monkeypatch):
    import base64
    import http.client
    import http.server
    import json
    import tempfile
    import threading
    import speechlens.align as align_mod
    import speechlens.text as text_mod
    import viz.server as srv
    from speechlens.io import save_audio

    wav_path = tmp_path / "sample.wav"
    save_audio(wav_path, np.zeros(1600, dtype=np.float32))
    wav_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    # Avoid loading the 1.2 GB MMS_FA model while still exercising real normalize_words() + analyze() ValueErrors
    def fake_align(audio_path, transcript, device="cpu"):
        words = text_mod.normalize_words(transcript)
        if not words:
            raise ValueError("Transcript has no alignable words")
        extra = ["extra"] if "_part.wav" in str(audio_path) else []
        return [{"word": w, "start": 0.1 * i, "end": 0.1 * (i + 1)} for i, w in enumerate(words + extra)]

    monkeypatch.setattr(align_mod, "align_words", fake_align)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), srv.SpeechLensHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        # 1. Digits in transcript -> normalize_words raises ValueError -> HTTP 400
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        body = json.dumps({"participant_b64": wav_b64, "baseline_b64": wav_b64, "transcript": "Score was 100"}).encode()
        conn.request("POST", "/api/analyze", body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
        resp = conn.getresponse()
        data_400 = json.loads(resp.read().decode())
        assert resp.status == 400 and "numbers must be spelled out" in data_400["error"]
        conn.close()

        # 2. Word-count mismatch in analyze() -> ValueError -> HTTP 400
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        body_mm = json.dumps({"participant_b64": wav_b64, "baseline_b64": wav_b64, "transcript": "hello world"}).encode()
        conn.request("POST", "/api/analyze", body=body_mm, headers={"Content-Type": "application/json", "Content-Length": str(len(body_mm))})
        resp = conn.getresponse()
        data_mm = json.loads(resp.read().decode())
        assert resp.status == 400 and "Alignments differ in length" in data_mm["error"]
        conn.close()

        # 3. Oversized upload -> HTTP 413
        monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 32)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/analyze", body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
        resp = conn.getresponse()
        data_413 = json.loads(resp.read().decode())
        assert resp.status == 413 and "exceeds" in data_413["error"]
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
