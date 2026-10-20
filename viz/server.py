"""SpeechLens Interactive Dashboard Server.

Runs a lightweight, zero-dependency local web server on port 3000 that supports:
- Interactive file upload of participant and baseline audio + transcript.
- One-click demo presets (e.g. Rushed, Monotone, Dragging, Clean).
- Dynamic forced alignment and contrastive feature analysis.
- Live time-series canvas overlays, interactive word-sync strip, and audible flaw playback.
"""
import base64
import http.server
import json
import os
import sys
import tempfile
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speechlens.analyze import analyze
from speechlens.io import load_audio
from speechlens.text import load_alignment, normalize_words, save_alignment

PORT = 3000

PRESETS = {
    "real01_rushed": {
        "title": "Real Voice: Rushed Delivery (Campus Life)",
        "participant": ROOT / "data" / "real" / "real01__rushed_s4.wav",
        "participant_align": ROOT / "data" / "real" / "alignments" / "real01__rushed_s4.json",
        "baseline": ROOT / "data" / "real" / "real01__baseline.wav",
        "baseline_align": ROOT / "data" / "real" / "alignments" / "real01__baseline.json",
        "transcript_path": ROOT / "data" / "transcripts" / "real_passage.txt",
    },
    "synthetic01_rushed": {
        "title": "Synthetic: Rushed Delivery (Severity 4)",
        "participant": ROOT / "data" / "flawed" / "synthetic01__rushed_s4.flac",
        "participant_align": ROOT / "data" / "flawed" / "synthetic01__rushed_s4.align.json",
        "baseline": ROOT / "data" / "baseline" / "synthetic01.wav",
        "baseline_align": ROOT / "data" / "alignments" / "synthetic01.json",
        "transcript_path": ROOT / "data" / "transcripts" / "synthetic01.txt",
    },
    "synthetic01_monotone": {
        "title": "Synthetic: Monotone Delivery (Severity 4)",
        "participant": ROOT / "data" / "flawed" / "synthetic01__monotone_s4.flac",
        "participant_align": ROOT / "data" / "flawed" / "synthetic01__monotone_s4.align.json",
        "baseline": ROOT / "data" / "baseline" / "synthetic01.wav",
        "baseline_align": ROOT / "data" / "alignments" / "synthetic01.json",
        "transcript_path": ROOT / "data" / "transcripts" / "synthetic01.txt",
    },
    "synthetic01_dragging": {
        "title": "Synthetic: Dragging Delivery (Severity 4)",
        "participant": ROOT / "data" / "flawed" / "synthetic01__dragging_s4.flac",
        "participant_align": ROOT / "data" / "flawed" / "synthetic01__dragging_s4.align.json",
        "baseline": ROOT / "data" / "baseline" / "synthetic01.wav",
        "baseline_align": ROOT / "data" / "alignments" / "synthetic01.json",
        "transcript_path": ROOT / "data" / "transcripts" / "synthetic01.txt",
    },
    "real01_clean": {
        "title": "Real Voice: Baseline Control (Clean Reference)",
        "participant": ROOT / "data" / "real" / "real01__baseline.wav",
        "participant_align": ROOT / "data" / "real" / "alignments" / "real01__baseline.json",
        "baseline": ROOT / "data" / "real" / "real01__baseline.wav",
        "baseline_align": ROOT / "data" / "real" / "alignments" / "real01__baseline.json",
        "transcript_path": ROOT / "data" / "transcripts" / "real_passage.txt",
    },
}


class SpeechLensHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.serve_index()
        elif path == "/api/presets":
            self.serve_presets()
        elif path.startswith("/audio/"):
            self.serve_audio(path[7:])
        elif path == "/health":
            self.send_json({"status": "ok", "app": "SpeechLens", "version": "1.0.0"})
        elif path.startswith("/samples/"):
            sample_file = ROOT / "samples" / path[9:]
            if sample_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(sample_file.read_bytes())
            else:
                self.send_error(404, "Sample not found")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/analyze":
            self.handle_analyze()
        else:
            self.send_error(404, "Not Found")

    def serve_index(self):
        html_file = Path(__file__).resolve().parent / "index.html"
        if not html_file.exists():
            self.send_error(500, "Dashboard index.html missing")
            return
        content = html_file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def serve_presets(self):
        data = []
        for pid, p in PRESETS.items():
            data.append({
                "id": pid,
                "title": p["title"],
                "has_audio": p["participant"].exists() and p["baseline"].exists(),
                "transcript": p["transcript_path"].read_text(encoding="utf-8") if p["transcript_path"].exists() else "",
            })
        self.send_json({"presets": data})

    def serve_audio(self, audio_id):
        # audio_id is <preset_id>_part or <preset_id>_base, or a temp file token
        token_parts = audio_id.split("__")
        target_path = None
        if len(token_parts) == 2:
            pid, role = token_parts[0], token_parts[1]
            if pid in PRESETS:
                target_path = PRESETS[pid]["participant"] if role == "part" else PRESETS[pid]["baseline"]
        elif audio_id.startswith("tmp_"):
            cand = Path(tempfile.gettempdir()) / f"{audio_id}.wav"
            if cand.exists():
                target_path = cand

        if target_path and target_path.exists():
            mime = "audio/wav" if target_path.suffix.lower() == ".wav" else ("audio/flac" if target_path.suffix.lower() == ".flac" else "audio/mpeg")
            data = target_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404, f"Audio file not found for: {audio_id}")

    def handle_analyze(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except Exception as err:
            self.send_json({"error": f"Invalid JSON payload: {err}"}, status=400)
            return

        preset_id = req.get("preset_id")
        transcript_text = req.get("transcript", "").strip()

        try:
            if preset_id and preset_id in PRESETS:
                p_info = PRESETS[preset_id]
                part_path = p_info["participant"]
                base_path = p_info["baseline"]
                part_align_path = p_info.get("participant_align")
                base_align_path = p_info.get("baseline_align")
                if not transcript_text and p_info["transcript_path"].exists():
                    transcript_text = p_info["transcript_path"].read_text(encoding="utf-8")

                wp = load_alignment(part_align_path) if part_align_path and part_align_path.exists() else None
                wb = load_alignment(base_align_path) if base_align_path and base_align_path.exists() else None

                if wp is None or wb is None:
                    from speechlens.align import align_words
                    if wp is None:
                        wp = align_words(part_path, transcript_text)
                    if wb is None:
                        wb = align_words(base_path, transcript_text)

                yp = load_audio(part_path)
                yb = load_audio(base_path)
                part_audio_url = f"/audio/{preset_id}__part"
                base_audio_url = f"/audio/{preset_id}__base"
            else:
                # Custom file upload mode
                p_b64 = req.get("participant_b64")
                b_b64 = req.get("baseline_b64")
                if not p_b64 or not b_b64 or not transcript_text:
                    self.send_json({"error": "Missing participant_b64, baseline_b64, or transcript"}, status=400)
                    return

                # Save temp audio files
                import uuid
                uid = uuid.uuid4().hex[:8]
                p_tmp = Path(tempfile.gettempdir()) / f"tmp_{uid}_part.wav"
                b_tmp = Path(tempfile.gettempdir()) / f"tmp_{uid}_base.wav"
                p_tmp.write_bytes(base64.b64decode(p_b64.split(",")[-1]))
                b_tmp.write_bytes(base64.b64decode(b_b64.split(",")[-1]))

                from speechlens.align import align_words
                wp = align_words(p_tmp, transcript_text)
                wb = align_words(b_tmp, transcript_text)

                yp = load_audio(p_tmp)
                yb = load_audio(b_tmp)
                part_audio_url = f"/audio/tmp_{uid}_part"
                base_audio_url = f"/audio/tmp_{uid}_base"

            res = analyze(yp, wp, yb, wb)
            res["audio"] = {
                "participant_url": part_audio_url,
                "baseline_url": base_audio_url,
            }
            self.send_json(res)
        except Exception as err:
            self.send_json({"error": str(err)}, status=500)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port=PORT):
    server_address = ("", port)
    httpd = http.server.ThreadingHTTPServer(server_address, SpeechLensHandler)
    print(f"SpeechLens Dashboard running at: http://localhost:{port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port)
