import argparse
import json
import base64
import subprocess
import tempfile
import sys
from pathlib import Path

def convert_to_mp3(input_path):
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        out_path = f.name
    
    cmd = ['ffmpeg', '-y', '-i', input_path, '-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k', out_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg on {input_path}")
        sys.exit(1)
    
    return out_path

def file_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def make_html(report_data, part_b64, base_b64):
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SpeechLens Analysis Report</title>
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f9fafb; color: #111827; margin: 0; padding: 20px; }
    .container { max-width: 1000px; margin: 0 auto; background: #fff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 24px; }
    h1 { margin-top: 0; color: #1f2937; }
    .scores-grid { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }
    .score-card { background: #f3f4f6; padding: 16px; border-radius: 8px; flex: 1; min-width: 120px; text-align: center; }
    .score-card.overall { background: #e0e7ff; border: 2px solid #818cf8; }
    .score-title { font-size: 0.875rem; text-transform: uppercase; color: #6b7280; font-weight: bold; }
    .score-value { font-size: 1.5rem; font-weight: 800; color: #111827; margin-top: 8px; }
    .score-value.overall-val { font-size: 2rem; color: #4338ca; }
    .chart-container { position: relative; width: 100%; height: 250px; background: #fafafa; border: 1px solid #e5e7eb; border-radius: 4px; overflow: hidden; cursor: pointer; }
    canvas { display: block; width: 100%; height: 100%; }
    .words-strip { display: flex; overflow-x: auto; padding: 8px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 24px; white-space: nowrap; font-size: 0.875rem; }
    .word { margin-right: 4px; padding: 2px 4px; border-radius: 4px; cursor: pointer; color: #4b5563; }
    .word:hover { background: #e5e7eb; }
    .word.active { background: #dbeafe; color: #1e40af; font-weight: bold; }
    
    .regions-list { display: flex; flex-direction: column; gap: 16px; }
    .region-card { border: 1px solid #fca5a5; background: #fef2f2; padding: 16px; border-radius: 8px; }
    .region-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .region-type { font-weight: bold; color: #991b1b; text-transform: uppercase; background: #fee2e2; padding: 4px 8px; border-radius: 4px; font-size: 0.875rem; }
    .region-times { color: #7f1d1d; font-size: 0.875rem; }
    .region-text { font-style: italic; color: #374151; margin-bottom: 12px; padding-left: 12px; border-left: 4px solid #f87171; }
    .region-stats { display: flex; gap: 24px; font-size: 0.875rem; margin-bottom: 12px; color: #4b5563; flex-wrap: wrap; }
    .region-stats div span { font-weight: bold; color: #111827; }
    .region-explanation { font-size: 0.95rem; color: #1f2937; line-height: 1.5; margin-bottom: 16px; }
    .btn-group { display: flex; gap: 12px; }
    button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; font-size: 0.875rem; }
    button.play-btn { background: #3b82f6; color: white; }
    button.play-btn:hover { background: #2563eb; }
    button.play-base-btn { background: #9ca3af; color: white; }
    button.play-base-btn:hover { background: #6b7280; }
    
    .no-flaws { padding: 32px; text-align: center; background: #ecfdf5; border: 1px solid #6ee7b7; border-radius: 8px; color: #065f46; font-weight: bold; font-size: 1.125rem; }
    
    .playback-info { margin-bottom: 12px; font-size: 0.875rem; color: #6b7280; }
</style>
</head>
<body>

<div class="container">
    <h1>SpeechLens Analysis</h1>
    
    <div class="scores-grid" id="scoresGrid"></div>
    
    <h3>Delivery Profile</h3>
    <div class="playback-info" id="playbackInfo">Click chart or words to seek.</div>
    <div class="chart-container" id="chartContainer">
        <canvas id="chartCanvas"></canvas>
    </div>
    
    <div class="words-strip" id="wordsStrip"></div>
    
    <h3>Detected Flaws</h3>
    <div class="regions-list" id="regionsList"></div>
</div>

<audio id="audioParticipant" src="data:audio/mp3;base64,{{PART_B64}}"></audio>
<audio id="audioBaseline" src="data:audio/mp3;base64,{{BASE_B64}}"></audio>

<script>
    const reportData = {{REPORT_JSON}};
    const audioPart = document.getElementById('audioParticipant');
    const audioBase = document.getElementById('audioBaseline');
    
    let playTimeout = null;
    let playingSource = null;
    
    // Render Scores
    const scoresGrid = document.getElementById('scoresGrid');
    const scoreKeys = ['overall', 'pace', 'pausing', 'expressiveness', 'volume', 'clarity'];
    scoreKeys.forEach(k => {
        const val = reportData.scores[k];
        if (val === undefined) return;
        const div = document.createElement('div');
        div.className = `score-card ${k === 'overall' ? 'overall' : ''}`;
        div.innerHTML = `<div class="score-title">${k}</div><div class="score-value ${k === 'overall' ? 'overall-val' : ''}">${val.toFixed(1)}</div>`;
        scoresGrid.appendChild(div);
    });
    
    // Render Words
    const wordsStrip = document.getElementById('wordsStrip');
    reportData.words.forEach((w, i) => {
        const span = document.createElement('span');
        span.className = 'word';
        span.textContent = w.word;
        span.dataset.start = w.start;
        span.dataset.end = w.end;
        span.onclick = () => {
            playSpan('participant', w.start, w.end);
        };
        wordsStrip.appendChild(span);
    });
    
    // Highlight active word
    audioPart.addEventListener('timeupdate', () => {
        const t = audioPart.currentTime;
        Array.from(wordsStrip.children).forEach(span => {
            const start = parseFloat(span.dataset.start);
            const end = parseFloat(span.dataset.end);
            if (t >= start && t <= end) {
                span.classList.add('active');
                if (span.scrollIntoViewIfNeeded) {
                    span.scrollIntoViewIfNeeded();
                }
            } else {
                span.classList.remove('active');
            }
        });
    });
    
    // Render Regions
    const regionsList = document.getElementById('regionsList');
    if (!reportData.regions || reportData.regions.length === 0) {
        regionsList.innerHTML = '<div class="no-flaws">No flaws detected! Perfect delivery.</div>';
    } else {
        reportData.regions.forEach(r => {
            const card = document.createElement('div');
            card.className = 'region-card';
            
            let observedVal = typeof r.observed === 'number' ? r.observed.toFixed(3) : r.observed;
            let refVal = typeof r.reference === 'number' ? r.reference.toFixed(3) : r.reference;
            let unit = r.unit || '';
            
            card.innerHTML = `
                <div class="region-header">
                    <span class="region-type">${r.type}</span>
                    <span class="region-times">${r.start.toFixed(2)}s &mdash; ${r.end.toFixed(2)}s</span>
                </div>
                <div class="region-text">"${r.text}"</div>
                <div class="region-stats">
                    <div>Observed: <span>${observedVal} ${unit}</span></div>
                    <div>Reference: <span>${refVal} ${unit}</span></div>
                </div>
                <div class="region-explanation">${r.explanation}</div>
                <div class="btn-group">
                    <button class="play-btn" onclick="playSpan('participant', ${r.start}, ${r.end})">Play Yours</button>
                    <button class="play-base-btn" onclick="playSpan('baseline', ${r.base_start}, ${r.base_end})">Play Reference</button>
                </div>
            `;
            regionsList.appendChild(card);
        });
    }
    
    function playSpan(source, start, end) {
        if (playTimeout) clearTimeout(playTimeout);
        audioPart.pause();
        audioBase.pause();
        
        const audio = source === 'participant' ? audioPart : audioBase;
        audio.currentTime = start;
        audio.play().catch(e => console.log(e));
        
        const duration = (end - start) * 1000;
        playTimeout = setTimeout(() => {
            audio.pause();
        }, duration);
    }
    
    // Draw Chart
    const canvas = document.getElementById('chartCanvas');
    const ctx = canvas.getContext('2d');
    let width, height;
    
    function resizeCanvas() {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        width = canvas.width;
        height = canvas.height;
        drawChart();
    }
    window.addEventListener('resize', resizeCanvas);
    
    function drawChart() {
        ctx.clearRect(0, 0, width, height);
        const t = reportData.series.t;
        if (!t || t.length === 0) return;
        
        const tMin = t[0];
        const tMax = t[t.length - 1];
        
        function getX(time) {
            return (time - tMin) / (tMax - tMin) * width;
        }
        
        // Shade regions
        if (reportData.regions) {
            reportData.regions.forEach(r => {
                const x1 = getX(r.start);
                const x2 = getX(r.end);
                ctx.fillStyle = 'rgba(254, 202, 202, 0.4)'; // red-200
                ctx.fillRect(x1, 0, x2 - x1, height);
                
                ctx.fillStyle = '#991b1b';
                ctx.font = '10px sans-serif';
                ctx.fillText(r.type, x1 + 4, 14);
                ctx.beginPath();
                ctx.strokeStyle = '#f87171';
                ctx.lineWidth = 2;
                ctx.moveTo(x1, 0); ctx.lineTo(x1, height);
                ctx.moveTo(x2, 0); ctx.lineTo(x2, height);
                ctx.stroke();
            });
        }
        
        const pE = reportData.series.participant.energy_db;
        const bE = reportData.series.baseline.energy_db;
        const pF0 = reportData.series.participant.f0_st;
        
        // Find min/max for scaling
        let eMin = Math.min(...pE, ...bE);
        let eMax = Math.max(...pE, ...bE);
        let f0Min = 0, f0Max = 0;
        
        let validF0 = pF0.filter(v => v !== null);
        if(validF0.length > 0) {
            f0Min = Math.min(...validF0);
            f0Max = Math.max(...validF0);
        }
        
        function drawLine(data, minV, maxV, color, yOffset, yHeight, lineWidth=1, step=1) {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = lineWidth;
            let first = true;
            for (let i = 0; i < data.length; i+=step) {
                const val = data[i];
                if (val === null) {
                    first = true;
                    continue;
                }
                const x = getX(t[i]);
                // normalize
                let yNorm = 0;
                if (maxV > minV) yNorm = (val - minV) / (maxV - minV);
                const y = yOffset + yHeight - (yNorm * yHeight);
                
                if (first) {
                    ctx.moveTo(x, y);
                    first = false;
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }
        
        // Top half for Energy
        drawLine(bE, eMin, eMax, 'rgba(156, 163, 175, 0.4)', 0, height/2, 2); // Baseline
        drawLine(pE, eMin, eMax, '#1f2937', 0, height/2, 1.5); // Participant
        
        // Bottom half for F0
        if(f0Max > f0Min) {
            drawLine(pF0, f0Min, f0Max, '#3b82f6', height/2, height/2, 2);
        }
    }
    
    // Click on chart to seek
    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const tArray = reportData.series.t;
        if(tArray && tArray.length > 0) {
            const time = tArray[0] + (x / width) * (tArray[tArray.length-1] - tArray[0]);
            audioPart.currentTime = time;
            audioPart.play();
            if(playTimeout) clearTimeout(playTimeout);
        }
    });
    
    // Render logic
    setTimeout(resizeCanvas, 100);
</script>
</body>
</html>
"""
    
    html = html_template.replace("{{REPORT_JSON}}", json.dumps(report_data))
    html = html.replace("{{PART_B64}}", part_b64)
    html = html.replace("{{BASE_B64}}", base_b64)
    return html

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_json", help="Path to report.json")
    parser.add_argument("--participant", required=True, help="Participant audio file")
    parser.add_argument("--baseline", required=True, help="Baseline audio file")
    parser.add_argument("--out", required=True, help="Output HTML file")
    args = parser.parse_args()
    
    with open(args.report_json, "r") as f:
        report_data = json.load(f)
        
    part_mp3 = convert_to_mp3(args.participant)
    base_mp3 = convert_to_mp3(args.baseline)
    
    part_b64 = file_to_base64(part_mp3)
    base_b64 = file_to_base64(base_mp3)
    
    Path(part_mp3).unlink()
    Path(base_mp3).unlink()
    
    html = make_html(report_data, part_b64, base_b64)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"Generated standalone report at {args.out}")

if __name__ == "__main__":
    main()
