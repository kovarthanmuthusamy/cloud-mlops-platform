#!/usr/bin/env python3
"""Render the terminal demo video (MP4 + GIF) from REAL captured output.

The outputs shown are the actual results captured from running this project
locally on 2026-11-05 (training accuracy 1.0000, NO_DRIFT / DRIFT_DETECTED
runs, live API responses, 7 passed tests) — only line wrapping/trimming for
screen width was applied. Renders frames with Pillow; encode with ffmpeg:

    python3 docs/make_demo_video.py --frames-dir /tmp/demo-frames
    ffmpeg -framerate 15 -i /tmp/demo-frames/f%04d.png -c:v libx264 -pix_fmt yuv420p -crf 21 docs/demo.mp4
    ffmpeg -i docs/demo.mp4 -vf "fps=10,scale=880:-1:flags=lanczos,palettegen" -y /tmp/pal.png
    ffmpeg -i docs/demo.mp4 -i /tmp/pal.png -lavfi "fps=10,scale=880:-1:flags=lanczos [x]; [x][1:v] paletteuse" -y docs/demo.gif
"""
import argparse
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
PAD = 26
HEADER_H = 40
FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"
FS = 17
LH = 25
MAX_LINES = (H - HEADER_H - 18) // LH

BG = (13, 17, 23)
HEADER_BG = (22, 27, 34)
WHITE = (230, 237, 243)
DIM = (139, 148, 158)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
BLUE = (88, 166, 255)
YELLOW = (210, 153, 34)

font = ImageFont.truetype(FONT_PATH, FS)
font_b = ImageFont.truetype(FONT_BOLD, FS)
CHAR_W = font.getlength("M")

# Each step: ("cmd", [lines to type]) or ("out", [(text, colour), ...]) or
# ("pause", seconds)
DRIFT_HEAD = [
    ("alcohol                      psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("malic_acid                   psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("ash                          psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("alcalinity_of_ash            psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("magnesium                    psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("total_phenols                psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("flavanoids                   psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("nonflavanoid_phenols         psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("proanthocyanins              psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("color_intensity              psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("hue                          psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("od280/od315_of_diluted_wines psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("proline                      psi=  0.000 ks_p=  1.0000 [ok]", DIM),
]
DRIFTED = [
    ("alcohol                      psi=  3.076 ks_p=  0.0000 [DRIFT]", RED),
    ("malic_acid                   psi=  7.594 ks_p=  0.0000 [DRIFT]", RED),
    ("ash                          psi=  1.964 ks_p=  0.0000 [DRIFT]", RED),
    ("alcalinity_of_ash            psi=  1.815 ks_p=  0.0000 [DRIFT]", RED),
    ("magnesium                    psi=  5.685 ks_p=  0.0000 [DRIFT]", RED),
    ("total_phenols                psi=  4.195 ks_p=  0.0000 [DRIFT]", RED),
    ("flavanoids                   psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("nonflavanoid_phenols         psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("proanthocyanins              psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("color_intensity              psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("hue                          psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("od280/od315_of_diluted_wines psi=  0.000 ks_p=  1.0000 [ok]", DIM),
    ("proline                      psi=  0.000 ks_p=  1.0000 [ok]", DIM),
]

SCRIPT = [
    ("out", [("# cloud-mlops-platform — end-to-end demo", YELLOW)]),
    ("pause", 0.6),
    ("cmd", ["PYTHONPATH=src python scripts/run_training.py"]),
    ("out", [
        ("2026/11/05 11:46:17 INFO mlflow: Creating experiment 'cloud-mlops-wine' ...", DIM),
        ("run_id=b8608b6ef42b42738b48b4a0b4847e3b", WHITE),
        ("accuracy=1.0000 f1_weighted=1.0000", GREEN),
        ("model saved to models/model.pkl", WHITE),
    ]),
    ("pause", 0.8),
    ("cmd", ["PYTHONPATH=src python scripts/check_drift.py"]),
    ("out", DRIFT_HEAD + [("NO_DRIFT", GREEN)]),
    ("pause", 0.9),
    ("cmd", ["PYTHONPATH=src python scripts/check_drift.py --drifted"]),
    ("out", DRIFTED + [
        ("DRIFT_DETECTED flagged_features=['alcohol', 'malic_acid', 'ash',", RED),
        ("'alcalinity_of_ash', 'magnesium', 'total_phenols']", RED),
        ("(exit code 1 → Airflow branches to retraining)", YELLOW),
    ]),
    ("pause", 1.2),
    ("cmd", ["PYTHONPATH=src:. uvicorn app.main:app --port 8000"]),
    ("out", [
        ("INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)", DIM),
    ]),
    ("pause", 0.7),
    ("cmd", ["curl http://127.0.0.1:8000/health"]),
    ("out", [('{"status":"ok","model_loaded":true,"model_path":"models/model.pkl"}', BLUE)]),
    ("pause", 0.7),
    ("cmd", [
        "curl -X POST http://127.0.0.1:8000/predict \\",
        "  -H \"Content-Type: application/json\" \\",
        "  -d '{\"alcohol\": 13.2, \"malic_acid\": 1.78, \"ash\": 2.14, \"alcalinity_of_ash\": 11.2,",
        "       \"magnesium\": 100.0, \"total_phenols\": 2.65, \"flavanoids\": 2.76,",
        "       \"nonflavanoid_phenols\": 0.26, \"proanthocyanins\": 1.28, \"color_intensity\": 4.38,",
        "       \"hue\": 1.05, \"od280/od315_of_diluted_wines\": 3.4, \"proline\": 1050.0}'",
    ]),
    ("out", [('{"prediction":0,"class_name":"class_0"}', BLUE)]),
    ("pause", 0.7),
    ("cmd", ["curl http://127.0.0.1:8000/metrics | grep predict_"]),
    ("out", [
        ('predict_requests_total{status="ok"} 1.0', WHITE),
        ("predict_latency_seconds_count 1.0", WHITE),
    ]),
    ("pause", 0.8),
    ("cmd", ["pytest -q"]),
    ("out", [
        (".......                                                            [100%]", DIM),
        ("7 passed, 2 warnings in 20.24s", GREEN),
    ]),
    ("pause", 0.8),
    ("out", [
        ("", WHITE),
        ("== Demo complete — github.com/kovarthanmuthusamy/cloud-mlops-platform ==", GREEN),
    ]),
    ("pause", 3.0),
]

FPS = 15
TYPE_CHARS_PER_FRAME = 2
OUT_LINES_PER_FRAME = 3


def render(lines, cursor_on=True):
    """lines: list of (text, colour, bold). Returns a frame image."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, HEADER_H], fill=HEADER_BG)
    for i, c in enumerate([(248, 81, 73), (210, 153, 34), (63, 185, 80)]):
        d.ellipse([16 + i * 22, 14, 28 + i * 22, 26], fill=c)
    title = "kovarthanam — ~/cloud-mlops-platform"
    tw = font.getlength(title)
    d.text(((W - tw) / 2, 11), title, font=font, fill=DIM)
    vis = lines[-MAX_LINES:]
    y = HEADER_H + 12
    for text, colour, bold in vis:
        d.text((PAD, y), text, font=font_b if bold else font, fill=colour)
        y += LH
    return img


def build_frames():
    frames = []
    buffer = []  # (text, colour, bold)

    def snap():
        frames.append(render(list(buffer)))

    def hold(sec):
        for _ in range(int(sec * FPS)):
            snap()

    prompt = ("~/cloud-mlops-platform $ ", GREEN, True)
    for kind, payload in SCRIPT:
        if kind == "pause":
            hold(payload)
        elif kind == "cmd":
            done_lines = []
            for li, line in enumerate(payload):
                prefix = prompt[0] if li == 0 else ""
                for i in range(0, len(line), TYPE_CHARS_PER_FRAME):
                    typed = line[: i + TYPE_CHARS_PER_FRAME]
                    cur = list(buffer) + done_lines + [(prefix + typed, WHITE, False)]
                    frames.append(render(cur))
                done_lines.append((prefix + line, WHITE, False))
            buffer.extend(done_lines)
            snap()
        elif kind == "out":
            for i in range(0, len(payload), OUT_LINES_PER_FRAME):
                for text, colour in payload[i : i + OUT_LINES_PER_FRAME]:
                    buffer.append((text, colour, False))
                snap()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    args = ap.parse_args()
    import os
    os.makedirs(args.frames_dir, exist_ok=True)
    frames = build_frames()
    for i, fr in enumerate(frames):
        fr.save(f"{args.frames_dir}/f{i:04d}.png")
    print(f"{len(frames)} frames at {FPS} fps = {len(frames)/FPS:.1f}s")


if __name__ == "__main__":
    main()
