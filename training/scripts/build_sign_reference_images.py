"""Create one compact reference image per AzSL word from the source videos."""
import json
import subprocess
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
SOURCE, OUTPUT = ROOT / "AzSLD_Words_100", ROOT / "public/assets/signs"

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); items = []
    for folder in sorted(path for path in SOURCE.iterdir() if path.is_dir()):
        video = next(iter(sorted(folder.glob("*.mp4"))), None)
        if not video: continue
        capture = cv2.VideoCapture(str(video)); count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(count * .45)))
        ok, frame = capture.read(); capture.release()
        if not ok: continue
        height, width = frame.shape[:2]; scale = min(480 / width, 360 / height, 1)
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        filename = f"{len(items):03d}.jpg"
        clip = f"{len(items):03d}.webm"
        if not (OUTPUT / filename).exists() and not cv2.imwrite(str(OUTPUT / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 84]): raise RuntimeError(f"Could not write {filename}")
        if not (OUTPUT / clip).exists(): subprocess.run(["ffmpeg", "-y", "-ss", "0", "-i", str(video), "-t", "3", "-vf", "scale=480:-2,fps=12", "-an", "-c:v", "libvpx-vp9", "-b:v", "220k", str(OUTPUT / clip)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        items.append({"label": folder.name, "image": f"/assets/signs/{filename}", "video": f"/assets/signs/{clip}"})
    (OUTPUT / "manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {len(items)} sign references in {OUTPUT}")

if __name__ == "__main__": main()
