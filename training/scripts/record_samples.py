"""Record reviewed, labelled AzSL examples from a webcam into a local staging area.

Keys: Space starts/stops a clip, R discards the current clip, Esc exits. The tool
never touches AzSLD_Words_100 or the training split. Review recorded videos before
extracting landmarks or admitting them to a new training experiment.
"""
import argparse
import csv
import time
import uuid
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]


def draw(frame, label, state, saved, target):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 104), (9, 35, 43), -1)
    frame = cv2.addWeighted(overlay, .78, frame, .22, 0)
    lines = [f"Label: {label}", f"{state} | saved: {saved}/{target}",
             "Space start/stop  R discard  Esc exit"]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (18, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .65 if index < 2 else .48, (230, 245, 238), 2, cv2.LINE_AA)
    return frame


def append_manifest(path, row):
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Exact Azerbaijani word label, e.g. MƏNİM")
    parser.add_argument("--target", type=int, default=25, help="Clips to record in this session")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=ROOT / "training/data/added_videos")
    args = parser.parse_args()
    if args.target < 1 or args.max_seconds <= 0:
        parser.error("target and max-seconds must be positive")
    destination = args.output / args.label
    destination.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.csv"
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera")
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = camera.get(cv2.CAP_PROP_FPS) or 30.0
    recording = False
    writer = None
    partial = None
    started = 0.0
    saved = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read camera frame")
            if recording:
                writer.write(frame)
                elapsed = time.monotonic() - started
                state = f"RECORDING {elapsed:.1f}s"
                if elapsed >= args.max_seconds:
                    key = ord(" ")
                else:
                    key = cv2.waitKey(1) & 0xFF
            else:
                state = "READY"
                key = cv2.waitKey(1) & 0xFF
            cv2.imshow("AzSL collection", draw(frame, args.label, state, saved, args.target))
            if key == 27:
                break
            if key in (ord("r"), ord("R")) and recording:
                writer.release()
                partial.unlink(missing_ok=True)
                recording, writer, partial = False, None, None
                continue
            if key == ord(" "):
                if not recording:
                    partial = destination / f"{uuid.uuid4().hex}.part.mp4"
                    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Could not create MP4 writer")
                    recording, started = True, time.monotonic()
                else:
                    writer.release()
                    final = partial.with_suffix("").with_suffix(".mp4")
                    partial.replace(final)
                    append_manifest(manifest, {
                        "path": final.relative_to(args.output).as_posix(), "label": args.label,
                        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "fps": round(fps, 2), "width": width, "height": height,
                    })
                    saved += 1
                    recording, writer, partial = False, None, None
                    if saved >= args.target:
                        break
    finally:
        if writer:
            writer.release()
        if partial:
            partial.unlink(missing_ok=True)
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} clips in {destination}")


if __name__ == "__main__":
    main()
