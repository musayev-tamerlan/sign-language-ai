"""Create an actionable quality report for weak AzSL word classes."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sequence_stats(paths):
    lengths, no_hand = [], []
    for path in paths:
        array = np.load(path, allow_pickle=False)
        lengths.append(len(array))
        no_hand.append(float(np.mean(~np.any(array != 0, axis=1))))
    return {
        "landmark_files": len(paths),
        "frames_min": min(lengths),
        "frames_mean": round(float(np.mean(lengths)), 2),
        "frames_max": max(lengths),
        "no_hand_frame_percent": round(100 * float(np.mean(no_hand)), 2),
    }


def video_stats(paths):
    widths, heights, fps = [], [], []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        try:
            widths.append(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            heights.append(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps.append(float(capture.get(cv2.CAP_PROP_FPS)))
        finally:
            capture.release()
    return {
        "source_videos": len(paths),
        "video_sizes": sorted(set(f"{w}x{h}" for w, h in zip(widths, heights))),
        "video_fps": sorted(set(round(value, 2) for value in fps)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "training/runs/gru-v3-canonical/test.json")
    parser.add_argument("--split", type=Path, default=ROOT / "training/data/generated/splits-v2.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "AzSLD_Words_100")
    parser.add_argument("--landmarks", type=Path, default=ROOT / "training/data/landmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "training/data/generated/quality-audit.json")
    parser.add_argument("--csv", type=Path, default=ROOT / "training/data/generated/quality-audit.csv")
    parser.add_argument("--f1-threshold", type=float, default=.60)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    labels = report["labels"]
    label_ids = {label: index for index, label in enumerate(labels)}
    counts = {name: Counter(row["label_id"] for row in rows) for name, rows in split["splits"].items()}
    confusions = defaultdict(list)
    for actual, row in enumerate(report["confusion_matrix"]):
        for predicted, count in enumerate(row):
            if actual != predicted and count:
                confusions[actual].append({"predicted": labels[predicted], "count": count})
    rows = []
    for index, label in enumerate(labels):
        if report["per_class_f1"][index] >= args.f1_threshold:
            continue
        source = sorted((args.dataset / label).glob("*.mp4"))
        landmark = sorted((args.landmarks / label).glob("*.npy"))
        item = {
            "label": label,
            "label_id": index,
            "test_support": report["support"][index],
            "test_recall": round(report["per_class_accuracy"][index], 4),
            "test_f1": round(report["per_class_f1"][index], 4),
            "train_samples": counts["train"][index],
            "validation_samples": counts["validation"][index],
            "test_samples": counts["test"][index],
            "top_confusions": sorted(confusions[index], key=lambda item: item["count"], reverse=True)[:5],
            **sequence_stats(landmark),
            **video_stats(source),
        }
        item["suggested_new_videos"] = max(0, 150 - item["train_samples"])
        rows.append(item)
    rows.sort(key=lambda item: (item["test_f1"], item["train_samples"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["label", "train_samples", "validation_samples", "test_samples", "test_f1", "test_recall",
              "frames_mean", "no_hand_frame_percent", "suggested_new_videos", "top_confusions"]
    with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: (
                json.dumps(row[field], ensure_ascii=False) if field == "top_confusions" else row[field]
            ) for field in fields})
    print(f"Weak classes: {len(rows)}")
    print(f"JSON: {args.output}")
    print(f"CSV: {args.csv}")
    for row in rows[:10]:
        print(f"{row['label']}: F1={row['test_f1']:.3f}, train={row['train_samples']}, "
              f"no-hand={row['no_hand_frame_percent']:.1f}%, add≈{row['suggested_new_videos']}")


if __name__ == "__main__":
    main()
