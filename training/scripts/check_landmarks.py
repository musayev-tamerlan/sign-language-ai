"""Read-only validation; never runs extraction or modifies landmark files."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def validate(landmarks, dataset, min_frames=4):
    files = sorted(landmarks.rglob('*.npy'))
    videos = sorted(p for p in dataset.rglob('*') if p.suffix.lower() == '.mp4')
    expected = {p.relative_to(dataset).with_suffix('.npy').as_posix() for p in videos}
    actual = {p.relative_to(landmarks).as_posix() for p in files}
    expected_classes = {p.split('/')[0] for p in expected}
    counts, valid_counts = Counter(), Counter()
    issues, lengths = [], []
    categories = Counter()
    sample = None
    for path in files:
        rel = path.relative_to(landmarks).as_posix()
        label = rel.split('/')[0]
        counts[label] += 1
        reasons = []
        try:
            a = np.load(path, allow_pickle=False)
            if sample is None:
                sample = dict(path=rel, shape=list(a.shape), dtype=str(a.dtype))
            if a.ndim != 2 or a.shape[1] != 126:
                reasons.append('wrong_shape')
            else:
                lengths.append(len(a))
                if not len(a):
                    reasons.append('empty')
                elif len(a) < min_frames:
                    reasons.append('too_short')
                if a.dtype != np.float32:
                    reasons.append('wrong_dtype')
                if not np.isfinite(a).all():
                    reasons.append('nonfinite')
                elif len(a) and not np.any(a):
                    reasons.append('all_zero')
        except Exception as exc:
            reasons.append('unreadable')
            issues.append(dict(path=rel, error=str(exc)[:300]))
        if reasons:
            categories.update(reasons)
            issues.append(dict(path=rel, reasons=reasons))
        else:
            valid_counts[label] += 1
    sizes = [valid_counts[c] for c in expected_classes or counts]
    positive = [n for n in sizes if n]
    return dict(files=len(files), classes=len(counts), expected_videos=len(videos),
                expected_classes=len(expected_classes), valid_files=sum(valid_counts.values()),
                bad_files=len(files)-sum(valid_counts.values()), categories=dict(categories),
                missing=sorted(expected-actual), unexpected=sorted(actual-expected),
                classes_without_valid_samples=sorted(c for c in expected_classes if not valid_counts[c]),
                files_per_class=dict(sorted(counts.items())), valid_per_class=dict(sorted(valid_counts.items())),
                average_files_per_class=len(files)/max(len(expected_classes or counts),1),
                imbalance_ratio=max(positive)/min(positive) if positive else None,
                frames=dict(min=min(lengths), max=max(lengths), average=sum(lengths)/len(lengths)) if lengths else None,
                sample=sample, issues=issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--landmarks', type=Path, default=ROOT/'training/data/landmarks')
    parser.add_argument('--dataset', type=Path, default=ROOT/'AzSLD_Words_100')
    parser.add_argument('--min-frames', type=int, default=4)
    parser.add_argument('--report', type=Path, default=ROOT/'training/data/generated/validation.json')
    args = parser.parse_args()
    if not args.landmarks.is_dir() or not args.dataset.is_dir():
        parser.error('Landmarks and dataset directories must exist')
    if args.min_frames < 1:
        parser.error('--min-frames must be positive')
    report = validate(args.landmarks, args.dataset, args.min_frames)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for key in ('files', 'classes', 'expected_videos', 'expected_classes', 'valid_files', 'bad_files', 'categories', 'frames', 'imbalance_ratio'):
        print(f'{key}: {report[key]}')
    print(f'Missing outputs (pending or failed): {len(report["missing"])}')
    print(f'Classes without valid samples: {len(report["classes_without_valid_samples"])}')
    print(f'Report: {args.report}')
    print('Snapshot only while extraction is active; rerun after it finishes.')
    return int(bool(report['bad_files'] or report['missing'] or report['unexpected'] or not report['expected_videos']))


if __name__ == '__main__':
    raise SystemExit(main())
