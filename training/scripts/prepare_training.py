"""Validate outputs and create a reproducible, duplicate-safe video split."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
from baseline import CONTRACT, fingerprint

ROOT = Path(__file__).resolve().parents[2]


def prepare(dataset, landmarks, seed=42, allow_incomplete=False):
    videos = sorted(p for p in dataset.rglob('*') if p.suffix.lower() == '.mp4')
    if not videos:
        raise ValueError('No source videos')
    pending = [p for p in videos if not (landmarks / p.relative_to(dataset).with_suffix('.npy')).exists()]
    if pending and not allow_incomplete:
        raise ValueError(f'{len(pending)} outputs missing; extraction must finish. No split written.')
    labels = sorted({p.relative_to(dataset).parts[0] for p in videos})
    groups, rejected, missing = {}, [], []
    for video in videos:
        rel = video.relative_to(dataset).with_suffix('.npy')
        path = landmarks / rel
        if not path.exists():
            missing.append(rel.as_posix())
            continue
        try:
            a = np.load(path, allow_pickle=False)
            digest = fingerprint(a)
        except Exception as exc:
            rejected.append(dict(path=rel.as_posix(), reason=str(exc)))
            continue
        label = labels.index(rel.parts[0])
        if digest in groups:
            if groups[digest][0]['label_id'] != label:
                raise ValueError('Identical model inputs have conflicting labels: ' + rel.as_posix())
        groups.setdefault(digest, []).append(dict(path=rel.as_posix(), label_id=label, sha256=digest))
    if missing and not allow_incomplete:
        raise ValueError(f'{len(missing)} outputs missing; extraction must finish. No split written.')
    by_class = defaultdict(list)
    for group in groups.values():
        by_class[group[0]['label_id']].append(group)
    rng = random.Random(seed)
    splits = dict(train=[], validation=[], test=[])
    for label_id in range(len(labels)):
        items = by_class[label_id]
        if len(items) < 3:
            raise ValueError('Need at least 3 unique valid sequences for class: ' + labels[label_id])
        rng.shuffle(items)
        n = max(1, round(len(items)*0.15))
        for name, subset in [('test',items[:n]), ('validation',items[n:2*n]), ('train',items[2*n:])]:
            splits[name].extend(row for group in subset for row in group)
    return dict(schema=1, contract=CONTRACT, seed=seed, labels=labels, splits=splits,
                landmarks=str(landmarks.resolve()), expected_videos=len(videos), missing=missing,
                rejected=rejected, complete=not missing,
                evaluation_scope='video-level; signer independence NOT established',
                duplicate_groups=sum(len(g)>1 for g in groups.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'AzSLD_Words_100')
    p.add_argument('--landmarks',type=Path,default=ROOT/'training/data/landmarks')
    p.add_argument('--output',type=Path,default=ROOT/'training/data/generated/splits.json')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.output.exists():
        p.error('Split already exists; use a new --output to preserve the experiment')
    try:
        result=prepare(args.dataset,args.landmarks,args.seed)
    except ValueError as exc:
        p.error(str(exc))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:len(v) for k,v in result['splits'].items()}))
    print(f'Rejected: {len(result["rejected"])}; labels: {len(result["labels"])}')
    print(args.output)


if __name__ == '__main__':
    main()
