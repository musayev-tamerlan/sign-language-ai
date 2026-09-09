# Temporal baseline

Training scripts are prepared. Full training has NOT been run. The current extraction
is not modified or restarted. No test accuracy for real AzSL data is available yet.

## Features

Contract: unordered-hands-xy-centered-nearest64-v1. Input: float32 (T,126).
Reject NaN/Inf, all-zero sequences, fewer than 4 frames and wrong shape/dtype.
Resample uniformly to 64 frames with nearest indices. Map x/y to [-1,1], preserve z;
missing hands remain zero. No statistics are fitted using validation/test data.

Each hand uses the SAME encoder. Sum/max pooling across the two hands makes the
per-frame representation exactly invariant to hand slot swaps. A GRU then models time.
This accepts both detection-order and handedness-order arrays without pretending to
recover anatomical left/right identity. Tradeoff: anatomical identity is lost; tracking
continuity is implicit in temporal features. This is a baseline to evaluate, not a claim
that all ambiguity in the existing extraction is fixed. Browser preprocessing must use
this exact contract when an evaluated model is integrated. Existing arrays are preserved.

## Split and leakage

Local dataset inspection found MP4 files with opaque IDs, no signer metadata files.
The split is stratified by word with seed 42, approximately 70/15/15 by unique input group.
Identical preprocessed sequences (including hand slot swaps) stay in one split.
Conflicting labels for identical inputs abort preparation. This detects exact duplicates,
not near-duplicate videos or repeated signers. Metrics are VIDEO-LEVEL; they do not
establish generalization to unseen signers. Obtain signer/session metadata for that.

Preparation refuses missing outputs or classes with fewer than 3 unique valid samples.
Invalid existing outputs are excluded and listed in the split manifest. Review exclusions
before training. No files are deleted. A split file cannot be overwritten implicitly.
Training checks paths, split overlap and data fingerprints; it refuses incomplete splits.

## After extraction completes

From the repository root:

~~~powershell
.\training\.venv\Scripts\python.exe training\scripts\check_landmarks.py
.\training\.venv\Scripts\python.exe training\scripts\prepare_training.py
.\training\.venv\Scripts\python.exe training\scripts\train_baseline.py --output training/runs/gru-v1
.\training\.venv\Scripts\python.exe training\scripts\evaluate_baseline.py --checkpoint training/runs/gru-v1/best.pt --split training/runs/gru-v1/splits.json --output training/runs/gru-v1/test.json
~~~

First inspect validation.json and extraction errors. If outputs are missing, resolve
those failures after the active extraction ends. Do not launch a second extractor.
Training defaults: CPU, 2 threads, batch 32, 40 epochs maximum, early stopping after
7 epochs without validation macro-F1 improvement. Use a NEW output folder per experiment.
The best checkpoint is chosen using validation only. Test data is read only by the
separate evaluation command; do not tune on the test report. Report contains accuracy,
macro F1, per-class accuracy/F1, support and confusion matrix with Azerbaijani labels.

Artifacts live under ignored training/runs and training/data. Checkpoints embed labels,
feature contract, split hash and PyTorch version. Full split and history are saved.
ONNX export and browser word recognition follow real-data evaluation; they are not
included in this preparation step.

## Lightweight tests

~~~powershell
.\training\.venv\Scripts\python.exe -m unittest discover -s training/scripts -p test_baseline.py -v
~~~

Tests cover arbitrary hand swaps, missing hands, metrics, incomplete-data rejection,
repeatable splits, duplicate isolation and a tiny synthetic train/save/reload/test run.
Synthetic results are plumbing checks, not evidence of recognition accuracy.

## Quality audit and rejection UX — 2026-09-09

Test set: MƏN accounts for 378/1086 videos; 19 classes have zero F1. Some classes
have only 3–4 test examples. These support counts limit per-class certainty.
Next experiments should use validation to compare class-weighted loss or balanced
sampling (train counts only), inspect label/landmark quality and collect more varied
examples of weak classes. Do not repeatedly select experiments using this test set.
A fresh signer-held-out set is needed before making real-world accuracy claims.

UI now handles optional worker result.recognition {label, confidence, runnerUp}.
Low confidence, invalid values, no hands or an empty prediction clear the previous
word and show “İşarəni tanımaq mümkün olmadı”. Three consecutive accepted predictions
are required. Initial 0.90 confidence / 0.20 margin limits are provisional, not calibrated
and NOT a guarantee of 90% accuracy. The current hand-only worker does not emit word
predictions: this path is tested and prepared for classifier integration, not live yet.
Fit acceptance thresholds on validation, report both accepted-prediction accuracy AND
coverage, and test unknown gestures/no-gesture clips. Rejecting more examples can raise
accuracy on accepted examples without improving the underlying classifier.

## Balanced experiment

Use --balanced-sampling to sample with replacement with inverse TRAIN class frequencies.
Every class has equal expected sampling mass; epoch length stays equal to training size.
Validation and test keep their natural distributions. No loss weighting is combined with
the sampler. This can improve minority-class recall at the cost of frequent-class accuracy;
improvement is not guaranteed. Architecture, split and seed remain the same for comparison.
New runs save config.json and completed.json; best checkpoints still use validation macro F1.

Command: training/.venv/Scripts/python.exe -u training/scripts/train_baseline.py --split training/runs/gru-v1/splits.json --output training/runs/gru-v2-balanced --balanced-sampling

## Canonical feature experiment

`baseline_v2` fixes the extractor's detection-order ambiguity at training time: hands
are sorted by wrist position in every frame, coordinates are expressed relative to each
wrist and scaled by hand size, while the raw wrist anchor is retained. It uses a separate
fingerprint and split manifest, so v1 artifacts remain valid and untouched.

Command: `training/.venv/Scripts/python.exe -u training/scripts/train_baseline.py --split training/data/generated/splits-v2.json --output training/runs/gru-v3-canonical --feature-module baseline_v2 --hidden 112`

Select by validation macro F1 against v1's 56.53%; evaluate the winner once on its
matching untouched test split. This experiment is currently in progress.

## Soft class-weight experiment

`--class-weight-power 0.5` applies an inverse-square-root class weight only to the
cross-entropy loss. The rarest and most common labels therefore differ by much less
than with complete resampling; each real video still appears once per epoch. It is
compared on validation macro F1, not test data.
