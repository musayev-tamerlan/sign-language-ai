# ML status — 2026-09-08

Extraction is active: observed growth from 2273 to 2283 outputs in 8 seconds.
Validation snapshot: 2438 files, 53/100 classes, 7248 source videos; 2433 valid,
5 shorter than 4 frames (including 2 all-zero). Missing files are pending or failed,
not confirmed failures. The extractor writes its error log only at completion.
The active extraction script and existing results were left unchanged.

## Safe validation

check_landmarks.py previously contained another extractor. It now only reads arrays
and writes training/data/generated/validation.json. Run from the project root:

~~~powershell
.\training\.venv\Scripts\python.exe training\scripts\check_landmarks.py
~~~

Exit 1 indicates incomplete or invalid data, expected during extraction. Rerun after
completion. The report includes missing outputs, per-class counts, shape/dtype,
NaN/Inf, empty/short/all-zero sequences, frame statistics and imbalance ratio.
Do not treat partial-run class imbalance as final. Do not run concurrent extraction.

## Before training

The active extractor uses detection order for its two 63-feature hand slots, not
stable left/right slots. The former check_landmarks.py extractor used handedness;
existing arrays have no provenance metadata. Verify consistency before training;
do not silently mix formats. Preserve existing data. Sampling selects up to 64
frames uniformly over each clip; browser preprocessing must match training.

After extraction: review failures and validation, establish a reproducible split,
then train a temporal baseline and evaluate accuracy/macro F1 before browser export.
Investigate signer metadata: random video splits do not establish signer independence.
No trained word classifier exists yet.

Future extraction hardening: finally-based cleanup, atomic saves, validation before
skipping existing arrays, incremental error logs and progress counters. Apply after
this run with a documented feature contract, without restarting current extraction.

Git previously ignored all training sources; now only generated data, environments,
training models, runs and checkpoints are excluded. Training scripts have no Git history.

## Baseline preparation

Training, duplicate-safe video splits and evaluation are now implemented. See [TRAINING.md](TRAINING.md). The baseline uses symmetric per-frame hand pooling to handle slot swaps. Full training has not started.

## First real-data baseline completed — 2026-09-08

GRU v1 finished 40 epochs. Best checkpoint: epoch 35, validation accuracy 76.15%, macro F1 56.53%. Held-out test: accuracy 76.52%, macro F1 52.72% (1086 videos). Artifacts: training/runs/gru-v1/best.pt and test.json. These are video-level metrics, not unseen-signer or live-webcam accuracy. Browser classifier integration and ONNX export remain pending.


## Balanced training started — 2026-09-09

Experiment gru-v2-balanced uses the frozen gru-v1 split, same architecture/seed and inverse-frequency training sampler. Four regression tests passed. Training launched; no v2 metrics available at launch. Compare best validation macro F1 to v1's 56.53%, along with per-class recall and accuracy. Do not tune on test. Artifacts: training/runs/gru-v2-balanced; completed.json indicates successful completion.

