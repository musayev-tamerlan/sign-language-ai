# Improving the AzSL dataset

The best current model is `gru-v3-canonical`: test accuracy 79.9% and macro F1
64.8%. It is a useful baseline, not production-quality recognition for every word.

## Evidence-driven priority list

Run the audit after each evaluated experiment:

```powershell
.\training\.venv\Scripts\python.exe -X utf8 training\scripts\audit_training_data.py
```

It creates ignored local files in `training/data/generated/`:

- `quality-audit.json` — complete machine-readable report
- `quality-audit.csv` — review table for people

The first collection batch should prioritise the words with zero F1 and the known
confusions: `MƏNİM`, `MƏNƏ`, `ONUN`, `ORDA`, `BURA`, `BAZAR`, `NƏDİR`,
`EŞİTMƏ`, `UĞUR`, and `VAXT`. The report calculates the gap to 150 training
clips per weak label. More data matters most for labels with 12–32 source clips.

## Recording a labelled batch

For example, record 25 clips for MƏNİM:

```powershell
.\training\.venv\Scripts\python.exe training\scripts\record_samples.py MƏNİM --target 25
```

The recorder stores clips separately in `training/data/added_videos/`; it never
changes the official dataset or any existing experiment. Space starts/stops a clip,
R discards it, and Esc exits. Record one completed gesture per clip, with a short
neutral pause before and after it.

## Collection protocol

For each priority word, gather at least 150 usable clips, spread across at least
10 people. Vary lighting, room, clothing, camera distance and phone/webcam. Keep
the label exact. Do not copy, mirror or relabel clips from another word. Avoid
using the same person in both a future training set and its final signer-held-out
evaluation set.

Before admission, review every clip for the stated word, a visible signer, full
gesture completion and correct orientation. Then run landmark validation and create
a new, versioned split; never append unreviewed clips to an existing experiment.

## What software can and cannot solve

Canonical hand features and soft weighting improve use of existing data. They cannot
invent the missing variation for rare words or distinguish genuinely similar signs
when the available hand-only landmarks omit face, torso and context. The next
feature upgrade after data collection is pose landmarks (shoulders and elbows), then
facial landmarks if a signer review confirms facial grammar matters for target words.
