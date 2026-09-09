# Roadmap

## Phase 0 — Foundation

### DONE
- [x] Node.js project
- [x] Express server
- [x] Browser UI
- [x] Webcam
- [x] MediaPipe Hand Landmarker
- [x] 21 hand landmarks
- [x] 2-hand detection
- [x] Render account
- [x] GitHub connected
- [x] AzSLD Words 100 downloaded

### TODO
- [ ] clean realtime console logs
- [ ] commit current working state
- [ ] deploy baseline to Render

---

## Phase 1 — Dataset

Goal: understand the exact AzSLD_Words_100 structure.

Tasks:
- [ ] extract ZIP
- [ ] inspect directories
- [ ] identify class labels
- [ ] count videos per class
- [ ] inspect video FPS/resolution
- [ ] create train/validation/test split
- [ ] check class imbalance
- [ ] create dataset manifest

Do NOT assume folder names or labels before inspecting the downloaded ZIP.

---

## Phase 2 — Landmark extraction

Goal: convert videos into compact training data.

Pipeline:

```text
video
 ↓
MediaPipe
 ↓
hand landmarks
 ↓
normalization
 ↓
fixed-length sequence
 ↓
.npy / parquet / similar training data
```

Important:
- handle one/two hands
- normalize relative to wrist
- scale invariant
- preserve temporal information
- define padding/truncation strategy

---

## Phase 3 — First AI model

Start simple.

Recommended first experiment:
- GRU or LSTM
- input: temporal landmark sequence
- output: 100 classes

Metrics:
- accuracy
- macro F1
- per-class accuracy
- confusion matrix

Do not optimize for deployment until validation accuracy is acceptable.

---

## Phase 4 — Browser inference

Convert model to a browser-friendly format, preferably ONNX if compatible with the chosen runtime.

Target:

```text
camera
 ↓
MediaPipe
 ↓
landmarks
 ↓
AzSL model
 ↓
prediction + confidence
```

Only accept predictions above a confidence threshold.

Add temporal smoothing/debouncing so one gesture doesn't generate:

```text
SALAM SALAM SALAM SALAM
```

Instead:

```text
SALAM
```

---

## Phase 5 — Sentence construction

Create a buffer:

```text
[SALAM] [NECESEN] [MEN] [YAXŞI]
```

Then normalize into natural Azerbaijani text.

Potentially use an LLM only AFTER the classifier is working.

LLM should not be responsible for raw gesture recognition.

---

## Phase 6 — Fingerspelling

Use:

`AzSLD_Fingerspelling.zip`

Goal:
- recognize Azerbaijani alphabet
- spell unknown words
- combine letters into words

---

## Phase 7 — 200 words

Train/extend model using:

`AzSLD_Words_200.zip`

---

## Phase 8 — Continuous sentences

Later use:

`AzSLD_Sentences.zip`

This is large (~43.8 GB), so don't download/process it until the 100-word pipeline works.

Goal:

```text
continuous signing
 ↓
temporal segmentation
 ↓
gloss/word sequence
 ↓
Azerbaijani text
```

---

## Phase 9 — Communication features

Possible features:
- text-to-speech
- speech-to-text for the other person
- conversation history
- large text display
- mobile camera support
- offline model
- accessibility settings

---

## Phase 10 — Production

- [ ] Render deployment
- [ ] HTTPS
- [ ] custom domain if needed
- [ ] model hosting/CDN if model is too large for repo
- [ ] versioned models
- [ ] privacy notice
- [ ] camera permission UX
- [ ] performance testing on low-end Android devices
