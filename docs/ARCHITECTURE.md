# Architecture

## Current

```text
Browser
 ├── index.html
 ├── app.js
 ├── Camera API
 └── MediaPipe Hand Landmarker
          ↓
       landmarks

Node.js
 └── Express
      └── serves public/
```

## Target

```text
                    ┌─────────────────────┐
                    │       Browser       │
                    │                     │
Camera ────────────►│ MediaPipe           │
                    │      ↓              │
                    │ Landmarks            │
                    │      ↓              │
                    │ AzSL Model           │
                    │      ↓              │
                    │ Prediction           │
                    │      ↓              │
                    │ Sentence Buffer      │
                    └─────────┬───────────┘
                              │
                              ▼
                    Azerbaijani text

                    ┌─────────────────────┐
                    │ Node / Express      │
                    │                     │
                    │ static hosting      │
                    │ API (future)        │
                    └─────────────────────┘
```

## Why browser inference

Do NOT stream raw camera video to Node for every frame.

Browser inference:
- lower latency
- lower server cost
- better privacy
- easier free deployment
- works well with WebAssembly/WebGPU/ONNX-style inference

## Model stages

### Stage A

MediaPipe:

`image → hand landmarks`

### Stage B

Classifier:

`landmark sequence → word`

### Stage C

Language layer:

`word sequence → natural Azerbaijani sentence`

### Stage D

Optional voice:

`Azerbaijani text → speech`

## Performance rules

Never:
- `console.log()` every frame
- dump full landmark arrays to console
- make network requests every frame
- update large DOM trees every frame

Prefer:
- requestAnimationFrame for camera loop
- inference at controlled FPS
- UI update throttling
- prediction smoothing
- confidence threshold
