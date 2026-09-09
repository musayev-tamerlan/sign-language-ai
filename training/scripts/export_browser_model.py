"""Export the selected AzSL model to ONNX and calibrate browser rejection thresholds."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from baseline_v2 import CONTRACT, Classifier, Sequences

ROOT = Path(__file__).resolve().parents[2]


def choose_acceptance(logits, targets):
    scores = torch.softmax(torch.tensor(logits), dim=1).numpy()
    order = np.argsort(scores, axis=1)
    predicted = order[:, -1]
    confidence = scores[np.arange(len(scores)), predicted]
    runner_up = scores[np.arange(len(scores)), order[:, -2]]
    candidates = []
    for threshold in np.arange(.50, .96, .05):
        for margin in np.arange(0, .41, .05):
            accepted = (confidence >= threshold) & (confidence - runner_up >= margin)
            coverage = float(accepted.mean())
            accuracy = float((predicted[accepted] == targets[accepted]).mean()) if accepted.any() else 0.0
            candidates.append(dict(confidence=round(float(threshold), 2), margin=round(float(margin), 2),
                                   coverage=coverage, accepted_accuracy=accuracy, accepted=int(accepted.sum())))
    qualified = [item for item in candidates if item["accepted"] >= 30 and item["accepted_accuracy"] >= .90]
    best = max(qualified or candidates, key=lambda item: (item["coverage"], item["accepted_accuracy"]))
    return {"selected": best, "validation_only": True,
            "note": "Thresholds maximise coverage subject to 90% accepted accuracy when possible."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "training/runs/gru-v3-canonical/best.pt")
    parser.add_argument("--split", type=Path, default=ROOT / "training/data/generated/splits-v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "public/assets/azsl-gru-v3.onnx")
    parser.add_argument("--metadata", type=Path, default=ROOT / "public/assets/azsl-gru-v3.labels.json")
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["contract"] != CONTRACT or checkpoint["labels"] != split["labels"]:
        raise ValueError("Checkpoint does not match the v2 feature contract/split")
    model = Classifier(len(split["labels"]), checkpoint["hidden"]).eval()
    model.load_state_dict(checkpoint["state_dict"])
    example = torch.zeros((1, 64, 132), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep one self-contained browser asset. It avoids browser-specific loading
    # of ONNX external tensor files.
    temporary = args.output
    torch.onnx.export(model, example, temporary, input_names=["landmarks"], output_names=["logits"],
                      opset_version=18, do_constant_folding=True)
    exported = onnx.load(temporary)
    onnx.checker.check_model(exported)
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    validation = DataLoader(Sequences(split["splits"]["validation"], Path(split["landmarks"])), batch_size=32)
    expected, actual, logits = [], [], []
    with torch.inference_mode():
        for features, labels in validation:
            pytorch = model(features).numpy()
            # The production browser session has a fixed single-item input.
            browser = np.concatenate([
                session.run(["logits"], {"landmarks": item[None].numpy()})[0]
                for item in features
            ])
            expected.append(pytorch)
            actual.append(browser)
            logits.append(pytorch)
    expected, actual = np.concatenate(expected), np.concatenate(actual)
    maximum_error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-4, atol=1e-5):
        raise ValueError(f"ONNX parity check failed: max error {maximum_error}")
    onnx.save_model(exported, args.output, save_as_external_data=False)
    args.output.with_suffix(args.output.suffix + '.data').unlink(missing_ok=True)
    targets = np.array([row["label_id"] for row in split["splits"]["validation"]])
    metadata = {
        "version": "gru-v3-canonical",
        "contract": CONTRACT,
        "input": {"name": "landmarks", "shape": [1, 64, 132], "frames": 64},
        "labels": split["labels"],
        "acceptance": choose_acceptance(np.concatenate(logits), targets),
        "onnx_parity_max_abs_error": maximum_error,
        "model_scope": "100 isolated AzSLD words; not sentence translation",
    }
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ONNX: {args.output}")
    print(f"Metadata: {args.metadata}")
    print(f"ONNX parity max error: {maximum_error:.8f}")
    print("Acceptance:", json.dumps(metadata["acceptance"]["selected"]))


if __name__ == "__main__":
    main()
