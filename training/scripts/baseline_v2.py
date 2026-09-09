"""Canonical, wrist-normalised two-hand features for a temporal classifier."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

CONTRACT = "spatially-sorted-wrist-normalized-66x2-nearest64-v1"
HAND_FEATURES = 66


def check_array(array):
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 126:
        raise ValueError("Expected float32 (T,126)")
    if len(array) < 4 or not np.isfinite(array).all() or not np.any(array):
        raise ValueError("Short, nonfinite or all-zero sequence")


def preprocess(array):
    """Return (64, 132): two canonical 66-value hands per frame.

    A hand is its 21 landmarks relative to its wrist, normalised by median 2-D
    landmark distance, then followed by the raw wrist position. Sorting by wrist
    x coordinate makes the two slots stable even though the extractor used
    MediaPipe's detection order.
    """
    check_array(array)
    sampled = array[np.linspace(0, len(array) - 1, 64).round().astype(int)].copy()
    hands = sampled.reshape(64, 2, 21, 3)
    present = np.any(hands != 0, axis=(2, 3))
    # Missing hands sort last. Stable sort makes ties deterministic.
    wrist_x = np.where(present, hands[:, :, 0, 0], np.inf)
    order = np.argsort(wrist_x, axis=1, kind="stable")
    hands = np.take_along_axis(hands, order[:, :, None, None], axis=1)
    present = np.take_along_axis(present, order, axis=1)
    wrists = hands[:, :, 0, :].copy()
    relative = hands - wrists[:, :, None, :]
    distances = np.linalg.norm(relative[:, :, 1:, :2], axis=-1)
    scales = np.median(distances, axis=-1, keepdims=True)
    scales = np.where((scales > 1e-6) & present[:, :, None], scales, 1.0)
    relative /= scales[:, :, None, :]
    relative *= present[:, :, None, None]
    wrists *= present[:, :, None]
    return np.concatenate((relative.reshape(64, 2, 63), wrists), axis=-1).reshape(64, 132).astype(np.float32)


def fingerprint(array):
    features = preprocess(array).reshape(64, 2, HAND_FEATURES)
    return hashlib.sha256(features.astype("<f4").tobytes()).hexdigest()


class Sequences(Dataset):
    def __init__(self, records, root):
        self.records, self.root = records, root
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]
        array = np.load(self.root / row["path"], allow_pickle=False)
        if fingerprint(array) != row["sha256"]:
            raise ValueError("Data changed since split: " + row["path"])
        return torch.from_numpy(preprocess(array)), row["label_id"]


class Classifier(nn.Module):
    def __init__(self, classes, hidden=112):
        super().__init__()
        self.hand = nn.Sequential(
            nn.Linear(HAND_FEATURES, 80), nn.ReLU(),
            nn.Linear(80, 80), nn.ReLU(),
        )
        self.gru = nn.GRU(160, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(hidden * 2, classes))
    def forward(self, features):
        hands = features.reshape(features.shape[0], features.shape[1], 2, HAND_FEATURES)
        encoded = self.hand(hands)
        # Slots are spatially canonical; pooling retains identity-independent cues.
        combined = torch.cat((encoded.sum(2), encoded.max(2).values), dim=-1)
        _, state = self.gru(combined)
        return self.head(torch.cat((state[-2], state[-1]), dim=-1))


def metrics(truth, predictions, classes):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, predicted in zip(truth, predictions):
        matrix[target, predicted] += 1
    true_positive = np.diag(matrix)
    support = matrix.sum(1)
    precision = true_positive / np.maximum(matrix.sum(0), 1)
    recall = true_positive / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(accuracy=float(true_positive.sum() / max(matrix.sum(), 1)), macro_f1=float(f1.mean()),
                per_class_accuracy=recall.tolist(), per_class_f1=f1.tolist(),
                support=support.tolist(), confusion_matrix=matrix.tolist())
