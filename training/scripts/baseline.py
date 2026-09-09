"""Order-invariant hand encoder + temporal GRU. No MediaPipe or video processing."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

CONTRACT = 'unordered-hands-xy-centered-nearest64-v1'


def check_array(a):
    if a.dtype != np.float32 or a.ndim != 2 or a.shape[1] != 126:
        raise ValueError('Expected float32 (T,126)')
    if len(a) < 4 or not np.isfinite(a).all() or not np.any(a):
        raise ValueError('Short, nonfinite or all-zero sequence')


def preprocess(a):
    check_array(a)
    a = a[np.linspace(0, len(a)-1, 64).round().astype(int)].copy().reshape(64, 2, 21, 3)
    present = np.any(a != 0, axis=(2, 3))
    a[..., :2] = 2 * a[..., :2] - 1
    a *= present[..., None, None]
    return a.reshape(64, 126)


def fingerprint(a):
    # Hash exactly the model input, treating hand block swaps as equivalent.
    a = preprocess(a).reshape(64, 2, 63)
    payload = b''.join(b''.join(sorted(hand.astype('<f4').tobytes() for hand in frame)) for frame in a)
    return hashlib.sha256(payload).hexdigest()


class Sequences(Dataset):
    def __init__(self, records, root):
        self.records, self.root = records, root
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]
        a = np.load(self.root / row['path'], allow_pickle=False)
        if fingerprint(a) != row['sha256']:
            raise ValueError('Data changed since split: ' + row['path'])
        return torch.from_numpy(preprocess(a)), row['label_id']


class Classifier(nn.Module):
    def __init__(self, classes, hidden=96):
        super().__init__()
        self.hand = nn.Sequential(nn.Linear(63, 64), nn.ReLU(), nn.Linear(64,64), nn.ReLU())
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, classes))
    def forward(self, x):
        hands = x.reshape(x.shape[0], x.shape[1], 2, 63)
        mask = hands.abs().sum(-1, keepdim=True).ne(0)
        encoded = self.hand(hands) * mask
        # Shared encoder + symmetric pooling makes detection order irrelevant.
        features = torch.cat((encoded.sum(2), encoded.max(2).values), dim=-1)
        _, state = self.gru(features)
        return self.head(state[-1])


def metrics(truth, predictions, classes):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for target, predicted in zip(truth, predictions):
        matrix[target, predicted] += 1
    tp = np.diag(matrix)
    support = matrix.sum(1)
    precision = tp / np.maximum(matrix.sum(0), 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(accuracy=float(tp.sum()/max(matrix.sum(),1)), macro_f1=float(f1.mean()),
                per_class_accuracy=recall.tolist(), per_class_f1=f1.tolist(),
                support=support.tolist(), confusion_matrix=matrix.tolist())
