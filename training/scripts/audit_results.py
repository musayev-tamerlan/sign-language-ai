"""Summarise weak classes and largest held-out confusions from an evaluation report."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = list(zip(
        report["labels"], report["support"], report["per_class_accuracy"],
        report["per_class_f1"],
    ))
    print("Worst classes (label, test videos, recall, F1):")
    for row in sorted(rows, key=lambda item: item[3])[:args.limit]:
        print(f"{row[0]!r}: n={row[1]}, recall={row[2]:.3f}, f1={row[3]:.3f}")
    print("\nLargest mistakes (actual -> predicted, count):")
    labels, matrix, errors = report["labels"], report["confusion_matrix"], []
    for actual, row in enumerate(matrix):
        for predicted, count in enumerate(row):
            if actual != predicted and count:
                errors.append((count, labels[actual], labels[predicted]))
    for count, actual, predicted in sorted(errors, reverse=True)[:args.limit]:
        print(f"{actual!r} -> {predicted!r}: {count}")


if __name__ == "__main__":
    main()
