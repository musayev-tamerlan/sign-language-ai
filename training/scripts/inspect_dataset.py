from pathlib import Path
import zipfile
from collections import Counter

ZIP_PATH = Path("AzSLD_Words_100.zip")

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {ZIP_PATH}")

print("Reading dataset...")

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    files = [
        name
        for name in z.namelist()
        if name.lower().endswith(".mp4")
    ]

classes = Counter()

for file in files:
    parts = Path(file).parts

    # Expected:
    # AzSLD_Words_100 / CLASS / video.mp4
    if len(parts) >= 3:
        classes[parts[1]] += 1

print()
print("=" * 60)
print("AzSLD Words 100")
print("=" * 60)

print(f"Videos:  {len(files):,}")
print(f"Classes: {len(classes):,}")

if classes:
    counts = list(classes.values())

    print(f"Min/class: {min(counts):,}")
    print(f"Max/class: {max(counts):,}")
    print(f"Avg/class: {sum(counts) / len(counts):.1f}")

print()
print("Classes:")
print("-" * 60)

for name, count in sorted(classes.items()):
    print(f"{name:<30} {count:>6}")

print("=" * 60)