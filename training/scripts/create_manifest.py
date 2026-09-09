from pathlib import Path
import csv
import zipfile

ZIP_PATH = Path("AzSLD_Words_100.zip")
OUTPUT_PATH = Path("training/data/manifest.csv")


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {ZIP_PATH}"
        )

    videos = []

    print("Scanning dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith(".mp4"):
                continue

            parts = Path(name).parts

            if len(parts) < 3:
                continue

            # AzSLD_Words_100 / CLASS / file.mp4
            label = parts[1]

            videos.append({
                "path": name,
                "label": label,
            })

    labels = sorted(
        {item["label"] for item in videos}
    )

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "class_id",
            ]
        )

        writer.writeheader()

        for item in videos:
            writer.writerow({
                "path": item["path"],
                "label": item["label"],
                "class_id": label_to_id[item["label"]],
            })

    print()
    print("Manifest created.")
    print(f"Videos:  {len(videos):,}")
    print(f"Classes: {len(labels):,}")
    print(f"Output:  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()