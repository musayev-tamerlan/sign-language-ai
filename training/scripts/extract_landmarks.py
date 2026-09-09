import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "AzSLD_Words_100"
OUTPUT_DIR = PROJECT_ROOT / "training" / "data" / "landmarks"

MAX_FRAMES = 64


def extract_video(video_path, landmarker, timestamp_base):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has 0 frames")

    # Select frames evenly
    if total_frames <= MAX_FRAMES:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(
            0,
            total_frames - 1,
            MAX_FRAMES
        ).astype(int).tolist()

    results = []

    current_frame = 0
    target_pos = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if target_pos >= len(frame_indices):
            break

        if current_frame != frame_indices[target_pos]:
            current_frame += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = timestamp_base + current_frame * 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        features = np.zeros(126, dtype=np.float32)

        # First hand
        if len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]

            for i, landmark in enumerate(hand):
                features[i * 3] = landmark.x
                features[i * 3 + 1] = landmark.y
                features[i * 3 + 2] = landmark.z

        # Second hand
        if len(result.hand_landmarks) > 1:
            hand = result.hand_landmarks[1]

            offset = 63

            for i, landmark in enumerate(hand):
                features[offset + i * 3] = landmark.x
                features[offset + i * 3 + 1] = landmark.y
                features[offset + i * 3 + 2] = landmark.z

        results.append(features)

        target_pos += 1
        current_frame += 1

    cap.release()

    if len(results) == 0:
        raise RuntimeError("No frames extracted")

    return np.asarray(results, dtype=np.float32)


def main():

    print("=" * 60)
    print("AZSL LANDMARK EXTRACTION")
    print("=" * 60)

    videos = sorted(DATASET_DIR.rglob("*.mp4"))

    print(f"Dataset: {DATASET_DIR}")
    print(f"Videos:  {len(videos)}")
    print(f"Output:  {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # MediaPipe model
    model_path = PROJECT_ROOT / "training" / "models" / "hand_landmarker.task"

    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found:\n{model_path}"
        )

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2
    )

    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )

    ok = 0
    failed = 0
    skipped = 0

    errors = []

    for index, video_path in enumerate(
        tqdm(videos, desc="Extracting", unit="video")
    ):

        relative = video_path.relative_to(DATASET_DIR)

        class_name = relative.parts[0]

        output_class_dir = OUTPUT_DIR / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_class_dir / (
            video_path.stem + ".npy"
        )

        # Already extracted
        if output_path.exists():
            skipped += 1
            continue

        try:

            timestamp_base = index * 1_000_000

            landmarks = extract_video(
                video_path,
                landmarker,
                timestamp_base
            )

            np.save(output_path, landmarks)

            ok += 1

        except Exception as e:

            failed += 1

            error_text = (
                f"{video_path} -> "
                f"{type(e).__name__}: {e}"
            )

            errors.append(error_text)

            # Print first errors immediately
            if len(errors) <= 20:
                print()
                print("ERROR:")
                print(error_text)
                print()

    landmarker.close()

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Total videos: {len(videos)}")
    print(f"New:          {ok}")
    print(f"Skipped:      {skipped}")
    print(f"Failed:       {failed}")
    print()

    if errors:

        print("=" * 60)
        print("FIRST ERRORS")
        print("=" * 60)

        for error in errors[:20]:
            print(error)

        print()

        error_file = OUTPUT_DIR / "extraction_errors.txt"

        error_file.write_text(
            "\n".join(errors),
            encoding="utf-8"
        )

        print(f"Full error log:")
        print(error_file)

    print()
    print("Landmarks directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()