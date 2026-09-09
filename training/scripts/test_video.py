from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp


DATASET_DIR = Path("AzSLD_Words_100")
MODEL_DIR = Path("training/models")
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)


def get_first_video():
    videos = sorted(DATASET_DIR.glob("*/*.mp4"))

    if not videos:
        raise RuntimeError("No MP4 files found.")

    return videos[0]


def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        return

    print("Downloading MediaPipe model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("MediaPipe model downloaded.")


def main():
    video_path = get_first_video()

    print(f"Video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")

    download_model()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    processed = 0
    detected = 0

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                break

            processed += 1

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                processed * 1000 / max(fps, 1)
            )

            result = landmarker.detect_for_video(
                image,
                timestamp_ms
            )

            if result.hand_landmarks:
                detected += 1

    cap.release()

    detection_rate = (
        detected / processed * 100
        if processed
        else 0
    )

    print()
    print("=" * 50)
    print("TEST RESULT")
    print("=" * 50)
    print(f"Class:           {video_path.parent.name}")
    print(f"Processed:       {processed}")
    print(f"Hands detected:  {detected}")
    print(f"Detection rate:  {detection_rate:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()