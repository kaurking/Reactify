from pathlib import Path
import json
import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
POSE_DIR = BASE_DIR / "assets" / "poses"
OUTPUT_PATH = POSE_DIR / "absolute_cinema.json"

GESTURE_NAME = "absolute_cinema"

SAMPLE_COUNT = 5
SECONDS_BETWEEN_SAMPLES = 2.0
INITIAL_COUNTDOWN_SECONDS = 3.0
THRESHOLD = 0.55

POSE_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


def normalize_pose(landmarks):
    left_shoulder = landmarks[POSE_LANDMARKS["left_shoulder"]]
    right_shoulder = landmarks[POSE_LANDMARKS["right_shoulder"]]

    center_x = (left_shoulder.x + right_shoulder.x) / 2
    center_y = (left_shoulder.y + right_shoulder.y) / 2

    shoulder_width = math.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2
        + (left_shoulder.y - right_shoulder.y) ** 2
    )

    if shoulder_width == 0:
        return None

    normalized = {}

    for name, index in POSE_LANDMARKS.items():
        point = landmarks[index]

        normalized[name] = [
            (point.x - center_x) / shoulder_width,
            (point.y - center_y) / shoulder_width,
        ]

    return normalized


def create_new_pose_file():
    return {
        "gesture": GESTURE_NAME,
        "threshold": THRESHOLD,
        "samples": [],
    }


def save_pose_file(pose_file):
    POSE_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(pose_file, file, indent=2)

    print(f"[INFO] Saved {len(pose_file['samples'])} samples to {OUTPUT_PATH}")


def draw_text(frame, text, position, scale=1.0, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    pose_file = create_new_pose_file()

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Automatic pose sampler running.")
    print(f"Taking {SAMPLE_COUNT} samples, one every {SECONDS_BETWEEN_SAMPLES} seconds.")
    print("Press Q to cancel.")

    start_time = time.time()
    next_sample_time = start_time + INITIAL_COUNTDOWN_SECONDS
    samples_taken = 0
    last_status_message = ""

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            has_pose = bool(result.pose_landmarks)
            now = time.time()

            seconds_until_sample = max(0.0, next_sample_time - now)

            draw_text(
                frame,
                f"Automatic sampler: {samples_taken}/{SAMPLE_COUNT}",
                (30, 40),
                scale=0.8,
            )

            draw_text(
                frame,
                "Get into the Absolute Cinema pose",
                (30, 80),
                scale=0.8,
            )

            draw_text(
                frame,
                f"Pose detected: {'YES' if has_pose else 'NO'}",
                (30, 120),
                scale=0.8,
            )

            if samples_taken < SAMPLE_COUNT:
                draw_text(
                    frame,
                    f"Next snapshot in: {seconds_until_sample:.1f}s",
                    (30, 170),
                    scale=1.0,
                    thickness=2,
                )

            if last_status_message:
                draw_text(
                    frame,
                    last_status_message,
                    (30, 220),
                    scale=0.8,
                    thickness=2,
                )

            cv2.imshow("Automatic Pose Sampler", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Cancelled.")
                break

            if samples_taken >= SAMPLE_COUNT:
                save_pose_file(pose_file)
                print("Finished sampling.")
                break

            if now >= next_sample_time:
                if not has_pose:
                    last_status_message = "No pose detected. Retrying..."
                    print("[WARN] No pose detected. Retrying in 2 seconds.")
                    next_sample_time = now + SECONDS_BETWEEN_SAMPLES
                    continue

                landmarks = result.pose_landmarks[0]
                normalized_pose = normalize_pose(landmarks)

                if normalized_pose is None:
                    last_status_message = "Could not normalize pose. Retrying..."
                    print("[WARN] Could not normalize pose. Retrying in 2 seconds.")
                    next_sample_time = now + SECONDS_BETWEEN_SAMPLES
                    continue

                pose_file["samples"].append(normalized_pose)
                samples_taken += 1

                last_status_message = f"Saved sample {samples_taken}/{SAMPLE_COUNT}"
                print(f"[INFO] Saved sample {samples_taken}/{SAMPLE_COUNT}")

                next_sample_time = now + SECONDS_BETWEEN_SAMPLES

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()