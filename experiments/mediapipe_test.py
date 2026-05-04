import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent.parent
HAND_MODEL = BASE_DIR / "models" / "hand_landmarker.task"
FACE_MODEL = BASE_DIR / "models" / "face_landmarker.task"


def draw_landmarks(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 2, color, -1)


def main():
    if not HAND_MODEL.exists():
        raise FileNotFoundError(f"Missing: {HAND_MODEL}")

    if not FACE_MODEL.exists():
        raise FileNotFoundError(f"Missing: {FACE_MODEL}")

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:

        print("MediaPipe Tasks test running. Press Q to quit.")

        frame_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(frame_index * (1000 / 30))
            frame_index += 1

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw hand landmarks
            for hand_landmarks in hand_result.hand_landmarks:
                draw_landmarks(frame, hand_landmarks, color=(0, 255, 0))

            # Draw face landmarks
            for face_landmarks in face_result.face_landmarks:
                draw_landmarks(frame, face_landmarks, color=(255, 0, 0))

            cv2.imshow("Reactify MediaPipe Tasks Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()