import cv2
import mediapipe as mp
import time


def draw_status_box(image, face_detected, left_hand_detected, right_hand_detected, fps):
    """
    Draws simple detection status text on the camera frame.
    """
    status_lines = [
        f"FPS: {fps:.1f}",
        f"Face: {'YES' if face_detected else 'NO'}",
        f"Left hand: {'YES' if left_hand_detected else 'NO'}",
        f"Right hand: {'YES' if right_hand_detected else 'NO'}",
        "Press Q to quit",
    ]

    x, y = 20, 30
    for line in status_lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30


def print_sample_landmarks(results):
    """
    Prints a few useful landmark positions to the terminal.
    Coordinates are normalized:
    x and y are usually between 0 and 1.
    z is relative depth.
    """

    if results.face_landmarks:
        nose_tip = results.face_landmarks.landmark[1]
        print(
            f"Face nose tip: "
            f"x={nose_tip.x:.3f}, y={nose_tip.y:.3f}, z={nose_tip.z:.3f}"
        )

    if results.left_hand_landmarks:
        left_index_tip = results.left_hand_landmarks.landmark[8]
        print(
            f"Left index tip: "
            f"x={left_index_tip.x:.3f}, y={left_index_tip.y:.3f}, z={left_index_tip.z:.3f}"
        )

    if results.right_hand_landmarks:
        right_index_tip = results.right_hand_landmarks.landmark[8]
        print(
            f"Right index tip: "
            f"x={right_index_tip.x:.3f}, y={right_index_tip.y:.3f}, z={right_index_tip.z:.3f}"
        )


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera. Try changing VideoCapture(0) to VideoCapture(1).")
        return

    # Optional camera resolution. Lower this if FPS is bad.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previous_time = time.time()
    frame_count = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            success, frame = cap.read()

            if not success:
                print("Warning: Failed to read frame from camera.")
                break

            # Mirror image so it feels natural like a webcam preview.
            frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB, OpenCV uses BGR.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Improves performance because MediaPipe does not need to write to the image.
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Draw face landmarks.
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.face_landmarks,
                    connections=mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style(),
                )

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.face_landmarks,
                    connections=mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style(),
                )

            # Draw left hand landmarks.
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.left_hand_landmarks,
                    connections=mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_hand_connections_style(),
                )

            # Draw right hand landmarks.
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.right_hand_landmarks,
                    connections=mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_hand_connections_style(),
                )

            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            face_detected = results.face_landmarks is not None
            left_hand_detected = results.left_hand_landmarks is not None
            right_hand_detected = results.right_hand_landmarks is not None

            draw_status_box(
                frame,
                face_detected,
                left_hand_detected,
                right_hand_detected,
                fps,
            )

            # Print landmark samples every 30 frames so the terminal does not become unusable.
            frame_count += 1
            if frame_count % 30 == 0:
                print_sample_landmarks(results)

            cv2.imshow("Reactify Camera Keypoint Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()