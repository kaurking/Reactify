from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.paths import app_base_dir


BASE_DIR = app_base_dir()
POSE_MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
POSE_DIR = BASE_DIR / "assets" / "poses"


@dataclass
class DetectionResult:
    gesture: Optional[str]
    confidence: float
    landmarks: Optional[Dict[str, Any]] = None
    normalized_pose: Optional[Dict[str, Any]] = None
    normalized_face: Optional[Dict[str, Any]] = None
    normalized_hands: Optional[Dict[str, Any]] = None
    derived_features: Optional[Dict[str, float]] = None
    sample_data: Optional[Dict[str, Any]] = None


class GestureDetector:
    """
    MediaPipe Tasks-based detector using layered profile samples.

    Samples can include pose, face, hands, and derived feature layers. All point
    coordinates are normalized around the face so facecam-centered gestures can
    compare face expressions, arms, shoulders, wrists, hands, and fingers in one
    coordinate space.
    """

    POSE_LANDMARKS = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
    }

    FACE_LANDMARKS = {
        "nose": 1,
        "mouth_center": 13,
        "upper_lip": 13,
        "lower_lip": 14,
        "left_mouth_corner": 61,
        "right_mouth_corner": 291,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_eyebrow": 105,
        "right_eyebrow": 334,
        "chin": 152,
    }

    HAND_LANDMARKS = {
        "wrist": 0,
        "thumb_cmc": 1,
        "thumb_mcp": 2,
        "thumb_ip": 3,
        "thumb_tip": 4,
        "index_mcp": 5,
        "index_pip": 6,
        "index_dip": 7,
        "index_tip": 8,
        "middle_mcp": 9,
        "middle_pip": 10,
        "middle_dip": 11,
        "middle_tip": 12,
        "ring_mcp": 13,
        "ring_pip": 14,
        "ring_dip": 15,
        "ring_tip": 16,
        "pinky_mcp": 17,
        "pinky_pip": 18,
        "pinky_dip": 19,
        "pinky_tip": 20,
    }

    DEFAULT_LAYER_WEIGHTS = {
        "pose": 0.20,
        "face": 0.25,
        "hands": 0.25,
        "derived": 0.30,
    }

    def __init__(
        self,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_visibility: float = 0.4,
        debug: bool = True,
        debug_interval: float = 0.5,
    ):
        if not POSE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing pose model: {POSE_MODEL_PATH}\n"
                "Expected model filename: pose_landmarker_lite.task"
            )

        self.min_visibility = min_visibility
        self.debug = debug
        self.debug_interval = debug_interval
        self.last_debug_time = 0.0
        self.master_strictness = 50.0

        self.pose_templates = self._load_pose_templates()

        pose_base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )

        self.detector = vision.PoseLandmarker.create_from_options(pose_options)
        self.face_detector = None
        self.hand_detector = None

        if FACE_MODEL_PATH.exists():
            face_base_options = python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
            face_options = vision.FaceLandmarkerOptions(
                base_options=face_base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
        else:
            print(f"[WARN] Face model missing: {FACE_MODEL_PATH}")

        if HAND_MODEL_PATH.exists():
            hand_base_options = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
            hand_options = vision.HandLandmarkerOptions(
                base_options=hand_base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        else:
            print(f"[WARN] Hand model missing: {HAND_MODEL_PATH}")

    def detect(self, frame) -> DetectionResult:
        if frame is None:
            return DetectionResult(
                gesture=None,
                confidence=0.0,
                landmarks=None,
                normalized_pose=None,
                normalized_face=None,
                normalized_hands=None,
                derived_features=None,
                sample_data=None,
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = int(time.time() * 1000)
        pose_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        face_result = None
        if self.face_detector is not None:
            face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)

        hand_result = None
        if self.hand_detector is not None:
            hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)

        face_landmarks = None
        if face_result is not None and face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]

        if face_landmarks is None:
            self._debug_print("[FACE DEBUG] no face landmarks")
            return DetectionResult(
                gesture=None,
                confidence=0.0,
                landmarks=None,
                normalized_pose=None,
                normalized_face=None,
                normalized_hands=None,
                derived_features=None,
                sample_data=None,
            )

        landmark_dict = {
            "face": self._landmarks_to_dict(
                face_landmarks,
                self._face_landmark_names(),
            )
        }

        pose_landmarks = None
        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]
            landmark_dict["pose"] = self._landmarks_to_dict(
                pose_landmarks,
                self._pose_landmark_names(),
            )

        hand_landmarks = []
        handedness = []
        if hand_result is not None and hand_result.hand_landmarks:
            hand_landmarks = hand_result.hand_landmarks
            handedness = getattr(hand_result, "handedness", []) or []
            landmark_dict["hands"] = self._hands_to_dict(hand_landmarks, handedness)

        reference = self._get_face_normalization_reference(face_landmarks)
        if reference is None:
            self._debug_print("[FACE DEBUG] face found, but could not normalize")
            return DetectionResult(
                gesture=None,
                confidence=0.0,
                landmarks=landmark_dict,
                normalized_pose=None,
                normalized_face=None,
                normalized_hands=None,
                derived_features=None,
                sample_data=None,
            )

        face_center, face_scale = reference
        normalized_pose = self._normalize_pose_with_reference(
            pose_landmarks,
            face_center,
            face_scale,
        )
        normalized_face = self._normalize_face(
            face_landmarks,
            face_center,
            face_scale,
        )
        normalized_hands = self._normalize_hands(
            hand_landmarks,
            handedness,
            face_center,
            face_scale,
        )
        derived_features = self._build_derived_features(
            normalized_pose,
            normalized_face,
            normalized_hands,
        )
        sample_data = self._build_sample_data(
            normalized_pose,
            normalized_face,
            normalized_hands,
            derived_features,
        )

        match = self._match_sample(sample_data)

        if match is not None:
            gesture_name, distance, threshold = match
            confidence = self._distance_to_confidence(distance, threshold)

            return DetectionResult(
                gesture=gesture_name,
                confidence=confidence,
                landmarks=landmark_dict,
                normalized_pose=normalized_pose,
                normalized_face=normalized_face,
                normalized_hands=normalized_hands,
                derived_features=derived_features,
                sample_data=sample_data,
            )

        return DetectionResult(
            gesture=None,
            confidence=0.0,
            landmarks=landmark_dict,
            normalized_pose=normalized_pose,
            normalized_face=normalized_face,
            normalized_hands=normalized_hands,
            derived_features=derived_features,
            sample_data=sample_data,
        )

    def reload_templates(self):
        self.pose_templates = self._load_pose_templates()

    def set_master_strictness(self, strictness: float):
        self.master_strictness = max(0.0, min(100.0, float(strictness)))

    def _load_pose_templates(self) -> List[Dict[str, Any]]:
        templates = []

        if not POSE_DIR.exists():
            print(f"[WARN] Pose directory missing: {POSE_DIR}")
            return templates

        json_files = list(POSE_DIR.glob("*.json"))

        if not json_files:
            print(f"[WARN] No pose JSON files found in: {POSE_DIR}")
            return templates

        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                gesture = data.get("gesture")
                manual_threshold = float(data.get("threshold", 0.55))
                samples = data.get("samples", [])
                detection_layers = data.get("detection_layers", ["pose"])
                layer_weights = data.get("layer_weights", self.DEFAULT_LAYER_WEIGHTS)

                if not gesture:
                    print(f"[WARN] Missing gesture name in {path}")
                    continue

                if not isinstance(samples, list):
                    print(f"[WARN] Samples must be a list in {path}")
                    continue

                valid_samples = self._filter_valid_samples(samples)

                if not valid_samples:
                    print(f"[WARN] No valid samples found in {path}")
                    continue

                average_sample = {
                    "pose": self._average_layer(valid_samples, "pose"),
                    "face": self._average_layer(valid_samples, "face"),
                    "hands": self._average_layer(valid_samples, "hands"),
                    "derived": self._average_layer(valid_samples, "derived"),
                }

                template = {
                    "gesture": gesture,
                    "threshold": manual_threshold,
                    "manual_threshold": manual_threshold,
                    "detection_layers": detection_layers,
                    "layer_weights": layer_weights,
                    "average_sample": average_sample,
                    "source": str(path),
                }

                sample_distances = [
                    self._sample_distance(sample, template)
                    for sample in valid_samples
                ]
                sample_distances = [
                    distance for distance in sample_distances if distance is not None
                ]

                average_sample_distance = (
                    sum(sample_distances) / len(sample_distances)
                    if sample_distances
                    else 0.0
                )
                max_sample_distance = max(sample_distances) if sample_distances else 0.0

                auto_threshold = max_sample_distance * 1.4
                final_threshold = manual_threshold
                template["threshold"] = final_threshold
                template["average_sample_distance"] = average_sample_distance
                template["max_sample_distance"] = max_sample_distance
                template["suggested_threshold"] = auto_threshold
                template["sample_count"] = len(valid_samples)

                templates.append(template)

                print(
                    f"[INFO] Loaded layered profile for '{gesture}' "
                    f"from {len(valid_samples)} samples. "
                    f"layers={detection_layers}, "
                    f"avg_dist={average_sample_distance:.3f}, "
                    f"max_dist={max_sample_distance:.3f}, "
                    f"suggested_threshold={auto_threshold:.3f}, "
                    f"threshold={final_threshold:.3f}"
                )

            except json.JSONDecodeError:
                print(f"[WARN] Invalid JSON in pose file: {path}")
            except Exception as error:
                print(f"[WARN] Could not load pose template {path}: {error}")

        return templates

    def _filter_valid_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_samples = []

        for sample_index, sample in enumerate(samples):
            layered_sample = self._normalize_sample_format(sample)

            if layered_sample is None:
                print(f"[WARN] Skipping sample {sample_index}: no usable layers")
                continue

            valid_samples.append(layered_sample)

        return valid_samples

    def _normalize_sample_format(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(sample, dict):
            return None

        has_layer = any(
            isinstance(sample.get(layer_name), dict)
            for layer_name in ("pose", "face", "hands", "derived")
        )

        if has_layer:
            layered = {}

            for layer_name in ("pose", "face", "hands", "derived"):
                layer = sample.get(layer_name)

                if isinstance(layer, dict) and layer:
                    layered[layer_name] = layer

            return layered if layered else None

        pose_layer = {}

        for landmark_name in self.POSE_LANDMARKS.keys():
            value = sample.get(landmark_name)

            if self._is_point(value):
                pose_layer[landmark_name] = value

        if not pose_layer:
            return None

        return {"pose": pose_layer}

    def _average_layer(self, samples: List[Dict[str, Any]], layer_name: str) -> Dict[str, Any]:
        values_by_key = {}

        for sample in samples:
            layer = sample.get(layer_name)

            if not isinstance(layer, dict):
                continue

            for key, value in layer.items():
                if self._is_point(value):
                    values_by_key.setdefault(key, {"x": [], "y": []})
                    values_by_key[key]["x"].append(float(value[0]))
                    values_by_key[key]["y"].append(float(value[1]))
                elif isinstance(value, (int, float)):
                    values_by_key.setdefault(key, {"scalar": []})
                    values_by_key[key]["scalar"].append(float(value))

        average = {}

        for key, values in values_by_key.items():
            if "scalar" in values:
                average[key] = sum(values["scalar"]) / len(values["scalar"])
            else:
                average[key] = [
                    sum(values["x"]) / len(values["x"]),
                    sum(values["y"]) / len(values["y"]),
                ]

        return average

    def _get_face_normalization_reference(
        self,
        face_landmarks,
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        if face_landmarks is None:
            return None

        face_points = [[point.x, point.y] for point in face_landmarks]

        if not face_points:
            return None

        xs = [point[0] for point in face_points]
        ys = [point[1] for point in face_points]

        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        face_width = max(xs) - min(xs)
        face_height = max(ys) - min(ys)
        face_scale = max(face_width, face_height)

        left_mouth = face_landmarks[self.FACE_LANDMARKS["left_mouth_corner"]]
        right_mouth = face_landmarks[self.FACE_LANDMARKS["right_mouth_corner"]]
        mouth_width = self._point_distance(
            [left_mouth.x, left_mouth.y],
            [right_mouth.x, right_mouth.y],
        )

        face_scale = max(face_scale, mouth_width)

        if face_scale <= 0:
            return None

        return (center_x, center_y), face_scale

    def _normalize_pose_with_reference(
        self,
        landmarks,
        face_center: Tuple[float, float],
        face_scale: float,
    ) -> Optional[Dict[str, List[float]]]:
        if landmarks is None or face_scale <= 0:
            return None

        normalized = {}

        for landmark_name, index in self.POSE_LANDMARKS.items():
            point = landmarks[index]

            normalized[landmark_name] = [
                (point.x - face_center[0]) / face_scale,
                (point.y - face_center[1]) / face_scale,
            ]

        return normalized

    def _normalize_face(
        self,
        face_landmarks,
        face_center: Tuple[float, float],
        face_scale: float,
    ) -> Optional[Dict[str, List[float]]]:
        if face_landmarks is None or face_scale <= 0:
            return None

        normalized = {}

        for landmark_name, index in self.FACE_LANDMARKS.items():
            if index >= len(face_landmarks):
                continue

            point = face_landmarks[index]
            normalized[landmark_name] = [
                (point.x - face_center[0]) / face_scale,
                (point.y - face_center[1]) / face_scale,
            ]

        return normalized if normalized else None

    def _normalize_hands(
        self,
        hand_landmarks,
        handedness,
        face_center: Tuple[float, float],
        face_scale: float,
    ) -> Optional[Dict[str, List[float]]]:
        if not hand_landmarks or face_scale <= 0:
            return None

        normalized = {}

        for hand_index, landmarks in enumerate(hand_landmarks):
            hand_label = self._hand_label(handedness, hand_index)

            for landmark_name, landmark_index in self.HAND_LANDMARKS.items():
                if landmark_index >= len(landmarks):
                    continue

                point = landmarks[landmark_index]
                normalized[f"{hand_label}_{landmark_name}"] = [
                    (point.x - face_center[0]) / face_scale,
                    (point.y - face_center[1]) / face_scale,
                ]

        return normalized if normalized else None

    def _build_derived_features(
        self,
        normalized_pose,
        normalized_face,
        normalized_hands,
    ) -> Optional[Dict[str, float]]:
        if normalized_face is None:
            return None

        features = {}
        mouth = normalized_face.get("mouth_center")
        nose = normalized_face.get("nose")
        upper_lip = normalized_face.get("upper_lip")
        lower_lip = normalized_face.get("lower_lip")
        left_mouth = normalized_face.get("left_mouth_corner")
        right_mouth = normalized_face.get("right_mouth_corner")
        left_eye = normalized_face.get("left_eye_outer")
        right_eye = normalized_face.get("right_eye_outer")
        left_eyebrow = normalized_face.get("left_eyebrow")
        right_eyebrow = normalized_face.get("right_eyebrow")
        chin = normalized_face.get("chin")

        if upper_lip and lower_lip:
            features["mouth_opening"] = self._point_distance(upper_lip, lower_lip)

        if left_mouth and right_mouth:
            features["mouth_width"] = self._point_distance(left_mouth, right_mouth)

        if "mouth_opening" in features and features.get("mouth_width", 0) > 0:
            features["mouth_open_ratio"] = (
                features["mouth_opening"] / features["mouth_width"]
            )

        if left_eye and right_eye:
            features["eye_line_tilt"] = right_eye[1] - left_eye[1]
            features["eye_distance"] = self._point_distance(left_eye, right_eye)

        if nose and chin:
            features["nose_to_chin"] = self._point_distance(nose, chin)

        if left_eyebrow and left_eye:
            features["left_eyebrow_raise"] = left_eye[1] - left_eyebrow[1]

        if right_eyebrow and right_eye:
            features["right_eyebrow_raise"] = right_eye[1] - right_eyebrow[1]

        if "left_eyebrow_raise" in features and "right_eyebrow_raise" in features:
            features["eyebrow_raise_difference"] = (
                features["left_eyebrow_raise"] - features["right_eyebrow_raise"]
            )

        if normalized_pose:
            features.update(self._build_pose_derived_features(normalized_pose, mouth, nose))

        if normalized_hands:
            features.update(
                self._build_hand_derived_features(
                    normalized_hands,
                    normalized_pose,
                    mouth,
                    nose,
                )
            )

        return features if features else None

    def _build_pose_derived_features(self, normalized_pose, mouth, nose) -> Dict[str, float]:
        features = {}
        left_wrist = normalized_pose.get("left_wrist")
        right_wrist = normalized_pose.get("right_wrist")
        left_elbow = normalized_pose.get("left_elbow")
        right_elbow = normalized_pose.get("right_elbow")
        left_shoulder = normalized_pose.get("left_shoulder")
        right_shoulder = normalized_pose.get("right_shoulder")

        if mouth and left_wrist:
            features["left_wrist_to_mouth"] = self._point_distance(left_wrist, mouth)

        if mouth and right_wrist:
            features["right_wrist_to_mouth"] = self._point_distance(right_wrist, mouth)

        if nose and left_wrist:
            features["left_wrist_to_nose"] = self._point_distance(left_wrist, nose)

        if nose and right_wrist:
            features["right_wrist_to_nose"] = self._point_distance(right_wrist, nose)

        wrist_mouth_distances = [
            features[key]
            for key in ("left_wrist_to_mouth", "right_wrist_to_mouth")
            if key in features
        ]
        if wrist_mouth_distances:
            features["closest_wrist_to_mouth"] = min(wrist_mouth_distances)

        if left_shoulder and right_shoulder:
            features["shoulder_width"] = self._point_distance(left_shoulder, right_shoulder)

        if left_elbow and left_wrist:
            features["left_forearm_length"] = self._point_distance(left_elbow, left_wrist)

        if right_elbow and right_wrist:
            features["right_forearm_length"] = self._point_distance(right_elbow, right_wrist)

        if left_shoulder and left_elbow:
            features["left_upper_arm_length"] = self._point_distance(left_shoulder, left_elbow)

        if right_shoulder and right_elbow:
            features["right_upper_arm_length"] = self._point_distance(right_shoulder, right_elbow)

        return features

    def _build_hand_derived_features(
        self,
        normalized_hands,
        normalized_pose,
        mouth,
        nose,
    ) -> Dict[str, float]:
        features = {}

        for hand_label in self._detected_hand_labels(normalized_hands):
            wrist = normalized_hands.get(f"{hand_label}_wrist")
            thumb_tip = normalized_hands.get(f"{hand_label}_thumb_tip")
            index_tip = normalized_hands.get(f"{hand_label}_index_tip")
            middle_tip = normalized_hands.get(f"{hand_label}_middle_tip")
            ring_tip = normalized_hands.get(f"{hand_label}_ring_tip")
            pinky_tip = normalized_hands.get(f"{hand_label}_pinky_tip")

            fingertips = [
                point
                for point in (thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip)
                if point is not None
            ]

            if wrist and mouth:
                features[f"{hand_label}_hand_to_mouth"] = self._point_distance(wrist, mouth)

            if wrist and nose:
                features[f"{hand_label}_hand_to_nose"] = self._point_distance(wrist, nose)

            if wrist and index_tip:
                features[f"{hand_label}_index_tip_to_wrist"] = self._point_distance(
                    index_tip,
                    wrist,
                )

            if wrist and fingertips:
                distances = [
                    self._point_distance(fingertip, wrist)
                    for fingertip in fingertips
                ]
                features[f"{hand_label}_finger_to_wrist_average"] = (
                    sum(distances) / len(distances)
                )

            if thumb_tip and index_tip:
                features[f"{hand_label}_thumb_tip_to_index_tip"] = self._point_distance(
                    thumb_tip,
                    index_tip,
                )

            if index_tip and pinky_tip:
                features[f"{hand_label}_hand_span"] = self._point_distance(
                    index_tip,
                    pinky_tip,
                )

            if normalized_pose and wrist:
                for side in ("left", "right"):
                    shoulder = normalized_pose.get(f"{side}_shoulder")
                    if shoulder:
                        features[f"{hand_label}_hand_to_{side}_shoulder"] = (
                            self._point_distance(wrist, shoulder)
                        )

        return features

    @staticmethod
    def _build_sample_data(
        normalized_pose,
        normalized_face,
        normalized_hands,
        derived_features,
    ) -> Optional[Dict[str, Any]]:
        sample_data = {}

        if normalized_pose:
            sample_data["pose"] = normalized_pose

        if normalized_face:
            sample_data["face"] = normalized_face

        if normalized_hands:
            sample_data["hands"] = normalized_hands

        if derived_features:
            sample_data["derived"] = derived_features

        return sample_data if sample_data else None

    def _match_sample(
        self,
        current_sample: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[str, float, float]]:
        if not self.pose_templates:
            self._debug_print("[MATCH DEBUG] no templates loaded")
            return None

        if current_sample is None:
            return None

        best_match = None
        best_distance = float("inf")
        best_threshold = 0.0
        best_raw_threshold = 0.0

        for template in self.pose_templates:
            distance = self._sample_distance(current_sample, template)

            if distance is None:
                continue

            threshold = self._effective_template_threshold(template)

            if distance < best_distance:
                best_distance = distance
                best_match = template["gesture"]
                best_threshold = threshold
                best_raw_threshold = template["threshold"]

        self._debug_print(
            f"[MATCH DEBUG] best_match={best_match}, "
            f"best_distance={best_distance:.3f}, "
            f"threshold={best_threshold:.3f}, "
            f"profile_threshold={best_raw_threshold:.3f}, "
            f"master_strictness={self.master_strictness:.0f}"
        )

        if best_match is not None and best_distance <= best_threshold:
            return best_match, best_distance, best_threshold

        return None

    def _effective_template_threshold(self, template) -> float:
        threshold = float(template["threshold"])
        return threshold * self._master_strictness_threshold_multiplier()

    def _master_strictness_threshold_multiplier(self) -> float:
        # 0 is forgiving, 50 keeps profile thresholds unchanged, 100 is strict.
        return 1.75 - ((self.master_strictness / 100.0) * 1.5)

    def _sample_distance(self, current_sample, template) -> Optional[float]:
        detection_layers = template.get("detection_layers", ["pose"])
        layer_weights = template.get("layer_weights", self.DEFAULT_LAYER_WEIGHTS)

        weighted_total = 0.0
        weight_total = 0.0

        for layer_name in detection_layers:
            current_layer = current_sample.get(layer_name)
            template_layer = template["average_sample"].get(layer_name)
            distance = self._layer_distance(current_layer, template_layer)

            if distance is None:
                continue

            weight = float(layer_weights.get(layer_name, 1.0))

            weighted_total += distance * weight
            weight_total += weight

        if weight_total == 0:
            return None

        return weighted_total / weight_total

    def _layer_distance(self, current_layer, template_layer) -> Optional[float]:
        if not current_layer or not template_layer:
            return None

        total = 0.0
        count = 0

        for key, template_value in template_layer.items():
            if key not in current_layer:
                continue

            current_value = current_layer[key]

            if self._is_point(template_value) and self._is_point(current_value):
                distance = self._point_distance(current_value, template_value)
            elif isinstance(template_value, (int, float)) and isinstance(
                current_value,
                (int, float),
            ):
                distance = abs(float(current_value) - float(template_value))
            else:
                continue

            total += distance
            count += 1

        if count == 0:
            return None

        return total / count

    @staticmethod
    def _point_distance(a, b) -> float:
        return math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
        )

    @staticmethod
    def _is_point(value) -> bool:
        return (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        )

    @staticmethod
    def _distance_to_confidence(distance: float, threshold: float) -> float:
        if threshold <= 0:
            return 0.0

        ratio = distance / threshold
        confidence = 1.0 - (ratio * 0.8)

        return round(max(0.0, min(1.0, confidence)), 2)

    def _points_visible(self, points) -> bool:
        for point in points:
            visibility = getattr(point, "visibility", 1.0)
            presence = getattr(point, "presence", 1.0)

            if visibility < self.min_visibility or presence < self.min_visibility:
                return False

        return True

    @staticmethod
    def _pose_landmark_names() -> Dict[int, str]:
        return {
            0: "nose",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
        }

    @staticmethod
    def _face_landmark_names() -> Dict[int, str]:
        return {
            1: "nose",
            13: "mouth_center",
            14: "lower_lip",
            33: "left_eye_outer",
            61: "left_mouth_corner",
            105: "left_eyebrow",
            152: "chin",
            263: "right_eye_outer",
            291: "right_mouth_corner",
            334: "right_eyebrow",
        }

    @staticmethod
    def _hand_landmark_names() -> Dict[int, str]:
        return {
            0: "wrist",
            4: "thumb_tip",
            5: "index_mcp",
            8: "index_tip",
            9: "middle_mcp",
            12: "middle_tip",
            13: "ring_mcp",
            16: "ring_tip",
            17: "pinky_mcp",
            20: "pinky_tip",
        }

    def _hands_to_dict(self, hand_landmarks, handedness) -> Dict[str, Any]:
        hands = {}

        for hand_index, landmarks in enumerate(hand_landmarks):
            hand_label = self._hand_label(handedness, hand_index)
            hands[hand_label] = self._landmarks_to_dict(
                landmarks,
                self._hand_landmark_names(),
            )

        return hands

    @staticmethod
    def _hand_label(handedness, hand_index: int) -> str:
        if hand_index < len(handedness) and handedness[hand_index]:
            category = handedness[hand_index][0]
            category_name = getattr(category, "category_name", None)

            if category_name:
                return category_name.lower()

        return f"hand_{hand_index}"

    @staticmethod
    def _detected_hand_labels(normalized_hands) -> List[str]:
        labels = set()

        for key in normalized_hands.keys():
            parts = key.split("_")

            if len(parts) < 2:
                continue

            if parts[0] == "hand" and len(parts) >= 3:
                labels.add("_".join(parts[:2]))
            else:
                labels.add(parts[0])

        return sorted(labels)

    def _landmarks_to_dict(self, landmarks, names) -> Dict[str, Dict[str, float]]:
        landmark_dict = {}

        for index, point in enumerate(landmarks):
            name = names.get(index, f"landmark_{index}")

            landmark_dict[name] = {
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "visibility": getattr(point, "visibility", 1.0),
                "presence": getattr(point, "presence", 1.0),
            }

        return landmark_dict

    def _debug_print(self, message: str):
        if not self.debug:
            return

        now = time.time()

        if now - self.last_debug_time >= self.debug_interval:
            print(message)
            self.last_debug_time = now

    def close(self):
        if self.detector is not None:
            self.detector.close()

        if self.face_detector is not None:
            self.face_detector.close()

        if self.hand_detector is not None:
            self.hand_detector.close()
