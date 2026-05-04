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


BASE_DIR = Path(__file__).resolve().parent.parent
POSE_MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"
FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
POSE_DIR = BASE_DIR / "assets" / "poses"


@dataclass
class DetectionResult:
    gesture: Optional[str]
    confidence: float
    landmarks: Optional[Dict[str, Any]] = None
    normalized_pose: Optional[Dict[str, Any]] = None
    normalized_face: Optional[Dict[str, Any]] = None
    derived_features: Optional[Dict[str, float]] = None
    sample_data: Optional[Dict[str, Any]] = None


class GestureDetector:
    """
    MediaPipe Tasks-based detector using layered profile samples.

    Samples can include pose, face, and derived feature layers. All point
    coordinates are normalized around the shoulder midpoint and shoulder width
    so body, face, and hand-to-mouth distances share one coordinate space.
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
    }

    DEFAULT_LAYER_WEIGHTS = {
        "pose": 0.35,
        "face": 0.25,
        "derived": 0.40,
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

    def detect(self, frame) -> DetectionResult:
        if frame is None:
            return DetectionResult(
                gesture=None,
                confidence=0.0,
                landmarks=None,
                normalized_pose=None,
                normalized_face=None,
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

        if not pose_result.pose_landmarks:
            self._debug_print("[POSE DEBUG] no pose landmarks")
            return DetectionResult(
                gesture=None,
                confidence=0.0,
                landmarks=None,
                normalized_pose=None,
                normalized_face=None,
                derived_features=None,
                sample_data=None,
            )

        pose_landmarks = pose_result.pose_landmarks[0]
        landmark_dict = {
            "pose": self._landmarks_to_dict(pose_landmarks, self._pose_landmark_names())
        }

        face_landmarks = None
        if face_result is not None and face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            landmark_dict["face"] = self._landmarks_to_dict(
                face_landmarks,
                self._face_landmark_names(),
            )

        reference = self._get_pose_normalization_reference(
            pose_landmarks,
            require_visibility=True,
        )

        relaxed_reference = reference
        if relaxed_reference is None:
            relaxed_reference = self._get_pose_normalization_reference(
                pose_landmarks,
                require_visibility=False,
            )

        if relaxed_reference is None:
            self._debug_print("[POSE DEBUG] pose found, but shoulders could not normalize")
            return DetectionResult(
                gesture=None,
                confidence=0.0,
                landmarks=landmark_dict,
                normalized_pose=None,
                normalized_face=None,
                derived_features=None,
                sample_data=None,
            )

        shoulder_center, shoulder_width = relaxed_reference
        normalized_pose = self._normalize_pose_with_reference(
            pose_landmarks,
            shoulder_center,
            shoulder_width,
        )
        normalized_face = self._normalize_face(
            face_landmarks,
            shoulder_center,
            shoulder_width,
        )
        derived_features = self._build_derived_features(
            normalized_pose,
            normalized_face,
        )
        sample_data = self._build_sample_data(
            normalized_pose,
            normalized_face,
            derived_features,
        )

        if reference is None:
            self._debug_print("[POSE DEBUG] matching with relaxed landmark visibility")

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
                derived_features=derived_features,
                sample_data=sample_data,
            )

        return DetectionResult(
            gesture=None,
            confidence=0.0,
            landmarks=landmark_dict,
            normalized_pose=normalized_pose,
            normalized_face=normalized_face,
            derived_features=derived_features,
            sample_data=sample_data,
        )

    def reload_templates(self):
        self.pose_templates = self._load_pose_templates()

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
                final_threshold = max(manual_threshold, auto_threshold)
                template["threshold"] = final_threshold
                template["average_sample_distance"] = average_sample_distance
                template["max_sample_distance"] = max_sample_distance
                template["sample_count"] = len(valid_samples)

                templates.append(template)

                print(
                    f"[INFO] Loaded layered profile for '{gesture}' "
                    f"from {len(valid_samples)} samples. "
                    f"layers={detection_layers}, "
                    f"avg_dist={average_sample_distance:.3f}, "
                    f"max_dist={max_sample_distance:.3f}, "
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
            for layer_name in ("pose", "face", "derived")
        )

        if has_layer:
            layered = {}

            for layer_name in ("pose", "face", "derived"):
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

    def _get_pose_normalization_reference(
        self,
        landmarks,
        require_visibility: bool = True,
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        left_shoulder = landmarks[self.POSE_LANDMARKS["left_shoulder"]]
        right_shoulder = landmarks[self.POSE_LANDMARKS["right_shoulder"]]

        required_points = [
            landmarks[index] for index in self.POSE_LANDMARKS.values()
        ]

        if require_visibility and not self._points_visible(required_points):
            return None

        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2

        shoulder_width = self._point_distance(
            [left_shoulder.x, left_shoulder.y],
            [right_shoulder.x, right_shoulder.y],
        )

        if shoulder_width <= 0:
            return None

        return (center_x, center_y), shoulder_width

    def _normalize_pose_with_reference(
        self,
        landmarks,
        shoulder_center: Tuple[float, float],
        shoulder_width: float,
    ) -> Dict[str, List[float]]:
        normalized = {}

        for landmark_name, index in self.POSE_LANDMARKS.items():
            point = landmarks[index]

            normalized[landmark_name] = [
                (point.x - shoulder_center[0]) / shoulder_width,
                (point.y - shoulder_center[1]) / shoulder_width,
            ]

        return normalized

    def _normalize_face(
        self,
        face_landmarks,
        shoulder_center: Tuple[float, float],
        shoulder_width: float,
    ) -> Optional[Dict[str, List[float]]]:
        if face_landmarks is None or shoulder_width <= 0:
            return None

        normalized = {}

        for landmark_name, index in self.FACE_LANDMARKS.items():
            if index >= len(face_landmarks):
                continue

            point = face_landmarks[index]
            normalized[landmark_name] = [
                (point.x - shoulder_center[0]) / shoulder_width,
                (point.y - shoulder_center[1]) / shoulder_width,
            ]

        return normalized if normalized else None

    def _build_derived_features(
        self,
        normalized_pose,
        normalized_face,
    ) -> Optional[Dict[str, float]]:
        if normalized_pose is None or normalized_face is None:
            return None

        mouth = normalized_face.get("mouth_center")
        left_wrist = normalized_pose.get("left_wrist")
        right_wrist = normalized_pose.get("right_wrist")

        if mouth is None or left_wrist is None or right_wrist is None:
            return None

        left_distance = self._point_distance(left_wrist, mouth)
        right_distance = self._point_distance(right_wrist, mouth)

        return {
            "left_wrist_to_mouth": left_distance,
            "right_wrist_to_mouth": right_distance,
            "closest_wrist_to_mouth": min(left_distance, right_distance),
        }

    @staticmethod
    def _build_sample_data(
        normalized_pose,
        normalized_face,
        derived_features,
    ) -> Optional[Dict[str, Any]]:
        sample_data = {}

        if normalized_pose:
            sample_data["pose"] = normalized_pose

        if normalized_face:
            sample_data["face"] = normalized_face

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

        for template in self.pose_templates:
            distance = self._sample_distance(current_sample, template)

            if distance is None:
                continue

            if distance < best_distance:
                best_distance = distance
                best_match = template["gesture"]
                best_threshold = template["threshold"]

        self._debug_print(
            f"[MATCH DEBUG] best_match={best_match}, "
            f"best_distance={best_distance:.3f}, "
            f"threshold={best_threshold:.3f}"
        )

        if best_match is not None and best_distance <= best_threshold:
            return best_match, best_distance, best_threshold

        return None

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
            61: "left_mouth_corner",
            291: "right_mouth_corner",
        }

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
