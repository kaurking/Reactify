import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.paths import app_base_dir


BASE_DIR = app_base_dir()
POSE_DIR = BASE_DIR / "assets" / "poses"
IMAGE_DIR = BASE_DIR / "assets" / "images"
SOUND_DIR = BASE_DIR / "assets" / "sounds"


def sanitize_gesture_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_ -]", "", name)
    name = name.replace(" ", "_").replace("-", "_")
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def get_profile_path(gesture: str) -> Path:
    gesture = sanitize_gesture_name(gesture)
    return POSE_DIR / f"{gesture}.json"


def list_profiles() -> List[Dict[str, Any]]:
    POSE_DIR.mkdir(parents=True, exist_ok=True)

    profiles = []

    for path in POSE_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            data["_path"] = str(path)
            profiles.append(data)

        except Exception as error:
            print(f"[WARN] Could not load profile {path}: {error}")

    return profiles


def load_profile(gesture: str) -> Optional[Dict[str, Any]]:
    path = get_profile_path(gesture)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_profile(profile: Dict[str, Any]) -> Path:
    POSE_DIR.mkdir(parents=True, exist_ok=True)

    gesture = sanitize_gesture_name(profile["gesture"])
    profile["gesture"] = gesture

    path = get_profile_path(gesture)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    return path


def delete_profile(gesture: str) -> bool:
    path = get_profile_path(gesture)

    if not path.exists():
        return False

    path.unlink()
    return True


def copy_asset_to_project(source_path: str, asset_type: str) -> str:
    """
    Copies selected user image/sound into assets/images or assets/sounds.

    Returns project-relative path, for example:
    assets/images/my_image.png
    """

    source = Path(source_path)

    if not source.exists():
        raise FileNotFoundError(f"File does not exist: {source}")

    if asset_type == "image":
        target_dir = IMAGE_DIR
        relative_prefix = "assets/images"
    elif asset_type == "sound":
        target_dir = SOUND_DIR
        relative_prefix = "assets/sounds"
    else:
        raise ValueError("asset_type must be 'image' or 'sound'")

    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / source.name

    if source.resolve() != target_path.resolve():
        shutil.copy2(source, target_path)

    return f"{relative_prefix}/{source.name}"