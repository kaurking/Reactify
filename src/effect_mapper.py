from typing import Optional, Dict, Any


EFFECTS = {
    "absolute_cinema": {
        "image": "assets/images/Absolute_Cinema.png",
        "sound": "assets/sounds/vine-boom.mp3",
        "duration_frames": 45,
    },
}


def get_effect_for_gesture(gesture_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Returns effect config for a detected gesture.

    Example:
        "absolute_cinema" -> image + sound config
    """

    if gesture_name is None:
        return None

    return EFFECTS.get(gesture_name)