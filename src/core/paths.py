from pathlib import Path
import sys

def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()

        # macOS .app:
        # Reactify.app/Contents/MacOS/Reactify
        if exe_path.parent.name == "MacOS" and exe_path.parent.parent.name == "Contents":
            return exe_path.parents[3]

        # Windows onedir:
        return exe_path.parent

    return Path(__file__).resolve().parents[2]