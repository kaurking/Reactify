from pathlib import Path
import sys

def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent

    return Path(__file__).resolve().parents[2]