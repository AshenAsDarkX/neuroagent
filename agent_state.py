from typing import List
from models import DetectedElement


def infer_state(elements: List[DetectedElement]) -> dict:
    names = [e.name.lower() for e in elements]

    return {
        "spotify_visible": any("spotify" in n for n in names),
        "chrome_visible": any("chrome" in n for n in names),
        "explorer_visible": any("explorer" in n for n in names),
        "search_visible": any("search" in n for n in names),
        "start_visible": any("windows" in n for n in names),
        "close_visible": any("close" in n for n in names),
    }