import re
from typing import List
from models import DetectedElement


def _to_state_key(name: str) -> str:
    """Convert element name to a snake_case state key, e.g. 'Google Chrome' → 'google_chrome_visible'."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{slug}_visible"


def infer_state(elements: List[DetectedElement]) -> dict:
    names = [e.name.lower() for e in elements]

    # Static keys — preserved for goal_evaluator compatibility
    state = {
        "spotify_visible"  : any("spotify"   in n for n in names),
        "chrome_visible"   : any("chrome"    in n for n in names),
        "explorer_visible" : any("explorer"  in n for n in names),
        "search_visible"   : any("search"    in n for n in names),
        "start_visible"    : any("windows"   in n for n in names),
        "close_visible"    : any("close"     in n for n in names),
    }

    # Dynamic layer: every detected element gets its own state key
    # Lets goal_evaluator handle arbitrary app names from LLM output
    for element in elements:
        key = _to_state_key(element.name)
        state[key] = True

    return state