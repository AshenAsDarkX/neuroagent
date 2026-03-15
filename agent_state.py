"""
agent_state.py

Goal satisfaction uses Win32 actual window titles — ground truth, works for
ANY app without hardcoding. OmniParser elements are still used for planning
context but NOT for deciding whether a goal is achieved.
"""

import re
from typing import List, Tuple
from models import DetectedElement

try:
    import win32gui
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False

TASKBAR_Y_THRESHOLD = 0.93
WINDOW_MODE_MIN_ELEMENTS = 5


# ---------------------------------------------------------------------------
# Win32 window title utilities  (ground truth — works for any app)
# ---------------------------------------------------------------------------

def get_open_window_titles() -> List[str]:
    """
    Return lowercase titles of all currently visible top-level windows.
    Uses Win32 API — works for every app, no hardcoding needed.
    """
    if not _WIN32_AVAILABLE:
        return []
    titles = []
    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if t and len(t.strip()) > 1:
                titles.append(t.lower().strip())
    try:
        win32gui.EnumWindows(_cb, None)
    except Exception:
        pass
    return titles


def _normalize_goal(goal: str) -> str:
    """Strip action verb prefixes so 'open Spotify' → 'spotify'."""
    g = goal.lower().strip()
    for prefix in ["open ", "launch ", "start ", "run ", "toggle ", "close ", "click "]:
        if g.startswith(prefix):
            g = g[len(prefix):]
            break
    return g.strip()


def window_open_for_goal(goal: str) -> bool:
    """
    Check whether an app matching the goal is currently open as a window.

    Algorithm:
      1. Normalize goal → app name  (e.g. 'open Google Chrome' → 'google chrome')
      2. Split into significant words (len > 2)
      3. Check if ANY open window title contains ALL significant words
         OR contains the full normalized name

    This works for any app — Chrome, Spotify, PyCharm, WhatsApp, VLC, etc.
    No hardcoded app list needed.
    """
    app_name = _normalize_goal(goal)
    if not app_name:
        return False

    titles = get_open_window_titles()
    if not titles:
        return False

    # Significant words (skip short words like 'the', 'a', 'my')
    sig_words = [w for w in app_name.split() if len(w) > 2]

    for title in titles:
        # Skip system non-app windows
        if title in ("program manager", "desktop window manager", ""):
            continue
        # Full name match
        if app_name in title:
            return True
        # All significant words present in title
        if sig_words and all(w in title for w in sig_words):
            return True

    return False


# ---------------------------------------------------------------------------
# OmniParser-based state  (used for planning context, NOT goal satisfaction)
# ---------------------------------------------------------------------------

def _is_taskbar(element: DetectedElement, img_h: int = 1080) -> bool:
    _, y1, _, y2 = element.bbox
    return ((y1 + y2) / 2) / img_h > TASKBAR_Y_THRESHOLD


def _to_state_key(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{slug}_visible"


def infer_state(elements: List[DetectedElement], img_h: int = 1080) -> dict:
    """
    Build a state dict.

    The state dict is used by:
    - AgentPlanner  → needs element names for context (planning)
    - GoalEvaluator → uses window_open_for_goal() (Win32), NOT these keys

    The static *_visible keys are kept for backward compatibility but
    goal_evaluator.py now uses Win32 window titles as the primary signal.
    """
    window_elements = [e for e in elements if not _is_taskbar(e, img_h)]
    all_names       = [e.name.lower() for e in elements]
    window_names    = [e.name.lower() for e in window_elements]
    in_window_mode  = len(window_elements) >= WINDOW_MODE_MIN_ELEMENTS

    state = {
        "window_open"             : in_window_mode,
        # True = user is on desktop; False = user is inside an app
        # Used by goal_evaluator to distinguish launch goals from in-app actions
        "was_on_desktop"          : not in_window_mode,
        "spotify_visible"         : any("spotify"  in n for n in window_names) and in_window_mode,
        "chrome_visible"          : any("chrome"   in n for n in window_names) and in_window_mode,
        "explorer_visible"        : any("explorer" in n for n in window_names) and in_window_mode,
        "taskbar_chrome_visible"  : any("chrome"   in n for n in all_names),
        "taskbar_spotify_visible" : any("spotify"  in n for n in all_names),
        # Live Win32 open windows for planner context
        "open_windows"            : get_open_window_titles(),
    }

    if in_window_mode:
        for element in window_elements:
            state[_to_state_key(element.name)] = True

    return state