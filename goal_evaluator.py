"""
goal_evaluator.py

Two types of goals:

1. APP LAUNCH goals (user on desktop, picking an app to open)
   e.g. "open Spotify", "Control Panel", "Google Chrome"
   → Satisfied when Win32 reports the app window is open
   → Verified AFTER the click

2. IN-APP ACTION goals (user inside an app, picking what to do)
   e.g. "View by Category", "Hardware and Sound settings", "Go back"
   → Satisfied IMMEDIATELY after the click — there is no window to verify
   → The action happened; we move on

The controller sets goal_type on the DetectedElement so we know
which type we're dealing with.
"""

from agent_state import window_open_for_goal, _normalize_goal, get_open_window_titles

RISKY_GOAL_KEYWORDS = {
    "delete", "remove", "uninstall", "format", "wipe", "erase",
    "rename", "terminate", "kill", "end task", "shutdown", "restart",
    "disable", "reset", "clear all", "empty",
}

def _is_risky(goal: str) -> bool:
    return any(kw in goal.lower() for kw in RISKY_GOAL_KEYWORDS)

def _is_close_goal(goal: str) -> bool:
    g = goal.lower().strip()
    return g == "close" or g.startswith("close ")

def is_launch_goal(goal: str, was_on_desktop: bool) -> bool:
    """
    Return True if this goal requires opening a new window (app launch).
    Return False if this is an in-app action that is satisfied by clicking.

    Primary signal: was the user on the desktop when they made this choice?
    - On desktop → they are launching an app → launch goal
    - Inside an app → they are performing an action → action goal
    """
    return was_on_desktop

def goal_satisfied(goal: str, state: dict, was_on_desktop: bool = True) -> bool:
    """
    Check whether a goal has been satisfied.

    was_on_desktop: True if the BCI scan that generated this goal happened
                    while the user was on the desktop (no window open).
                    False if they were inside an app.

    - Risky goals: never auto-satisfied
    - In-app action goals (was_on_desktop=False): always satisfied after click
    - "close X": satisfied when X window is gone
    - Launch goals (was_on_desktop=True): satisfied when Win32 sees the window
    """
    if _is_risky(goal):
        print(f"[GoalEval] Risky goal '{goal}' — will not auto-satisfy")
        return False

    # In-app action — clicking was the entire goal, always satisfied
    if not was_on_desktop:
        print(f"[GoalEval] In-app action '{goal}' — satisfied after click")
        return True

    # Bare "close" — satisfied when back on desktop
    if goal.lower().strip() == "close":
        return not state.get("window_open", False)

    # "close X" — satisfied when X window is gone
    if _is_close_goal(goal):
        app_name = _normalize_goal(goal)
        return not window_open_for_goal(app_name)

    # Launch goal — satisfied when Win32 sees the app window
    satisfied = window_open_for_goal(goal)
    if not satisfied:
        print(f"[GoalEval] '{goal}' not yet satisfied — window not detected")
    return satisfied