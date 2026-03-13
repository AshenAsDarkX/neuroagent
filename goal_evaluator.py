"""
Goal evaluator — checks whether a goal has been satisfied given the current
screen state produced by agent_state.infer_state().

Matching is keyword-based and case-insensitive so it handles the LLM-generated
goal labels like "open Google Chrome" or "open Spotify" naturally.
"""

_GOAL_KEYWORD_MAP = {
    # state key          : trigger keywords in the goal string
    "spotify_visible"    : ["spotify"],
    "chrome_visible"     : ["chrome", "google chrome", "browser"],
    "explorer_visible"   : ["explorer", "file explorer", "files"],
    "search_visible"     : ["search"],
    "start_visible"      : ["start", "windows", "start menu"],
    "close_visible"      : ["close", "close window", "quit", "exit"],
}

# Goals satisfied when the state value is FALSE (i.e. something was closed)
_INVERT_KEYS = {"close_visible"}


def goal_satisfied(goal: str, state: dict) -> bool:
    goal_lower = goal.lower()

    for state_key, keywords in _GOAL_KEYWORD_MAP.items():
        if any(kw in goal_lower for kw in keywords):
            value = state.get(state_key, False)
            return (not value) if state_key in _INVERT_KEYS else bool(value)

    # Dynamic fallback: check if any detected element name is in state
    # agent_state.infer_state() adds "{slug}_visible: True" for every element
    for state_key, value in state.items():
        if not state_key.endswith("_visible"):
            continue
        slug = state_key[: -len("_visible")]          # e.g. "google_chrome"
        slug_words = set(slug.replace("_", " ").split())
        goal_words = set(goal_lower.replace("_", " ").split())
        # If 2+ words overlap it's likely the same app
        if len(slug_words & goal_words) >= 2:
            return bool(value)

    print(f"[GoalEval] Warning: no rule matched goal '{goal}' — returning False")
    return False
