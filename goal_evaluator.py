def goal_satisfied(goal: str, state: dict) -> bool:
    if goal == "Open Spotify":
        return state["spotify_visible"]

    if goal == "Open Chrome":
        return state["chrome_visible"]

    if goal == "Open File Explorer":
        return state["explorer_visible"]

    if goal == "Search":
        return state["search_visible"]

    if goal == "Close Window":
        return not state["close_visible"]

    return False