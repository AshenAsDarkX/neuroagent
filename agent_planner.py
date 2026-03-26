import json
import requests
from typing import List, Optional
from models import DetectedElement


def _get_llm_model() -> str:
    """Read the configured LLM model from AppConfig so planner stays in sync."""
    try:
        from app_config import AppConfig
        return AppConfig.load().llm_model
    except Exception:
        return "gemma3:4b"


class AgentPlanner:

    def __init__(self, model: str = ""):
        self.model = model or _get_llm_model()

    def next_action(
        self,
        goal: str,
        state: dict,
        elements: List[DetectedElement],
        failed_actions: Optional[List[str]] = None,
    ) -> str | None:

        element_names = [e.name for e in elements]
        failed_section = ""
        if failed_actions:
            failed_section = f"""
Already tried (do NOT repeat these):
{failed_actions}
"""

        prompt = f"""
You are a desktop automation agent helping a user achieve a goal using only mouse clicks.

Goal: {goal}

Current screen state:
{state}

Available clickable elements:
{element_names}
{failed_section}
Return ONLY JSON like:
{{"click": "Element Name"}}

Choose the element most likely to progress toward the goal.
If the goal is already achieved or no action is possible, return:
{{"click": null}}
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1},
                },
                timeout=120,
            )

            if response.status_code == 200:
                raw = response.json().get("response", "")
                obj = json.loads(raw)
                return obj.get("click")

        except Exception:
            pass

        return None