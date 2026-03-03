import json
import requests
from typing import List
from models import DetectedElement


class AgentPlanner:

    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def next_action(self, goal: str, state: dict, elements: List[DetectedElement]) -> str | None:

        element_names = [e.name for e in elements]

        prompt = f"""
You are a desktop agent.

Goal: {goal}

Current State:
{state}

Available Clickable Elements:
{element_names}

Return ONLY JSON like:
{{"click": "Element Name"}}

If no action is needed, return:
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
                timeout=20,
            )

            if response.status_code == 200:
                raw = response.json().get("response", "")
                obj = json.loads(raw)
                return obj.get("click")

        except Exception:
            pass

        return None
