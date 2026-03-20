"""
goal_verifier.py

Vision-based goal verification using Gemma3:4b multimodal via Ollama.

After an in-app action is executed (e.g. "Play Music", "Open Downloads"),
this module takes a screenshot, encodes it as base64, and asks Gemma3:4b
to visually reason about whether the goal was achieved.

This is more reliable than OmniParser-based verification because the LLM
sees the actual screen state rather than OmniParser's interpretation of it.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from io import BytesIO

import requests
from PIL import Image


@dataclass
class VerificationResult:
    achieved: bool
    confidence: str   # "high", "medium", "low"
    reason: str       # LLM explanation in plain English
    raw_response: str # full LLM output for debugging


class GoalVerifier:
    """
    Verifies whether a BCI-selected in-app goal was achieved by visually
    inspecting the screen after the action using Gemma3:4b vision.

    Only used for in-app actions (was_on_desktop=False).
    Launch goals are already verified by Win32 window title checking.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        llm_url: str = "http://localhost:11434/api/chat",
        timeout: int = 30,
        max_dim: int = 1024,
    ):
        self.model = model
        self.llm_url = llm_url
        self.timeout = timeout
        self.max_dim = max_dim

    def verify(
        self,
        goal: str,
        screenshot: Image.Image,
        context: str = "",
    ) -> VerificationResult:
        """
        Verify whether the goal was achieved based on the current screenshot.

        Args:
            goal:       The action just executed e.g. "Play Music"
            screenshot: PIL Image of screen state AFTER the click
            context:    Optional extra context e.g. "User was inside Spotify"

        Returns:
            VerificationResult with achieved flag, confidence, and reason.
        """
        t0 = time.perf_counter()
        print(f"[Verify] Checking goal: '{goal}'")

        try:
            img_b64 = self._encode_screenshot(screenshot)
            prompt = self._build_prompt(goal, context)
            raw = self._call_llm(prompt, img_b64)
            result = self._parse_response(raw)
            elapsed = time.perf_counter() - t0
            status = "✓ ACHIEVED" if result.achieved else "✗ NOT ACHIEVED"
            print(f"[Verify] {status} — {result.reason} ({elapsed:.1f}s)")
            return result

        except Exception as exc:
            print(f"[Verify] Error: {exc} — assuming achieved to avoid blocking")
            return VerificationResult(
                achieved=True,
                confidence="low",
                reason=f"Verification error ({exc}), assuming achieved",
                raw_response="",
            )

    def _encode_screenshot(self, screenshot: Image.Image) -> str:
        """Resize and base64-encode screenshot as JPEG."""
        img = screenshot.copy()
        w, h = img.size
        if max(w, h) > self.max_dim:
            scale = self.max_dim / max(w, h)
            img = img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS,
            )
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_prompt(self, goal: str, context: str) -> str:
        ctx_line = f"\nContext: {context}" if context else ""
        return f"""You are a desktop automation verifier. An AI agent just executed the action: "{goal}".{ctx_line}

Look at the screenshot carefully and determine whether the action was successfully completed.

Examples of what to look for:
- "Play Music" → a pause button is visible, a progress bar or song name is shown
- "Open Downloads" → a file explorer window shows Downloads folder contents
- "Open Spotify" → the Spotify application window is visible
- "Go back" → the previous screen or folder is now shown
- "Launch Chrome" → Chrome browser window is open
- "Open Pictures" → Pictures folder content is visible in File Explorer
- "Launch Videos" → Videos folder is open and showing its contents
- "Open Control Panel" → Control Panel window with categories is visible

Respond ONLY with valid JSON in exactly this format:
{{
  "achieved": true or false,
  "confidence": "high" or "medium" or "low",
  "reason": "one sentence explaining what you see that confirms or denies the goal"
}}

Do not include any text outside the JSON."""

    def _call_llm(self, prompt: str, img_b64: str) -> str:
        """Call Gemma3:4b via Ollama /api/chat with vision."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        response = requests.post(self.llm_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    def _parse_response(self, raw: str) -> VerificationResult:
        """Parse LLM JSON response into VerificationResult."""
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()
            obj = json.loads(clean)
            return VerificationResult(
                achieved=bool(obj.get("achieved", False)),
                confidence=str(obj.get("confidence", "medium")).lower(),
                reason=str(obj.get("reason", "No reason provided")),
                raw_response=raw,
            )
        except Exception as exc:
            print(f"[Verify] Parse error: {exc} | raw: {raw[:100]}")
            return VerificationResult(
                achieved=False,
                confidence="low",
                reason="Could not parse verification response",
                raw_response=raw,
            )