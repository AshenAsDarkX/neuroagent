from __future__ import annotations

import json
import re
from typing import List

from app_config import AppConfig
from debug_writer import DebugArtifactWriter
from models import DetectedElement


class ElementRanker:
    def __init__(self, config: AppConfig, debug_writer: DebugArtifactWriter) -> None:
        self.config = config
        self.debug_writer = debug_writer

    @staticmethod
    def is_garbage_label(name: str) -> bool:
        trimmed = name.strip()
        if trimmed.endswith(".") and len(trimmed) > 20:
            return True
        if trimmed.lower().startswith(("a ", "an ", "the ")):
            return True
        compact = trimmed.replace(",", "").replace("-", "").replace(" ", "").replace(".", "")
        if compact.isdigit():
            return True
        if trimmed.lower() in {"increase", "decrease", "toggle", "sending a message or message."}:
            return True
        if len(trimmed) < 2:
            return True
        return False

    @staticmethod
    def make_friendly_label(name: str) -> str:
        normalized = re.sub(r"\s+", " ", name).strip().rstrip(".")

        for suffix in (" browser", " application", " app"):
            if normalized.lower().endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()

        app_names = {
            "spotify", "chrome", "google chrome", "edge", "microsoft edge",
            "microsoft 365", "explorer", "file explorer", "settings",
            "youtube", "discord", "slack", "vscode", "terminal",
            "ink pro", "toggle terminal", "notepad",
        }
        if normalized.lower() in app_names:
            overrides = {
                "google chrome": "Chrome",
                "microsoft edge": "Edge",
                "microsoft edge browser": "Edge",
                "microsoft 365": "M365",
                "toggle terminal": "Terminal",
                "file explorer": "Explorer",
            }
            short_name = overrides.get(normalized.lower(), normalized.title())
            return f"Open {short_name}"

        if normalized.lower() in ("search", "search bar"):
            return "Click Search"

        if normalized.lower() == "windows":
            return "Open Start"

        if len(normalized.split()) <= 2:
            return f"Click {normalized.title()}"

        return normalized[:28]

    def rank(self, elements: List[DetectedElement]) -> List[DetectedElement]:
        if not elements:
            return elements

        survivors = self._prefilter(elements)
        print(f"[LLM] Pre-filter: {len(elements)} -> {len(survivors)} survivors")
        if not survivors:
            return []

        prompt = self._build_prompt(survivors)
        raw_response = self._query_llm(prompt)
        chosen_indices = self._parse_chosen_indices(raw_response)

        result = self._map_ranked_elements(elements, survivors, chosen_indices)
        if not result:
            print("[LLM] Fallback: using pre-filtered survivors directly")
            for _, element in survivors[:14]:
                copied = DetectedElement(**element.__dict__)
                copied.name = self.make_friendly_label(copied.name)
                result.append(copied)

        self.debug_writer.save_llm_artifacts(
            survivors_count=len(survivors),
            raw_response=raw_response,
            chosen_indices=chosen_indices,
            final_actions=[item.name for item in result],
        )

        print(f"[LLM] Final {len(result)} actions: {[item.name for item in result]}")
        return result

    def _prefilter(self, elements: List[DetectedElement]) -> List[tuple[int, DetectedElement]]:
        survivors: List[tuple[int, DetectedElement]] = []
        seen_names = set()
        for index, element in enumerate(elements):
            name = (element.name or "").strip()
            if not element.interactive:
                continue
            if self.is_garbage_label(name):
                continue

            key = name.lower()
            if key in seen_names:
                continue

            seen_names.add(key)
            survivors.append((index, element))

        return survivors

    def _build_prompt(self, survivors: List[tuple[int, DetectedElement]]) -> str:
        candidates = [
            {"idx": index, "name": element.name.strip()[:50]}
            for index, element in survivors
        ]
        candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
        return (

            "You are an intelligent agent ranking desktop UI actions for a Brain-Computer Interface (BCI) system.\n"
            "The user selects actions using brain signals, so the MOST LIKELY user intentions must appear first.\n\n"

            "OBJECTIVE:\n"
            "Rank the detected UI elements by how likely a normal computer user would interact with them.\n\n"

            "IMPORTANT:\n"
            "The user can only select from a small number of options.\n"
            "Therefore show the MOST USEFUL and COMMON actions first.\n\n"

            "RANKING CRITERIA:\n"

            "HIGHEST PRIORITY:\n"
            "- Search bars\n"
            "- Frequently used applications\n"
            "- Open windows\n"
            "- File Explorer folders\n"
            "- Browser icons\n\n"

            "HIGH PRIORITY:\n"
            "- Open\n"
            "- New\n"
            "- Copy\n"
            "- Paste\n"
            "- Delete\n"
            "- Rename\n"
            "- Save\n"
            "- Download\n\n"

            "MEDIUM PRIORITY:\n"
            "- Back\n"
            "- Forward\n"
            "- Refresh\n"
            "- Sort\n"
            "- View\n"
            "- Navigation buttons\n\n"

            "LOW PRIORITY:\n"
            "- Settings\n"
            "- Rare tools\n"
            "- Advanced menus\n"
            "- Developer tools\n\n"

            "REMOVE:\n"
            "- Descriptions\n"
            "- Sentences\n"
            "- OCR garbage\n"
            "- SVG text\n"
            "- Non-clickable items\n"
            "- Duplicates\n\n"

            "LABELING RULES:\n"
            "- Convert names into short action labels\n"
            "- Examples:\n"
            "  Spotify -> Open Spotify\n"
            "  Downloads -> Open Downloads\n"
            "  Search -> Search\n"
            "  Chrome -> Open Chrome\n\n"

            "OUTPUT FORMAT:\n"
            "Return ONLY a numbered list.\n"
            "Example:\n"
            "1. Search\n"
            "2. Open Chrome\n"
            "3. Open Downloads\n"
            "4. Copy\n"
            "5. Paste\n\n"

            "CANDIDATES:\n"
            f"Candidates:\n{candidates_json}\n"
        )

    def _query_llm(self, prompt: str) -> str:
        print(f"[LLM] Sending prompt to {self.config.llm_model}...")
        try:
            import requests

            response = requests.post(
                self.config.llm_url,
                json={
                    "model": self.config.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1},
                },
                timeout=25,
            )
            if response.status_code == 200:
                raw_text = response.json().get("response", "")
                print(f"[LLM] Raw response: {raw_text[:300]}")
                return raw_text
            print(f"[LLM] Request failed: HTTP {response.status_code}")
        except Exception as exc:
            print(f"[LLM] Ollama call failed: {exc}")
        return ""

    @staticmethod
    def _parse_chosen_indices(raw_response: str) -> List[int]:
        if not raw_response:
            return []

        parsed = None
        try:
            parsed = json.loads(raw_response)
        except Exception:
            match = re.search(r"\{.*\}", raw_response, flags=re.S)
            if match:
                try:
                    parsed = json.loads(match.group(0).replace("'", '"'))
                except Exception:
                    parsed = None

        if not isinstance(parsed, dict):
            return []

        chosen = parsed.get("chosen", [])
        indices: List[int] = []
        for item in chosen:
            try:
                indices.append(int(item))
            except Exception:
                continue

        print(f"[LLM] Chosen indices: {indices}")
        return indices

    def _map_ranked_elements(
        self,
        elements: List[DetectedElement],
        survivors: List[tuple[int, DetectedElement]],
        chosen_indices: List[int],
    ) -> List[DetectedElement]:
        if not chosen_indices:
            return []

        valid_indices = {index for index, _ in survivors}
        result: List[DetectedElement] = []

        for idx in chosen_indices:
            if idx not in valid_indices:
                print(f"[LLM] Skipping invalid idx {idx}")
                continue

            copied = DetectedElement(**elements[idx].__dict__)
            copied.name = self.make_friendly_label(copied.name)
            result.append(copied)

            if len(result) >= 14:
                break

        return result
