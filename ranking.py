from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from app_config import AppConfig
from debug_writer import DebugArtifactWriter
from models import DetectedElement


class ElementRanker:
    def __init__(self, config: AppConfig, debug_writer: DebugArtifactWriter) -> None:
        self.config = config
        self.debug_writer = debug_writer
        self.bci_filter_inputs = bool(getattr(config, "bci_filter_inputs", True))

    @staticmethod
    def is_garbage_label(name: str) -> bool:
        trimmed = name.strip()
        low = trimmed.lower()

        if len(trimmed) < 2:
            return True
        # Long sentences are not clickable labels
        if trimmed.endswith(".") and len(trimmed) > 20:
            return True
        if low.startswith(("a ", "an ", "the ")):
            return True
        # Pure numbers or ratio strings like "470/0", "1.5x"
        compact = trimmed.replace(",", "").replace("-", "").replace(" ", "").replace(".", "")
        if compact.isdigit():
            return True
        if re.match(r"^\d+[/x:]\d*$", trimmed):
            return True
        # Single words that are definitely not UI element names
        if low in {
            "increase", "decrease", "toggle", "sending a message or message.",
            "this", "that", "here", "there", "yes", "no", "ok",
            "eng us", "eng", "save", "subtitles", "pencil", "toggie",
            "m0,0l9,0 4.5,5z", "a video game or video game.", "a&t video game",
        }:
            return True
        # Date strings like "3, September, 2024"
        if re.search(r"\d{4}", trimmed) and any(
            m in low for m in (
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
            )
        ):
            return True
        return False

    @staticmethod
    def make_friendly_label(name: str) -> str:
        # Clean OCR artefacts
        normalized = re.sub(r"\s+", " ", name).strip().rstrip(".").rstrip(":")

        for suffix in (" browser", " application", " app"):
            if normalized.lower().endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()

        low = normalized.lower()

        app_names = {
            "spotify", "chrome", "google chrome", "edge", "microsoft edge",
            "microsoft 365", "explorer", "file explorer", "settings",
            "youtube", "discord", "slack", "vscode", "terminal",
            "ink pro", "toggle terminal", "notepad", "lightroom",
        }
        if low in app_names:
            overrides = {
                "google chrome": "Chrome",
                "microsoft edge": "Edge",
                "microsoft edge browser": "Edge",
                "microsoft 365": "M365",
                "toggle terminal": "Terminal",
                "file explorer": "Explorer",
            }
            short_name = overrides.get(low, normalized.title())
            return f"Launch {short_name}"

        if low in ("search", "search bar"):
            return "Click Search"

        if low == "windows":
            return "Open Start Menu"

        if low == "save":
            return "Save file"

        # Pure action buttons — no prefix
        action_buttons = {
            "extract", "test", "wizard", "info", "convert", "compress",
            "encrypt", "split", "combine", "repair", "benchmark",
            "cut", "copy", "paste", "rename", "delete", "refresh",
            "sort", "group", "filter", "select all", "properties",
        }
        if low in action_buttons:
            return normalized.title()

        # File names
        if "." in normalized and len(normalized.split()) <= 3:
            return f"Open {normalized}"[:40]

        if len(normalized.split()) <= 2:
            return f"Open {normalized.title()}"

        return normalized[:28]

    def rank(self, elements: List[DetectedElement]) -> List[DetectedElement]:
        if not elements:
            return elements

        survivors = self._prefilter(elements)

        # Re-rank survivors so the most useful items reach the LLM first.
        # Priority tiers:
        #   0 — main content area folders/files (the items the user is browsing)
        #   1 — sidebar navigation folders (Downloads, Pictures, tutorial, etc.)
        #   2 — main content area non-folder items (toolbar-like)
        #   3 — toolbar buttons (top strip) and taskbar
        def _content_priority(item: tuple) -> int:
            _, el = item
            x1, y1, x2, y2 = el.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            img_w, img_h = 1920, 1080
            cx_norm = cx / img_w
            cy_norm = cy / img_h

            in_toolbar   = cy_norm < 0.15                          # top strip
            in_taskbar   = cy_norm > 0.93                          # bottom strip
            in_sidebar   = cx_norm < 0.15 and not in_toolbar       # left nav panel
            in_content   = not in_toolbar and not in_taskbar and not in_sidebar

            is_folder_or_file = el.element_type.lower() in ("folder", "file", "icon")

            if in_content and is_folder_or_file:
                return 0   # main content folders/drives — always first
            if in_sidebar and is_folder_or_file:
                return 1   # sidebar nav folders — very useful
            if in_content:
                return 2   # other content-area elements
            return 3       # toolbar buttons and taskbar — last

        survivors.sort(key=_content_priority)
        survivors = survivors[:20]
        print(f"[LLM] Pre-filter: {len(elements)} -> {len(survivors)} survivors")
        if not survivors:
            return []

        ui_state = self._build_ui_state(survivors)
        goal_prompt = self._build_goal_prompt(ui_state)
        raw_response = self._query_llm(goal_prompt, force_json=True)

        valid_ids = {index for index, _ in survivors}
        actions = self._parse_actions(raw_response, valid_ids)
        result = self._map_goal_elements(elements, actions)

        chosen_indices: List[int] = [target for _, target in actions]

        # Minimum viable result: if LLM returned fewer than 3 actions but there
        # are at least 3 survivors, the LLM under-delivered — skip legacy ranking
        # and go straight to the direct survivor fallback so the user always gets
        # a full set of options.
        MIN_ACTIONS = 4
        llm_under_delivered = len(result) < MIN_ACTIONS and len(survivors) >= MIN_ACTIONS

        if not result:
            print("[LLM] Goal generation failed or invalid JSON. Falling back to legacy ranking.")
            ranking_prompt = self._build_prompt(survivors)
            fallback_raw = self._query_llm(ranking_prompt, force_json=False)
            chosen_indices = self._parse_chosen_indices(fallback_raw)
            result = self._map_ranked_elements(elements, survivors, chosen_indices)
            if result:
                raw_response = fallback_raw

        if not result or llm_under_delivered:
            if llm_under_delivered and result:
                print(f"[LLM] Only {len(result)} action(s) from LLM — too few, using survivors directly.")
            else:
                print("[LLM] Fallback: using pre-filtered survivors directly")
            # Build labels directly from survivor element names — no LLM needed
            direct: List[DetectedElement] = []
            seen = {e.name.lower() for e in result}  # keep any good LLM results
            for _, element in survivors:
                label = self.make_friendly_label(element.name)
                if label.lower() not in seen:
                    copied = DetectedElement(**element.__dict__)
                    copied.name = label
                    direct.append(copied)
                    seen.add(label.lower())
            # Merge: LLM results first (they have better labels), direct fills the rest
            result = result + direct
            result = result[:14]

        self.debug_writer.save_llm_artifacts(
            survivors_count=len(survivors),
            raw_response=raw_response,
            chosen_indices=chosen_indices,
            final_actions=[item.name for item in result],
        )

        # Push risky/destructive actions to the end so BCI top 5 are never dangerous
        result = self._push_risky_actions_last(result)

        # Detect if scrolling is available and inject scroll options
        scroll_elements = self._make_scroll_elements(elements)
        if scroll_elements:
            result = result + scroll_elements
            print(f"[Scroll] Added scroll options: {[e.name for e in scroll_elements]}")

        print(f"[LLM] Final {len(result)} actions: {[item.name for item in result]}")
        return result

    # ---------------------------------------------------------------------------
    # Risky action detection and reranking
    # ---------------------------------------------------------------------------

    # Actions that are potentially destructive or irreversible.
    # These are pushed to the END of the options list so they never appear
    # in the top 5 BCI slots unless there is genuinely nothing else to show.
    # They are NOT removed — the user may legitimately want to close a window.
    _RISKY_KEYWORDS: frozenset = frozenset({
        "close", "delete", "remove", "uninstall", "format", "wipe", "erase",
        "rename", "terminate", "kill", "end task", "shutdown", "restart",
        "disable", "reset", "clear all", "empty trash", "sign out", "log out",
    })

    def _is_risky_action(self, name: str) -> bool:
        """Return True if this action name contains a risky/destructive keyword."""
        lowered = name.lower()
        return any(kw in lowered for kw in self._RISKY_KEYWORDS)

    def _push_risky_actions_last(
        self, elements: List[DetectedElement]
    ) -> List[DetectedElement]:
        """
        Reorder so safe actions come first, risky actions come last.
        BCI top options (slots 1-5) should never be destructive.
        Risky actions still appear — they are just pushed to later pages.
        """
        safe   = [e for e in elements if not self._is_risky_action(e.name)]
        risky  = [e for e in elements if     self._is_risky_action(e.name)]
        if risky:
            risky_names = [e.name for e in risky]
            print(f"[Ranker] Pushed risky actions to end: {risky_names}")
        return safe + risky

    # Scrollbar-related element type names OmniParser may assign
    _SCROLLBAR_TYPES: frozenset = frozenset({
        "scrollbar", "scroll", "scroll_bar", "scrollbar_vertical",
        "scrollbar_horizontal", "vertical_scroll", "slider",
    })
    # Keywords in element names that indicate a scrollbar
    _SCROLLBAR_NAME_KEYWORDS: frozenset = frozenset({
        "scrollbar", "scroll bar", "vertical scroll", "horizontal scroll",
        "scroll thumb", "scroll track",
    })
    # Minimum number of elements in a window before we assume scrolling may help
    _SCROLL_ELEMENT_THRESHOLD: int = 12

    def _make_scroll_elements(
        self, elements: List[DetectedElement]
    ) -> List[DetectedElement]:
        """
        Detect whether the current screen has scrollable content and return
        synthetic Scroll Down / Scroll Up DetectedElement objects if so.

        Detection strategy (two-signal approach):
          1. Primary  — OmniParser detected a scrollbar element
          2. Fallback — window has many elements (long content likely off-screen)

        Scroll elements are given a special sentinel bbox/center of (0,0,0,0)
        so controller.execute_click can identify them and call
        pyautogui.scroll() instead of doubleClick().
        """
        if not elements:
            return []

        # Check if we are inside a window (not on desktop)
        img_h = 1080
        taskbar_threshold = 0.93
        window_elements = [
            e for e in elements
            if ((e.bbox[1] + e.bbox[3]) / 2) / img_h < taskbar_threshold
        ]
        if len(window_elements) < 5:
            # On desktop — scrolling not applicable
            return []

        # Signal 1: OmniParser detected a scrollbar
        has_scrollbar = any(
            e.element_type.lower() in self._SCROLLBAR_TYPES or
            any(kw in e.name.lower() for kw in self._SCROLLBAR_NAME_KEYWORDS)
            for e in elements
        )

        # Signal 2: many elements in window — likely more content below
        has_many_elements = len(window_elements) >= self._SCROLL_ELEMENT_THRESHOLD

        if not has_scrollbar and not has_many_elements:
            return []

        reason = "scrollbar detected" if has_scrollbar else f"{len(window_elements)} elements visible"
        print(f"[Scroll] Scrolling available ({reason})")

        # Sentinel values — controller identifies these by name
        sentinel_bbox = (0, 0, 0, 0)
        sentinel_center = (0, 0)

        scroll_down = DetectedElement(
            name="Scroll Down",
            bbox=sentinel_bbox,
            center=sentinel_center,
            interactive=True,
            element_type="scroll_action",
        )
        scroll_up = DetectedElement(
            name="Scroll Up",
            bbox=sentinel_bbox,
            center=sentinel_center,
            interactive=True,
            element_type="scroll_action",
        )
        # Down first — much more common use case
        return [scroll_down, scroll_up]

    # ---------------------------------------------------------------------------
    # Keywords that indicate an element requires keyboard input — unusable in BCI
    # ---------------------------------------------------------------------------

    # Keywords that indicate an element requires keyboard input — unusable in BCI
    _INPUT_KEYWORDS: frozenset[str] = frozenset({
        "search", "type here", "enter", "address bar", "url", "find",
        "input", "text box", "text field", "query", "keyword",
    })

    def _is_input_element(self, element: DetectedElement, name: str) -> bool:
        """Return True if this element requires keyboard input to use."""
        lowered = name.lower()

        # Check element type set by OmniParser
        if element.element_type.lower() in ("input", "textbox", "text_field"):
            return True

        # Check name against known input keywords
        if any(kw in lowered for kw in self._INPUT_KEYWORDS):
            return True

        return False

    def _prefilter(self, elements: List[DetectedElement]) -> List[tuple[int, DetectedElement]]:
        survivors: List[tuple[int, DetectedElement]] = []
        filtered_inputs = 0

        for index, element in enumerate(elements):
            # remove non-interactive items
            if not element.interactive:
                continue

            name = (element.name or "").strip()

            if not name:
                continue

            # remove numeric labels
            if name.replace(".", "").isdigit():
                continue

            # remove long sentences
            if len(name.split()) > 4:
                continue

            # BCI filter — remove elements that require typing
            if self.bci_filter_inputs and self._is_input_element(element, name):
                filtered_inputs += 1
                print(f"[BCI Filter] Removed input element: '{name}'")
                continue

            survivors.append((index, element))

        if filtered_inputs:
            print(f"[BCI Filter] Removed {filtered_inputs} input-requiring element(s).")

        return survivors

    def _infer_active_context(self, survivors):
        """
        Dynamically detect what app is in the foreground by examining
        which non-taskbar elements dominate the screen.

        Works for ANY app (Spotify, PyCharm, WhatsApp, VLC, etc.)
        because it reads OmniParser-detected element names rather than
        matching against a hardcoded app list.

        Returns:
            active_app (str): name of the active app, or "" for desktop.
            is_desktop (bool): True when no app window is open.
        """
        TASKBAR_Y_NORM = 0.93
        IMG_H = 1080

        window_elements = [
            e for _, e in survivors
            if ((e.bbox[1] + e.bbox[3]) / 2) / IMG_H < TASKBAR_Y_NORM
        ]
        taskbar_elements = [
            e for _, e in survivors
            if ((e.bbox[1] + e.bbox[3]) / 2) / IMG_H >= TASKBAR_Y_NORM
        ]

        if not window_elements:
            return "", True

        window_names = [e.name.strip() for e in window_elements if e.name.strip()]
        taskbar_names = [e.name.strip().lower() for e in taskbar_elements]

        # Match taskbar app names against window content — works for any app
        for t_name in taskbar_names:
            if not t_name or len(t_name) < 3:
                continue
            for w_name in window_names:
                if t_name in w_name.lower() or w_name.lower() in t_name:
                    return t_name.title(), False

        # Fallback: most frequent first-word in window element names
        name_freq: dict = {}
        for name in window_names:
            key = name.lower().split()[0] if name else ""
            if len(key) > 2:
                name_freq[key] = name_freq.get(key, 0) + 1

        if name_freq:
            best = max(name_freq, key=lambda k: name_freq[k])
            return best.title(), False

        return "unknown app", False

    def _build_ui_state(self, survivors: List[tuple[int, DetectedElement]]) -> Dict[str, Any]:
        ui_elements: List[Dict[str, Any]] = []

        for index, element in survivors:
            clean_name = re.sub(r"\s+", " ", (element.name or "").strip())
            if not clean_name:
                clean_name = f"Element {index}"

            ui_elements.append(
                {
                    "id": index,
                    "name": clean_name[:80],
                    "interactive": bool(element.interactive),
                    "type": self._infer_element_type(element, clean_name),
                }
            )

        active_app, is_desktop = self._infer_active_context(survivors)

        return {
            "environment": "desktop" if is_desktop else "application",
            "active_app": active_app,
            "is_desktop": is_desktop,
            "elements": ui_elements,
        }

    @staticmethod
    def _infer_element_type(element: DetectedElement, clean_name: str) -> str:
        raw_type = (element.element_type or "").strip().lower()
        if raw_type:
            return raw_type

        lowered = clean_name.lower()
        if any(token in lowered for token in ("search", "input", "type here")):
            return "input"
        if any(token in lowered for token in ("download", "folder", "documents", "explorer")):
            return "folder"
        if any(token in lowered for token in ("button", "ok", "cancel", "submit")):
            return "button"
        return "icon"

    def _build_goal_prompt(self, ui_state: Dict[str, Any]) -> str:
        elements = ui_state.get("elements", [])

        # Deduplicate by name (case-insensitive) — keep first occurrence.
        # This prevents three "Google Chrome" entries all competing for slots.
        seen_names: set = set()
        deduped_elements = []
        for el in elements:
            key = el.get("name", "").strip().lower()
            if key and key not in seen_names:
                seen_names.add(key)
                deduped_elements.append(el)

        # How many actions to request — fill the BCI display (6 tiles max)
        n_request = min(6, len(deduped_elements))

        deduped_state = dict(ui_state)
        deduped_state["elements"] = deduped_elements
        ui_state_json = json.dumps(deduped_state, ensure_ascii=False, indent=2)

        is_desktop = ui_state.get("is_desktop", True)
        active_app = ui_state.get("active_app", "")

        if is_desktop or not active_app:
            context_line = "The user is on the Windows desktop. Suggest app launching actions."
        else:
            context_line = (
                f"The user is currently INSIDE {active_app}. "
                f"Suggest actions relevant to using {active_app} (e.g. its menus, buttons, and controls). "
                f"Do NOT suggest opening {active_app} again — it is already open."
            )

        # Build few-shot example from ACTUAL elements on screen.
        # Never hardcode labels — Gemma/Qwen echo examples literally,
        # so if the example says "Open Control Panel" it returns that
        # label on every screen including Spotify.
        def _make_label(name: str, el_type: str) -> str:
            n = name.strip().rstrip(":").rstrip(".")  # clean OCR artefacts like trailing : or .
            n = re.sub(r"\s+", " ", n)
            low = n.lower()

            # Already a full verb+noun phrase — keep it
            good_prefixes = ("go back", "go forward", "open ", "launch ",
                             "view ", "toggle ", "adjust ", "close ",
                             "switch ", "add ", "download ", "minimize",
                             "maximize", "bookmark", "next page", "back ",
                             "extract", "test ", "wizard", "info", "delete ",
                             "rename ", "copy ", "cut ", "paste ")
            if any(low.startswith(p) for p in good_prefixes):
                return n[:32]

            # Pure action buttons — these are their own label, no prefix needed
            # e.g. "Extract", "Test", "Wizard", "Info", "Add", "Convert"
            action_buttons = {
                "extract", "test", "wizard", "info", "convert", "compress",
                "encrypt", "split", "combine", "repair", "benchmark",
                "new folder", "properties", "details", "preview",
                "cut", "copy", "paste", "rename", "delete", "refresh",
                "sort", "group", "filter", "select all",
            }
            if low in action_buttons or any(low.startswith(a + " ") for a in action_buttons):
                return n.title()[:32]

            # Bare single verbs with no noun
            bare_verbs = {"open", "launch", "view", "toggle", "adjust",
                          "close", "switch", "back", "next", "new",
                          "share", "add"}
            if low in bare_verbs:
                ctx = f" in {active_app}" if active_app and not is_desktop else ""
                return f"{n.title()}{ctx}"[:32]

            # File names — keep as-is with "Open" prefix
            if "." in n and len(n.split()) <= 3:
                return f"Open {n}"[:40]

            apps = ("spotify", "chrome", "edge", "firefox", "discord",
                    "slack", "steam", "vlc", "notepad", "vscode", "teams",
                    "lightroom", "photoshop", "illustrator", "premiere")
            if any(a in low for a in apps):
                return f"Launch {n.title()}"
            if any(k in low for k in ("wifi", "sound", "volume", "bluetooth", "network")):
                return f"Toggle {n.title()}"
            if any(k in low for k in ("close", "minimize", "maximize")):
                return n.title()
            return f"Open {n.title()}"

        example_items = []
        for el in deduped_elements[:n_request]:
            label = _make_label(el.get("name", "Element"), el.get("type", "icon"))
            example_items.append(
                f'    {{"goal": "{label}", "target": {el.get("id", 0)}}}'
            )
        example_json = (
            "{\n  \"actions\": [\n"
            + ",\n".join(example_items)
            + "\n  ]\n}"
        )

        return (
            "You are a computer control agent for a Brain-Computer Interface (BCI) system.\n\n"
            "CRITICAL CONSTRAINT: The user CANNOT type. They can ONLY click.\n"
            "Do NOT suggest any action that requires keyboard input.\n\n"
            f"CURRENT CONTEXT: {context_line}\n\n"
            f"YOUR TASK: Generate EXACTLY {n_request} actions from the elements "
            f"listed in UI STATE JSON below. Every target id MUST appear in that list.\n\n"
            "LABEL RULES:\n"
            "- Derive the goal label FROM the element name — do NOT invent labels.\n"
            "- Start with a verb: Open, Launch, Go back, View, Toggle, Adjust, Close, Add, Download\n"
            "- For app icons: 'Launch [AppName]'\n"
            "- For navigation: 'Go back', 'Go forward'\n"
            "- For window controls: 'Minimize', 'Maximize', 'Close window'\n\n"
            "STRICT RULES:\n"
            "- Use ONLY element ids from the UI STATE JSON below.\n"
            "- Do NOT copy goal labels from the example — write labels based on real element names.\n"
            "- Each target id must be different.\n"
            "- Do NOT add search bars or text inputs.\n\n"
            f"Example format (ids and labels below are EXAMPLES only — "
            f"use the real ids and names from UI STATE JSON):\n"
            f"{example_json}\n\n"
            "Return ONLY the JSON object. No explanation, no markdown.\n\n"
            f"UI STATE JSON:\n{ui_state_json}\n"
        )

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

    def _query_llm(self, prompt: str, force_json: bool = True) -> str:
        print(f"[LLM] Sending prompt to {self.config.llm_model} (json={force_json})...")
        try:
            import requests

            payload: dict = {
                "model": self.config.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            }

            if force_json:
                payload["format"] = "json"

            response = requests.post(
                self.config.llm_url,
                json=payload,
                timeout=60,
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
    def _safe_parse_json_object(raw_response: str) -> Any:
        if not raw_response:
            return None

        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None

        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json.loads(candidate.replace("'", '"'))
            except Exception:
                return None

    # Goals that are just bare verbs with no noun — useless on BCI screen
    _BARE_VERB_GOALS: frozenset = frozenset({
        "open", "launch", "view", "toggle", "adjust", "switch", "next",
    })

    def _parse_actions(self, raw_response: str, valid_ids: set[int]) -> List[Tuple[str, int]]:
        parsed = self._safe_parse_json_object(raw_response)
        if not isinstance(parsed, dict):
            return []

        actions = parsed.get("actions")
        if not isinstance(actions, list):
            return []

        seen_targets = set()
        parsed_actions: List[Tuple[str, int]] = []

        for item in actions:
            if not isinstance(item, dict):
                continue

            goal = item.get("goal", "")
            if not isinstance(goal, str):
                continue
            goal = re.sub(r"\s+", " ", goal).strip()
            if not goal:
                continue

            # Normalise "Launch Close" / "Open Close" → "Close window"
            goal_low = goal.lower()
            if goal_low in ("launch close", "open close", "launch close window",
                            "minimize close", "close in help"):
                goal = "Close window"
            # Fix common bad labels from the LLM
            elif goal_low in ("open save", "launch save"):
                goal = "Save file"
            elif goal_low in ("open windows", "launch windows"):
                goal = "Open Start Menu"
            elif goal_low == "open windows start":
                goal = "Open Start Menu"
            # Drop bare single-verb goals — meaningless on BCI screen
            elif goal_low in self._BARE_VERB_GOALS:
                print(f"[LLM] Dropped bare verb goal: '{goal}'")
                continue
            # Drop single-word goals that have no verb (bare nouns like "Info")
            # unless they are known standalone actions
            elif len(goal.split()) == 1 and goal_low not in {
                "minimize", "maximize", "back", "next", "home",
                "extract", "wizard", "properties", "refresh",
            }:
                print(f"[LLM] Dropped bare noun goal: '{goal}'")
                continue
            # Drop ratio/numeric goals like "Open 470/0"
            elif re.search(r"\d+[/x:]\d*", goal):
                print(f"[LLM] Dropped numeric/ratio goal: '{goal}'")
                continue

            target = item.get("target")
            try:
                target_id = int(target)
            except Exception:
                continue

            if target_id not in valid_ids:
                continue
            if target_id in seen_targets:
                continue

            seen_targets.add(target_id)
            parsed_actions.append((goal, target_id))

            if len(parsed_actions) >= 14:
                break

        print(f"[LLM] Parsed actions: {parsed_actions}")
        return parsed_actions

    @staticmethod
    def _parse_chosen_indices(raw_response: str) -> List[int]:
        parsed = ElementRanker._safe_parse_json_object(raw_response)
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

    def _map_goal_elements(
        self,
        elements: List[DetectedElement],
        actions: List[Tuple[str, int]],
    ) -> List[DetectedElement]:
        if not actions:
            return []

        result: List[DetectedElement] = []
        for goal, target_id in actions:
            if target_id < 0 or target_id >= len(elements):
                continue

            copied = DetectedElement(**elements[target_id].__dict__)
            copied.name = goal
            result.append(copied)

        return result

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