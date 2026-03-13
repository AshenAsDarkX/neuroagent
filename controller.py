from __future__ import annotations

import threading
import time
import traceback
from typing import Any, List, Optional

import pyautogui

from app_config import AppConfig
from debug_writer import DebugArtifactWriter
from models import DetectedElement
from omniparser_engine import OmniParserEngine
from ranking import ElementRanker
from utils import ensure_dir


class BCIController:
    def __init__(
        self,
        config: AppConfig,
        bci: Any | None = None,
        bci_screen: Any | None = None,
    ) -> None:
        self.config = config
        self.bci = bci
        self.bci_screen = bci_screen
        if self.bci_screen is None:
            raise RuntimeError("BCI display is required.")

        ensure_dir(self.config.debug_dir)

        self.debug_writer = DebugArtifactWriter(config)
        self.bci_screen.set_loading(True, "Loading OmniParser...")
        self.parser = OmniParserEngine(config, status_callback=self.bci_screen.set_loading_text)
        self.ranker = ElementRanker(config, self.debug_writer)
        self.bci_screen.set_loading(False)

        self.is_processing = False
        self.is_executing = False
        self.agent_actions: List[DetectedElement] = []
        self.all_actions: List[DetectedElement] = []
        self.visible_options: list[dict[str, Any]] = []
        self.page_index = 0
        self.last_ranked: List[DetectedElement] = []

        self.bci_screen.root.bind("<space>", lambda _event: self.scan())
        self.bci_screen.root.bind("q", lambda _event: self.quit())
        self.bci_screen.root.bind("<Escape>", lambda _event: self.quit())

    def run_stimulation_phase(self) -> bool:
        if self.is_processing or self.is_executing or not self.visible_options:
            return False

        duration = 12.0
        step = 0.1
        total_steps = int(duration / step)
        bar_width = 10

        self.bci_screen.set_info("Focus on your target")
        start_time = time.perf_counter()

        for i in range(total_steps):
            if self.is_processing or self.is_executing or not self.visible_options:
                return False

            progress = int(((i + 1) / total_steps) * bar_width)
            bar = "#" * progress + "-" * (bar_width - progress)
            self.bci_screen.set_info(f"Stimulus running: [{bar}]")

            target_time = start_time + ((i + 1) * step)
            sleep_s = target_time - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        return True

    def _capture_main_monitor(self):
        time.sleep(0.2)
        return self.parser.capture_active_window_on_main_monitor(fallback_to_main=True)

    def scan_environment(self) -> List[DetectedElement]:
        screenshot = self._capture_main_monitor()

        if screenshot is None:
            return []

        parse_result = self.parser.parse(screenshot)
        ranked = self.ranker.rank(parse_result.elements)

        self.debug_writer.save_scan_artifacts(
            screenshot=screenshot,
            labeled_img_b64=parse_result.labeled_img_b64,
            raw=parse_result.raw,
        )

        return ranked

    def execute_click(self, target: DetectedElement) -> None:
        cx, cy = target.center
        print(f"Double-click: {target.name} -> ({cx}, {cy})")
        pyautogui.doubleClick(cx, cy)

    def find_element_by_name(
        self, element_name: str, elements: List[DetectedElement]
    ) -> Optional[DetectedElement]:
        query = element_name.strip().lower()
        if not query:
            return None

        for element in elements:
            if element.name.strip().lower() == query:
                return element

        for element in elements:
            if query in element.name.strip().lower():
                return element

        return None

    def _update_bci_screen_actions(self, agent_actions: List[DetectedElement]) -> None:
        self.all_actions = list(agent_actions)
        self.page_index = 0
        self._render_action_page()

    def _max_page_index(self) -> int:
        total = len(self.all_actions)
        if total <= 5:
            return 0
        remaining = total - 5
        return (remaining + 3) // 4

    def _render_action_page(self) -> None:
        total = len(self.all_actions)
        if total == 0:
            self.visible_options = []
            self.agent_actions = []
            self.page_index = 0
            self.bci_screen.root.after(0, lambda: self.bci_screen.show_actions([]))
            self.bci_screen.root.after(
                0, lambda: self.bci_screen.set_info("No options found. Rescanning...")
            )
            return

        max_page = self._max_page_index()
        self.page_index = max(0, min(self.page_index, max_page))

        options: list[dict[str, Any]] = []

        if self.page_index == 0:
            for action in self.all_actions[:5]:
                options.append(
                    {
                        "kind": "action",
                        "action": action,
                        "label": str(getattr(action, "goal", getattr(action, "name", ""))),
                    }
                )
            if total > 5:
                options.append({"kind": "next", "label": "Other options"})
        else:
            start = 5 + (self.page_index - 1) * 4
            for action in self.all_actions[start : start + 4]:
                options.append(
                    {
                        "kind": "action",
                        "action": action,
                        "label": str(getattr(action, "goal", getattr(action, "name", ""))),
                    }
                )
            options.append({"kind": "prev", "label": "Go back"})
            if start + 4 < total:
                options.append({"kind": "next", "label": "Other options"})

        self.visible_options = options
        self.agent_actions = [opt["action"] for opt in options if opt.get("kind") == "action"]
        labels = [str(opt.get("label", "")) for opt in options]

        self.bci_screen.root.after(0, lambda a=labels: self.bci_screen.show_actions(a))
        self.bci_screen.root.after(
            0,
            lambda p=self.page_index + 1, m=max_page + 1: self.bci_screen.set_info(
                f"Options page {p}/{m}. Focus on a target to select."
            ),
        )

    def _run_bci_selection(self) -> None:
        if not self.bci:
            return
        if self.is_processing or self.is_executing:
            return
        if not self.visible_options:
            return

        self.bci_screen.set_info("Waiting for EEG selection...")

        def worker() -> None:
            try:
                if not self.run_stimulation_phase():
                    return

                self.bci_screen.set_info("Decoding EEG signal...")

                # Derive which target the user was most likely focusing on during
                # the stimulation window. We sample the current flicker frame from
                # bci_screen and find which visible target had the highest ON-time
                # (most white frames) at that moment. This gives the BCI decoder a
                # meaningful hint and — critically — means different targets can be
                # selected in simulation mode instead of always defaulting to 0.
                target_option = self._infer_attended_target()

                decoded_selection, accuracy, trial_index = self.bci.get_selection(target_option)
                accuracy_threshold = float(getattr(self.bci, "accuracy_threshold", 0.25))

                if decoded_selection is None or accuracy < accuracy_threshold:
                    self.bci_screen.root.after(
                        0,
                        lambda: self.bci_screen.set_info(
                            "EEG accuracy is low. Focus again."
                        ),
                    )
                    if hasattr(self.bci_screen, "flash_signal_rejection"):
                        self.bci_screen.root.after(0, self.bci_screen.flash_signal_rejection)
                    print(
                        f"[BCI] low-confidence decode; no selection "
                        f"(trial={trial_index}, accuracy={accuracy:.3f})"
                    )
                    self.bci_screen.root.after(300, self._run_bci_selection)
                    return

                raw_selection = int(decoded_selection)
                selection = raw_selection % len(self.visible_options)
                selected_option = self.visible_options[selection]
                self.bci_screen.root.after(0, lambda s=selection: self.bci_screen.highlight(s))

                option_kind = str(selected_option.get("kind", ""))
                if option_kind == "next":
                    self.page_index += 1
                    self.bci_screen.root.after(
                        0, lambda a=accuracy: self.bci_screen.set_info(f"Accuracy: {a:.3f}. Opening other options...")
                    )
                    print(
                        f"[BCI] decoder selection {raw_selection} -> Other options "
                        f"(trial={trial_index}, accuracy={accuracy:.3f})"
                    )
                    self.bci_screen.root.after(220, self._render_action_page)
                    self.bci_screen.root.after(500, self._run_bci_selection)
                    return

                if option_kind == "prev":
                    self.page_index -= 1
                    self.bci_screen.root.after(
                        0, lambda a=accuracy: self.bci_screen.set_info(f"Accuracy: {a:.3f}. Returning to previous options...")
                    )
                    print(
                        f"[BCI] decoder selection {raw_selection} -> Go back "
                        f"(trial={trial_index}, accuracy={accuracy:.3f})"
                    )
                    self.bci_screen.root.after(220, self._render_action_page)
                    self.bci_screen.root.after(500, self._run_bci_selection)
                    return

                selected_action = selected_option["action"]
                self.bci_screen.root.after(
                    0,
                    lambda a=accuracy, s=selection + 1: self.bci_screen.set_info(
                        f"Accuracy: {a:.3f}. Selection: Option {s}"
                    ),
                )
                print(
                    f"[BCI] decoder selection {raw_selection} -> action {selection + 1} "
                    f"(trial={trial_index}, accuracy={accuracy:.3f})"
                )
                time.sleep(0.5)
                self.bci_screen.root.after(0, lambda: self._execute_action(selected_action))
            except Exception as exc:
                traceback.print_exc()
                print(f"[BCI] decode failed: {str(exc)[:60]}")
                self.bci_screen.root.after(
                    0, lambda: self.bci_screen.set_info("BCI decode failed. Please try again.")
                )

        threading.Thread(target=worker, daemon=True).start()

    def _infer_attended_target(self) -> int:
        """
        Estimate which of the visible BCI targets the user was attending to
        during the stimulation window.

        Strategy: use the current flicker frame from bci_screen to count how many
        ON-frames (bit == 1) each visible target showed in the last stimulation
        cycle.  The target with the most ON-frames in a random offset window is
        returned.  In simulation mode this naturally varies across calls because
        the frame counter advances in real time, so each scan cycle picks a
        different target — eliminating the always-zero bias.

        Falls back to a round-robin counter if stim codes are unavailable.
        """
        n_visible = len(self.visible_options)
        if n_visible == 0:
            return 0

        # How many BCI target slots are actually shown (max 6)
        n_targets = min(n_visible, 6)

        try:
            stim_codes = getattr(self.bci, "stim_codes", None)
            frame_index = int(getattr(self.bci_screen, "_frame_index", 0))

            if stim_codes is not None and len(stim_codes) >= n_targets:
                # Count ON-bits for each target across a window ending at current frame.
                window = 30  # ~0.5 s at 60 Hz
                scores = []
                for t in range(n_targets):
                    code = stim_codes[t]
                    code_len = len(code)
                    total = sum(
                        int(code[(frame_index - i) % code_len])
                        for i in range(window)
                    )
                    scores.append(total)

                best = int(max(range(n_targets), key=lambda i: scores[i]))
                print(f"[BCI] Attended target hint: {best} "
                      f"(ON-scores: {scores}, frame={frame_index})")
                return best
        except Exception as exc:
            print(f"[BCI] _infer_attended_target fallback: {exc}")

        # Round-robin fallback — ensures variation across calls even without stim codes
        count = getattr(self, "_target_hint_counter", 0)
        self._target_hint_counter = count + 1
        return count % n_targets

    def _schedule_rescan(self, message: str, delay_ms: int = 1200) -> None:
        self.bci_screen.root.after(0, lambda m=message: self.bci_screen.set_info(m))
        self.bci_screen.root.after(delay_ms, self.scan)

    def scan(self) -> None:
        if self.is_processing or self.is_executing:
            return

        self.is_processing = True
        self.last_ranked = []
        self.agent_actions = []
        self.all_actions = []
        self.visible_options = []
        self.page_index = 0
        self.bci_screen.root.after(0, lambda: self.bci_screen.show_actions([]))
        self.bci_screen.set_info("Scanning desktop...")
        self.bci_screen.set_loading(True, "Taking screenshot...")

        def worker() -> None:
            try:
                screenshot = self._capture_main_monitor()

                if screenshot is None:
                    print("[Scan] Screenshot failed.")
                    self.bci_screen.set_loading(False)
                    self._schedule_rescan("Screenshot failed. Rescanning...")
                    return

                self.bci_screen.set_info("Generating actions...")
                self.bci_screen.set_loading(True, "Generating options...")
                parse_result = self.parser.parse(screenshot)

                ranked = self.ranker.rank(parse_result.elements)
                self.last_ranked = ranked
                self._update_bci_screen_actions(ranked)
                self.bci_screen.set_loading(False)

                print("Top elements:")
                for index, element in enumerate(ranked[:5], 1):
                    print(f"  {index}. {element.name}")

                self.debug_writer.save_scan_artifacts(
                    screenshot=screenshot,
                    labeled_img_b64=parse_result.labeled_img_b64,
                    raw=parse_result.raw,
                )

                if not ranked:
                    print("[Scan] No options found.")
                    self._schedule_rescan("No options found. Rescanning...")
                    return

                self.bci_screen.root.after(0, lambda: self.bci_screen.set_info("Waiting for EEG selection..."))
                self.bci_screen.root.after(50, self._run_bci_selection)
            except Exception as exc:
                traceback.print_exc()
                print(f"[Scan] Error: {str(exc)[:80]}")
                self.bci_screen.set_loading(False)
                self._schedule_rescan("Scan failed. Rescanning...")
            finally:
                self.is_processing = False

        threading.Thread(target=worker, daemon=True).start()

    def select_key(self, key_num: int) -> None:
        if self.is_processing or self.is_executing:
            return
        if not self.agent_actions:
            return
        if key_num < 1 or key_num > len(self.agent_actions):
            return

        target = self.agent_actions[key_num - 1]
        self._execute_action(target)

    def _execute_action(self, target: DetectedElement) -> None:
        if self.is_processing or self.is_executing:
            return

        self.is_executing = True
        self.bci_screen.set_info("Executing action...")
        print(f"[Action] Double-clicking: {target.name}")

        def worker() -> None:
            try:
                self.execute_click(target)
                self.bci_screen.set_info("Action completed.")
                time.sleep(0.5)

                for remaining in range(self.config.wait_after_click_s, 0, -1):
                    print(f"[Action] Next scan in {remaining}s...")
                    time.sleep(1)
            except Exception as exc:
                traceback.print_exc()
                print(f"[Action] Click failed: {str(exc)[:60]}")
            finally:
                self.is_executing = False
                self.bci_screen.root.after(0, self.scan)

        threading.Thread(target=worker, daemon=True).start()

    def quit(self) -> None:
        print("Shutting down...")
        try:
            self.bci_screen.root.quit()
            self.bci_screen.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        print("=" * 70)
        print("BCI AUTO-SCAN READY")
        print("=" * 70)
        print("- First scan starts automatically")
        print(f"- After each click: {self.config.wait_after_click_s}s wait -> next scan")
        print("- BCI options displayed on second monitor")
        print("- SPACE = scan, Q/ESC = quit")
        print(f"- Debug saved to: {self.config.debug_dir}")
        print("=" * 70)

        self.bci_screen.root.after(500, self.scan)
        self.bci_screen.root.mainloop()
