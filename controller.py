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
        self.parser = OmniParserEngine(config, status_callback=self.bci_screen.set_info)
        self.ranker = ElementRanker(config, self.debug_writer)

        self.is_processing = False
        self.is_executing = False
        self.agent_actions: List[DetectedElement] = []
        self.last_ranked: List[DetectedElement] = []

        self.bci_screen.root.bind("<space>", lambda _event: self.scan())
        self.bci_screen.root.bind("q", lambda _event: self.quit())
        self.bci_screen.root.bind("<Escape>", lambda _event: self.quit())

    def _capture_main_monitor(self):
        time.sleep(0.2)
        return self.parser.capture_main_monitor()

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
        self.agent_actions = list(agent_actions[:6])

        actions = [
            str(getattr(action, "goal", getattr(action, "name", "")))
            for action in self.agent_actions
        ]
        self.bci_screen.root.after(0, lambda: self.bci_screen.show_actions(actions))

    def _run_bci_selection(self) -> None:
        if not self.bci:
            return
        if self.is_processing or self.is_executing:
            return
        if not self.agent_actions:
            return

        def worker() -> None:
            try:
                raw_selection = int(self.bci.get_selection())
                selection = raw_selection % len(self.agent_actions)
                selected_action = self.agent_actions[selection]
                self.bci_screen.root.after(0, lambda s=selection: self.bci_screen.highlight(s))
                print(f"[BCI] decoder selection {raw_selection} -> action {selection + 1}")
                self.bci_screen.root.after(0, lambda: self._execute_action(selected_action))
            except Exception as exc:
                traceback.print_exc()
                print(f"[BCI] decode failed: {str(exc)[:60]}")

        threading.Thread(target=worker, daemon=True).start()

    def scan(self) -> None:
        if self.is_processing or self.is_executing:
            return

        self.is_processing = True
        self.last_ranked = []
        self.agent_actions = []
        self.bci_screen.root.after(0, lambda: self.bci_screen.show_actions([]))

        def worker() -> None:
            try:
                screenshot = self._capture_main_monitor()

                if screenshot is None:
                    print("[Scan] Screenshot failed.")
                    return

                parse_result = self.parser.parse(screenshot)

                ranked = self.ranker.rank(parse_result.elements)
                self.last_ranked = ranked
                self._update_bci_screen_actions(ranked)

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
                    return

                self.bci_screen.root.after(50, self._run_bci_selection)
            except Exception as exc:
                traceback.print_exc()
                print(f"[Scan] Error: {str(exc)[:80]}")
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
        print(f"[Action] Double-clicking: {target.name}")

        def worker() -> None:
            try:
                self.execute_click(target)
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
