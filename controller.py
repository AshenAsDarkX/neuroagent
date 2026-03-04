from __future__ import annotations

import threading
import time
import traceback
from typing import List, Optional

import pyautogui

from agent_controller import AgentController
from app_config import AppConfig
from debug_writer import DebugArtifactWriter
from models import DetectedElement
from omniparser_engine import OmniParserEngine
from overlay_ui import OverlayUI
from ranking import ElementRanker
from utils import compute_page_slice, ensure_dir


class BCIController:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        ensure_dir(self.config.debug_dir)

        self.debug_writer = DebugArtifactWriter(config)
        self.parser = OmniParserEngine(config)
        self.ranker = ElementRanker(config, self.debug_writer)
        self.agent_controller = AgentController(
            scan_function=self.scan_environment,
            execute_click_function=self.execute_click,
            find_element_function=self.find_element_by_name,
            log_action_function=self.log_agent_action,
        )

        self.is_processing = False
        self.is_executing = False
        self.page = 0
        self.all_elements: List[DetectedElement] = []
        self.overlay_hwnd: Optional[int] = None

        self.ui = OverlayUI(
            config=self.config,
            on_scan=self.scan,
            on_select=self.select_key,
            on_quit=self.quit,
        )
        self.ui.root.update_idletasks()

    def scan_environment(self) -> List[DetectedElement]:
        # hide overlay so it does not appear in screenshot
        try:
            import win32gui
            import win32con
            if self.overlay_hwnd:
                win32gui.ShowWindow(self.overlay_hwnd, win32con.SW_HIDE)
        except Exception:
            pass

        time.sleep(0.15)

        screenshot = self.parser.capture_screen_excluding_overlay(self.overlay_hwnd)

        # show overlay again
        try:
            import win32gui
            import win32con
            if self.overlay_hwnd:
                win32gui.ShowWindow(self.overlay_hwnd, win32con.SW_SHOW)
        except Exception:
            pass

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

    def start_goal_mode(self, goal: str):
        self.agent_controller.run_goal(goal)

    def log_agent_action(
        self,
        timestamp: str,
        selected_element: Optional[str],
        goal_state: Optional[dict],
        action_result: str,
        goal: Optional[str] = None,
    ) -> None:
        self.debug_writer.append_agent_action_log(
            timestamp=timestamp,
            selected_element=selected_element,
            goal_state=goal_state,
            action_result=action_result,
            goal=goal,
        )

    def _render_page(self) -> None:
        self.ui.render_page(self.all_elements, self.page, self.config.items_per_page)

    def scan(self) -> None:
        self.ui.stop_flicker()

        if self.is_processing or self.is_executing:
            return

        self.is_processing = True
        self.page = 0
        self.all_elements = []
        self.ui.clear_options()
        self.ui.set_status("Taking screenshot...", "#f39c12")

        def worker() -> None:
            try:
                screenshot = self.parser.capture_screen_excluding_overlay(self.overlay_hwnd)
                if screenshot is None:
                    self.ui.set_status_threadsafe("Screenshot failed", "#e74c3c")
                    return

                self.ui.set_status_threadsafe("Detecting elements...", "#f39c12")
                parse_result = self.parser.parse(screenshot)

                self.ui.set_status_threadsafe("LLM filtering...", "#f39c12")
                ranked = self.ranker.rank(parse_result.elements)

                print("Top elements:")
                for index, element in enumerate(ranked[:5], 1):
                    print(f"  {index}. {element.name}")

                self.debug_writer.save_scan_artifacts(
                    screenshot=screenshot,
                    labeled_img_b64=parse_result.labeled_img_b64,
                    raw=parse_result.raw,
                )

                if not ranked:
                    self.ui.set_status_threadsafe("No options found. Press SPACE to retry", "#e67e22")
                    return

                self.all_elements = ranked
                self.ui.schedule(0, self._render_page)
            except Exception as exc:
                traceback.print_exc()
                self.ui.set_status_threadsafe(f"Error: {str(exc)[:40]}", "#e74c3c")
            finally:
                self.is_processing = False
                self.ui.focus_overlay(self.overlay_hwnd)

        threading.Thread(target=worker, daemon=True).start()

    def select_key(self, key_num: int) -> None:
        if self.is_processing or self.is_executing:
            return
        if not self.all_elements:
            return

        total = len(self.all_elements)
        start, end, has_prev, has_next = compute_page_slice(
            total,
            self.page,
            self.config.items_per_page,
        )
        chunk = self.all_elements[start:end]

        if key_num == 5 and has_next:
            self.page += 1
            self._render_page()
            return

        if key_num == 6 and has_prev:
            self.page -= 1
            self._render_page()
            return

        if key_num < 1 or key_num > len(chunk):
            return

        target = chunk[key_num - 1]

        self.is_executing = True
        self.ui.clear_options()
        self.ui.set_status(f"Double-clicking: {target.name[:25]}...", "#f39c12")

        def worker() -> None:
            try:
                self.execute_click(target)
                time.sleep(0.5)

                for remaining in range(self.config.wait_after_click_s, 0, -1):
                    self.ui.set_status_threadsafe(f"Next scan in {remaining}s...", "#9b59b6")
                    time.sleep(1)

                self.ui.set_status_threadsafe("Taking screenshot...", "#f39c12")
            except Exception as exc:
                traceback.print_exc()
                self.ui.set_status_threadsafe(f"Click failed: {str(exc)[:30]}", "#e74c3c")
            finally:
                self.is_executing = False
                self.ui.schedule(0, self.scan)

        threading.Thread(target=worker, daemon=True).start()

    def quit(self) -> None:
        print("Shutting down...")
        self.ui.stop()

    def run(self) -> None:
        self.overlay_hwnd = self.ui.get_hwnd()
        print(f"[BCI] Overlay HWND = {self.overlay_hwnd}")

        print("=" * 70)
        print("BCI AUTO-SCAN READY")
        print("=" * 70)
        print("- First scan starts automatically")
        print(f"- After each click: {self.config.wait_after_click_s}s wait -> next scan")
        print("- Press 1-4 to select, 5=More, 6=Back, Q/ESC=Quit")
        print(f"- Overlay on {self.config.overlay_position.upper()} (excluded from scans)")
        print(f"- Debug saved to: {self.config.debug_dir}")
        print("=" * 70)

        self.ui.schedule(500, self.scan)
        self.ui.run()
