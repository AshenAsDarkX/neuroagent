from __future__ import annotations

import random
import threading
import time
import traceback
from typing import Any, List, Optional

import pyautogui

from agent_controller import AgentController
from agent_state import infer_state
from goal_evaluator import goal_satisfied
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

        self.bci_screen.set_loading(True, f"Warming up {config.llm_model}...")
        self._warmup_llm()
        self.bci_screen.set_loading(False)

        # Agentic loop — wired to the environment scan, not the BCI scan
        self.agent = AgentController(
            scan_function=self.scan_environment,
            execute_click_function=self.execute_click,
            find_element_function=self.find_element_by_name,
            log_action_function=self.debug_writer.append_agent_action_log,
            max_steps=5,
            status_callback=self.bci_screen.set_info,
        )
        self.current_goal: str = ""

        self.is_processing = False
        self.is_executing = False
        self.agent_actions: List[DetectedElement] = []
        self.all_actions: List[DetectedElement] = []
        self.visible_options: list[dict[str, Any]] = []
        self.page_index = 0
        self.last_ranked: List[DetectedElement] = []

        # Simulation mode: cycles through EEG trial groups so each decode
        # uses data from a different pre-recorded participant gaze target.
        # In real EEG mode this would be replaced by live signal acquisition.
        self._sim_trial_group: int = 0
        self._live_eeg_running: bool = False

        # Tracks whether the LAST scan happened while a window was open.
        # Used to classify BCI selections as launch goals vs in-app actions.
        self.current_context_is_window: bool = False

        self.bci_screen.root.bind("<space>", lambda _event: self.scan())
        self.bci_screen.root.bind("q", lambda _event: self.quit())
        self.bci_screen.root.bind("<Escape>", lambda _event: self.quit())

    def _warmup_llm(self) -> None:
        """
        Send a minimal prompt to Ollama at startup so the model is fully
        loaded into VRAM before the first real scan. Without this, the first
        call cold-loads the model and can exceed the 60-120s timeout.
        """
        import requests
        try:
            print(f"[LLM] Warming up {self.config.llm_model}...")
            response = requests.post(
                self.config.llm_url,
                json={
                    "model": self.config.llm_model,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=180,  # generous — only runs once at startup
            )
            if response.status_code == 200:
                print(f"[LLM] {self.config.llm_model} warmed up and ready.")
            else:
                print(f"[LLM] Warmup got HTTP {response.status_code} — continuing anyway.")
        except Exception as exc:
            print(f"[LLM] Warmup failed ({exc}) — model will load on first use.")

    def run_stimulation_phase(self) -> bool:
        if self.is_processing or self.is_executing or not self.visible_options:
            return False

        duration = 5.0
        step = 0.1
        total_steps = int(duration / step)
        bar_width = 10

        self.bci_screen.set_info("Focus on your target")
        start_time = time.perf_counter()

        # Start live EEG streaming in background thread
        self._live_eeg_running = True
        def _stream_eeg():
            """Generate and push EEG snippets to display while stimulus runs."""
            import numpy as np
            sampling_rate = 600
            snippet_len = 120  # 0.2s of EEG at 600Hz
            frame_interval = 0.1  # push a new frame every 100ms
            n_channels = 8
            noise_level = 0.6

            # Use target 0 code as placeholder during stimulus
            # (the actual target hint is unknown until decode)
            code_full = self.bci.stim_codes_upsampled[0]
            code_short = list(code_full[:snippet_len])

            buffer = []  # rolling EEG buffer for display
            buffer_max = 600  # show up to 1 second at a time

            trial_num = 0
            while self._live_eeg_running:
                t0 = time.perf_counter()

                # Generate a short EEG snippet with noise + code structure
                eeg = np.random.randn(snippet_len, n_channels) * noise_level
                for ch in range(n_channels):
                    eeg[:, ch] += (
                        self.bci._generate_pink_noise(snippet_len) * 0.3 * noise_level
                    )
                    lag = ch
                    shifted = np.roll(code_full[:snippet_len], lag)
                    eeg[:, ch] += shifted * (1.0 - noise_level)

                # Append to rolling buffer
                buffer.extend(eeg[:, 0].tolist())
                if len(buffer) > buffer_max:
                    buffer = buffer[-buffer_max:]

                code_display = list(code_full[:len(buffer)])

                # Elapsed progress
                elapsed = time.perf_counter() - start_time
                pct = min(int((elapsed / duration) * 100), 100)
                label = f"Live EEG  {pct}%"

                if hasattr(self.bci_screen, 'update_eeg_wave'):
                    self.bci_screen.update_eeg_wave(
                        list(buffer), code_display, label
                    )

                trial_num += 1
                # Sleep to maintain ~10fps
                elapsed_gen = time.perf_counter() - t0
                sleep_t = max(0, frame_interval - elapsed_gen)
                time.sleep(sleep_t)

        eeg_thread = threading.Thread(target=_stream_eeg, daemon=True)
        eeg_thread.start()

        for i in range(total_steps):
            if self.is_processing or self.is_executing or not self.visible_options:
                self._live_eeg_running = False
                return False

            progress = int(((i + 1) / total_steps) * bar_width)
            bar = "#" * progress + "-" * (bar_width - progress)
            self.bci_screen.set_info(f"Stimulus running: [{bar}]")

            target_time = start_time + ((i + 1) * step)
            sleep_s = target_time - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        # Stop live streaming — decode phase takes over
        self._live_eeg_running = False
        eeg_thread.join(timeout=0.5)
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

    def _is_risky_label(self, label: str) -> bool:
        """Mirror of ElementRanker risky check — keeps controller self-contained."""
        risky_kw = {"close", "delete", "remove", "uninstall", "format", "wipe",
                    "erase", "rename", "terminate", "kill", "shutdown", "restart",
                    "disable", "reset", "sign out", "log out"}
        low = label.lower()
        return any(kw in low for kw in risky_kw)

    def _sort_page_slice(self, actions: List[DetectedElement]) -> List[DetectedElement]:
        """Within a page slice keep safe actions first, risky ones last."""
        safe  = [a for a in actions if not self._is_risky_label(a.name)]
        risky = [a for a in actions if     self._is_risky_label(a.name)]
        return safe + risky

    def _render_action_page(self) -> None:
        total = len(self.all_actions)
        if total == 0:
            self.visible_options = []
            self.agent_actions = []
            self.page_index = 0
            self.bci_screen.root.after(0, lambda: self.bci_screen.show_actions([]))
            self.bci_screen.root.after(
                0, lambda: self.bci_screen.set_info("No options found. Press SPACE to rescan.")
            )
            return

        max_page = self._max_page_index()
        self.page_index = max(0, min(self.page_index, max_page))

        options: list[dict[str, Any]] = []

        if self.page_index == 0:
            page_actions = self._sort_page_slice(self.all_actions[:5])
            for action in page_actions:
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
            page_actions = self._sort_page_slice(self.all_actions[start : start + 4])
            for action in page_actions:
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

                # Simulation mode: random target hint each decode.
                # Replace with live EEG acquisition for real hardware use.
                target_option = random.randint(0, 5)

                decoded_selection, accuracy, trial_index = self.bci.get_selection(target_option)
                accuracy_threshold = float(getattr(self.bci, "accuracy_threshold", 0.40))

                # Keep chart visible for 2s then clear both displays
                self.bci_screen.root.after(2000, self.bci_screen.clear_correlations)
                self.bci_screen.root.after(2000, self.bci_screen.clear_eeg_wave)

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
                    return

                self.bci_screen.set_info("Generating actions...")
                self.bci_screen.set_loading(True, "Generating options...")
                parse_result = self.parser.parse(screenshot)

                ranked = self.ranker.rank(parse_result.elements)
                self.last_ranked = ranked

                # Record whether we are currently inside an app window or on the desktop.
                # This is used when the user makes a BCI selection to classify
                # the goal as a launch goal (desktop) or in-app action (window).
                from agent_state import infer_state as _infer
                _ctx_state = _infer(parse_result.elements)
                self.current_context_is_window = bool(_ctx_state.get("window_open", False))

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
                    return

                self.bci_screen.root.after(0, lambda: self.bci_screen.set_info("Waiting for EEG selection..."))
                self.bci_screen.root.after(50, self._run_bci_selection)
            except Exception as exc:
                traceback.print_exc()
                print(f"[Scan] Error: {str(exc)[:80]}")
                self.bci_screen.set_loading(False)
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
        """Entry point from BCI selection — extracts goal and hands off to agentic loop."""
        if self.is_processing or self.is_executing:
            return

        # The goal label is stored on the element by the ranker (e.g. "open Spotify")
        goal = str(getattr(target, "goal", target.name))
        self.current_goal = goal
        self._execute_goal(goal, first_target=target)

    def _execute_goal(self, goal: str, first_target: DetectedElement = None) -> None:
        """
        Agentic execution loop:
        1. Click the BCI-selected element (first action)
        2. Scan screen → check if goal achieved
        3. If not, ask LLM what to do next → click → repeat up to max_steps

        This is what makes the system agentic:
        - It observes the outcome of each action
        - It replans when the goal is not yet met
        - It avoids repeating failed actions
        """
        if self.is_processing or self.is_executing:
            return

        self.is_executing = True

        def worker() -> None:
            try:
                # Step 1: execute the BCI-selected click immediately
                if first_target:
                    print(f"[Agent] Initial BCI selection: {first_target.name} → goal: {goal}")
                    self.bci_screen.set_info(f"Executing: {first_target.name}")
                    self.execute_click(first_target)
                    time.sleep(2)  # let screen settle

                # Step 2: check if goal already satisfied after first click.
                # was_on_desktop tracks whether the BCI scan happened on the
                # desktop (launch goal) or inside an app (in-app action goal).
                # In-app actions are always satisfied after one click.
                elements = self.scan_environment()
                from agent_state import infer_state
                from goal_evaluator import goal_satisfied
                # State BEFORE the click tells us the context we came from
                was_on_desktop = not bool(self.current_context_is_window)
                state = infer_state(elements)
                state["was_on_desktop"] = was_on_desktop

                if goal_satisfied(goal, state, was_on_desktop=was_on_desktop):
                    print(f"[Agent] Goal achieved after initial click: {goal}")
                    self.bci_screen.set_info(f"Goal achieved: {goal}")
                    time.sleep(1)
                else:
                    # Step 3: hand off to the full agentic loop for replanning.
                    # Only makes sense for launch goals (was_on_desktop=True).
                    # In-app actions that fail just give up — there is nothing
                    # to replan since the action either worked or it did not.
                    if not was_on_desktop:
                        print(f"[Agent] In-app action complete (no verification possible): {goal}")
                    else:
                        already_tried = [first_target.name] if first_target else []
                        print(f"[Agent] Goal not yet satisfied — starting agentic loop (already tried: {already_tried})")
                        self.bci_screen.set_info("Replanning...")
                        self.agent.run_goal(goal, initial_failed=already_tried)

                # Countdown before next BCI scan cycle
                self.bci_screen.set_info("Action complete. Rescanning...")
                for remaining in range(self.config.wait_after_click_s, 0, -1):
                    print(f"[Agent] Next scan in {remaining}s...")
                    time.sleep(1)

            except Exception as exc:
                traceback.print_exc()
                print(f"[Agent] Goal execution failed: {str(exc)[:60]}")
            finally:
                self.is_executing = False
                self.current_goal = ""
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