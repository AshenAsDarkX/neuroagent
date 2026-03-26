import time
from datetime import datetime, timezone
from typing import List, Optional
from models import DetectedElement

from agent_state import infer_state
from goal_evaluator import goal_satisfied
from agent_planner import AgentPlanner


class AgentController:

    def __init__(
        self,
        scan_function,
        execute_click_function,
        find_element_function,
        log_action_function=None,
        max_steps: int = 5,
        status_callback=None,
    ):
        self.scan = scan_function
        self.execute_click = execute_click_function
        self.find_element = find_element_function
        self.log_action = log_action_function
        self.max_steps = max_steps
        self.status_callback = status_callback
        self.planner = AgentPlanner()

    def _log(self, selected_element, goal_state, action_result, goal):
        if not self.log_action:
            return
        self.log_action(
            timestamp=datetime.now(timezone.utc).isoformat(),
            selected_element=selected_element,
            goal_state=goal_state,
            action_result=action_result,
            goal=goal,
        )

    def _status(self, msg: str):
        print(f"[Agent] {msg}")
        if self.status_callback:
            try:
                self.status_callback(msg)
            except Exception:
                pass

    def run_goal(self, goal: str, initial_failed: Optional[List[str]] = None) -> bool:
        """
        Agentic loop:
          1. Scan screen → infer state
          2. Check if goal already satisfied
          3. Ask LLM for next action (passing what failed so far)
          4. Find element → click it
          5. Wait for screen to settle → go to 1
        Returns True if goal was achieved, False otherwise.

        initial_failed: actions already attempted before this loop started
          (e.g. the initial BCI click). Passed to the planner on the first
          step so the LLM does not repeat them.
        """
        self._status(f"Goal: {goal}")
        failed_actions: List[str] = list(initial_failed or [])
        if failed_actions:
            self._status(f"Already tried: {failed_actions}")

        for step in range(self.max_steps):
            self._status(f"Step {step + 1}/{self.max_steps} — scanning screen...")

            elements: List[DetectedElement] = self.scan()
            state = infer_state(elements)

            # --- Check goal satisfaction ---
            was_on_desktop = state.get("was_on_desktop", True)
            if goal_satisfied(goal, state, was_on_desktop=was_on_desktop):
                self._log(None, state, "goal_achieved", goal)
                self._status(f"Goal achieved in {step + 1} step(s).")
                return True

            # --- Ask LLM what to do next ---
            # Pass failed_actions so the planner avoids repeating them
            action_name = self.planner.next_action(
                goal, state, elements, failed_actions=failed_actions
            )

            if not action_name:
                self._log(None, state, "no_action_returned", goal)
                self._status("Planner returned no action. Stopping.")
                break

            # --- Find the element on screen ---
            target = self.find_element(action_name, elements)

            if not target:
                self._log(action_name, state, "element_not_found", goal)
                self._status(f"Element '{action_name}' not on screen. Trying next step.")
                failed_actions.append(action_name)
                continue  # replan rather than stop

            # --- Execute the click ---
            self._status(f"Clicking: {target.name}")
            self.execute_click(target)
            self._log(target.name, state, "clicked", goal)

            # Wait for screen to settle before next scan
            time.sleep(2)

        # --- Max steps reached without success ---
        self._log(None, {}, "max_steps_reached", goal)
        self._status(f"Goal not achieved after {self.max_steps} steps.")
        return False