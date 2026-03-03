import time
from datetime import datetime, timezone
from typing import List
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
    ):
        self.scan = scan_function
        self.execute_click = execute_click_function
        self.find_element = find_element_function
        self.log_action = log_action_function
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

    def run_goal(self, goal: str):

        print(f"Agent started with goal: {goal}")

        while True:
            elements: List[DetectedElement] = self.scan()
            state = infer_state(elements)

            if goal_satisfied(goal, state):
                self._log(selected_element=None, goal_state=state, action_result="goal_achieved", goal=goal)
                print("Goal achieved.")
                break

            action_name = self.planner.next_action(goal, state, elements)

            if not action_name:
                self._log(selected_element=None, goal_state=state, action_result="no_action_returned", goal=goal)
                print("No action returned by planner.")
                break

            target = self.find_element(action_name, elements)

            if not target:
                self._log(selected_element=action_name, goal_state=state, action_result="element_not_found", goal=goal)
                print(f"Element '{action_name}' not found.")
                break

            self.execute_click(target)
            self._log(selected_element=target.name, goal_state=state, action_result="clicked", goal=goal)
            time.sleep(2)
