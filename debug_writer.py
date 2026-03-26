from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image

from app_config import AppConfig
from utils import decode_b64_image_to_pil, ensure_dir, now_tag


class DebugArtifactWriter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        ensure_dir(self.config.debug_dir)

    def save_scan_artifacts(
        self,
        screenshot: Image.Image,
        labeled_img_b64: Optional[str],
        raw: Dict[str, Any],
    ) -> None:
        tag = now_tag()
        out_dir = os.path.join(self.config.debug_dir, tag)
        ensure_dir(out_dir)

        try:
            screenshot.save(os.path.join(out_dir, "screenshot_raw.png"))
        except Exception:
            pass

        labeled_img = decode_b64_image_to_pil(labeled_img_b64) if labeled_img_b64 else None
        if labeled_img:
            try:
                labeled_img.save(os.path.join(out_dir, "screenshot_labeled.png"))
            except Exception:
                pass

        try:
            with open(os.path.join(out_dir, "omniparser_raw.json"), "w", encoding="utf-8") as handle:
                json.dump(raw, handle, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def save_llm_artifacts(
        self,
        survivors_count: int,
        raw_response: str,
        chosen_indices: List[int],
        final_actions: List[str],
    ) -> None:
        try:
            ensure_dir(self.config.debug_dir)
            output_path = os.path.join(self.config.debug_dir, f"llm_proposed_{now_tag()}.json")
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "survivors_count": survivors_count,
                        "raw_response": raw_response,
                        "chosen_indices": chosen_indices,
                        "final_actions": final_actions,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

    def append_agent_action_log(
        self,
        timestamp: str,
        selected_element: Optional[str],
        goal_state: Optional[Dict[str, Any]],
        action_result: str,
        goal: Optional[str] = None,
    ) -> None:
        try:
            ensure_dir(self.config.debug_dir)
            output_path = os.path.join(self.config.debug_dir, "agent_actions.jsonl")
            payload: Dict[str, Any] = {
                "timestamp": timestamp,
                "selected_element": selected_element,
                "goal_state": goal_state,
                "action_result": action_result,
            }
            if goal is not None:
                payload["goal"] = goal

            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
