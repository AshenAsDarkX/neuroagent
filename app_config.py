from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    base_dir: str
    omniparser_dir: str
    debug_dir: str
    omni_box_thresh: float = 0.05
    omni_iou_thresh: float = 0.10
    omni_img_sz: int = 640
    omni_use_paddleocr: bool = False
    items_per_page: int = 5
    wait_after_click_s: int = 10
    llm_url: str = "http://localhost:11434/api/generate"
    llm_model: str = "gemma3:4b"

    @classmethod
    def load(cls) -> "AppConfig":
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)

        # Resolution order:
        # 1) Explicit environment override
        # 2) OmniParser inside this project folder
        # 3) OmniParser as a sibling folder (common local setup)
        omni_override = os.environ.get("OMNIPARSER_DIR", "").strip()
        candidates = [
            omni_override if omni_override else None,
            os.path.join(base_dir, "OmniParser"),
            os.path.join(parent_dir, "OmniParser"),
        ]
        omniparser_dir = next(
            (path for path in candidates if path and os.path.isdir(path)),
            os.path.join(base_dir, "OmniParser"),
        )

        return cls(
            base_dir=base_dir,
            omniparser_dir=omniparser_dir,
            debug_dir=os.path.join(base_dir, "bci_debug"),
        )
