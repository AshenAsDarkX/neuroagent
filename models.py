from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DetectedElement:
    name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    interactive: bool
    element_type: str = ""
    source: Any = None
    score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bbox": self.bbox,
            "center": self.center,
            "interactive": self.interactive,
            "type": self.element_type,
            "source": self.source,
            "score": self.score,
        }


@dataclass
class ParseResult:
    elements: List[DetectedElement]
    labeled_img_b64: Optional[str]
    raw: Dict[str, Any]
