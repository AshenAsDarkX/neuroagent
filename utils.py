from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from typing import List, Optional, Tuple

from PIL import Image


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_text(value: str) -> str:
    return (value or "").replace("\n", " ").replace("\r", " ").strip()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def bbox_ratio_xyxy_to_pixels(
    bbox: List[float],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    if not bbox or len(bbox) < 4:
        return None

    try:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except Exception:
        return None

    if x2 <= 1.5 and y2 <= 1.5 and (x2 < x1 or y2 < y1):
        return None

    px1 = clamp(int(x1 * img_w), 0, img_w - 1)
    py1 = clamp(int(y1 * img_h), 0, img_h - 1)
    px2 = clamp(int(x2 * img_w), 0, img_w - 1)
    py2 = clamp(int(y2 * img_h), 0, img_h - 1)

    if px2 <= px1 or py2 <= py1:
        return None

    return px1, py1, px2, py2


def decode_b64_image_to_pil(image_b64: str) -> Optional[Image.Image]:
    if not image_b64:
        return None

    try:
        data = image_b64
        if "," in data and data.strip().lower().startswith("data:"):
            data = data.split(",", 1)[1]

        decoded = base64.b64decode(data)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception:
        return None


def compute_page_slice(
    total: int,
    page: int,
    items_per_page: int,
) -> Tuple[int, int, bool, bool]:
    if total <= 0:
        return 0, 0, False, False

    if items_per_page <= 0:
        return 0, 0, False, False

    start = 0
    for p in range(page):
        has_prev = p > 0
        remaining = total - start
        if remaining <= 0:
            return 0, 0, False, False

        max_without_next = items_per_page - (1 if has_prev else 0)
        has_next = remaining > max_without_next
        content_slots = items_per_page - (1 if has_prev else 0) - (1 if has_next else 0)
        if content_slots < 1:
            content_slots = 1

        start += min(content_slots, remaining)

    has_prev = page > 0
    remaining = total - start
    if remaining <= 0:
        return 0, 0, False, False

    max_without_next = items_per_page - (1 if has_prev else 0)
    has_next = remaining > max_without_next
    content_slots = items_per_page - (1 if has_prev else 0) - (1 if has_next else 0)
    if content_slots < 1:
        content_slots = 1

    end = start + min(content_slots, remaining)
    return start, end, has_prev, has_next
