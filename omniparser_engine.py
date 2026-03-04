from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from PIL import Image, ImageGrab

from app_config import AppConfig
from models import DetectedElement, ParseResult
from utils import bbox_ratio_xyxy_to_pixels, safe_text


class OmniParserEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.omni_available = False
        self._omni: Dict[str, Any] = {}
        self._init_omniparser()

    def _init_omniparser(self) -> None:
        try:
            if not os.path.isdir(self.config.omniparser_dir):
                raise FileNotFoundError(f"OmniParser folder not found: {self.config.omniparser_dir}")

            if self.config.omniparser_dir not in sys.path:
                sys.path.insert(0, self.config.omniparser_dir)

            from util.utils import (
                check_ocr_box,
                get_caption_model_processor,
                get_som_labeled_img,
                get_yolo_model,
            )

            yolo_path = os.path.join(self.config.omniparser_dir, "weights", "icon_detect", "model.pt")
            caption_path = os.path.join(self.config.omniparser_dir, "weights", "icon_caption_florence")

            if not os.path.isfile(yolo_path):
                raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
            if not os.path.isdir(caption_path):
                raise FileNotFoundError(f"Caption model folder not found: {caption_path}")

            print("OmniParser detected")
            print("Loading models...")

            start = time.time()
            yolo_model = get_yolo_model(model_path=yolo_path)
            print(f"YOLO loaded ({time.time() - start:.1f}s)")

            start = time.time()
            caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path=caption_path,
            )
            print(f"Florence2 loaded ({time.time() - start:.1f}s)")

            self._omni = {
                "check_ocr_box": check_ocr_box,
                "get_som_labeled_img": get_som_labeled_img,
                "yolo_model": yolo_model,
                "caption_model_processor": caption_model_processor,
            }
            self.omni_available = True
            print("OmniParser ready")
        except Exception as exc:
            print(f"OmniParser init failed: {exc}")
            traceback.print_exc()
            raise

    def capture_screen_excluding_overlay(self, overlay_hwnd: Optional[int] = None):
        try:
            from PIL import ImageGrab

            # capture full screen
            img = ImageGrab.grab(all_screens=True)

            screen_w, screen_h = img.size
            overlay_w = self.config.overlay_width

            # remove overlay area
            if self.config.overlay_position.lower() == "right":
                img = img.crop((0, 0, screen_w - overlay_w, screen_h))
            else:
                img = img.crop((overlay_w, 0, screen_w, screen_h))

            return img

        except Exception as exc:
            print(f"Screenshot failed: {exc}")
            import traceback
            traceback.print_exc()
            return None

    def parse(self, screenshot: Image.Image) -> ParseResult:
        if not self.omni_available or screenshot is None:
            return ParseResult(elements=[], labeled_img_b64=None, raw={})

        start_time = time.time()
        check_ocr_box = self._omni["check_ocr_box"]
        get_som_labeled_img = self._omni["get_som_labeled_img"]
        yolo_model = self._omni["yolo_model"]
        caption_model_processor = self._omni["caption_model_processor"]

        img_w, img_h = screenshot.size

        try:
            ocr_bbox_result, _ = check_ocr_box(
                screenshot,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=self.config.omni_use_paddleocr,
            )
            ocr_text, ocr_bbox = ocr_bbox_result

            box_overlay_ratio = img_w / 3200
            draw_bbox_config = {
                "text_scale": 0.8 * box_overlay_ratio,
                "text_thickness": max(int(2 * box_overlay_ratio), 1),
                "text_padding": max(int(3 * box_overlay_ratio), 1),
                "thickness": max(int(3 * box_overlay_ratio), 1),
            }

            labeled_b64, label_coordinates, parsed_content_list = get_som_labeled_img(
                screenshot,
                yolo_model,
                BOX_TRESHOLD=self.config.omni_box_thresh,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=ocr_text,
                iou_threshold=self.config.omni_iou_thresh,
                imgsz=self.config.omni_img_sz,
            )

            normalized = self._normalize_parsed_content(parsed_content_list)
            elements = self._build_elements(normalized, img_w, img_h)
            elements.sort(key=lambda item: (1 if item.interactive else 0, item.score), reverse=True)

            total_time = time.time() - start_time
            print(f"Found {len(elements)} content-based elements in {total_time:.1f}s")

            raw_dump = {
                "label_coordinates": label_coordinates,
                "parsed_content_list": normalized,
            }

            return ParseResult(elements=elements, labeled_img_b64=labeled_b64, raw=raw_dump)
        except Exception as exc:
            print(f"OmniParser error: {exc}")
            traceback.print_exc()
            return ParseResult(elements=[], labeled_img_b64=None, raw={})

    @staticmethod
    def _normalize_parsed_content(parsed_content_list: Any) -> List[Dict[str, Any]]:
        if isinstance(parsed_content_list, list):
            return parsed_content_list
        if isinstance(parsed_content_list, dict):
            return [value for _, value in sorted(parsed_content_list.items(), key=lambda pair: str(pair[0]))]
        return []

    def _build_elements(
        self,
        parsed_content_list: List[Dict[str, Any]],
        img_w: int,
        img_h: int,
    ) -> List[DetectedElement]:
        screen_x_offset = self.config.overlay_width if self.config.overlay_position.lower() == "left" else 0
        seen_names = set()
        elements: List[DetectedElement] = []

        for item in parsed_content_list:
            try:
                if not isinstance(item, dict):
                    continue

                interactive = bool(item.get("interactivity", False))
                content = safe_text(item.get("content", ""))
                element_type = safe_text(item.get("type", ""))
                if not content:
                    continue

                name = " ".join(content.replace("PC ", "").split())
                key = name.lower()
                if key in seen_names:
                    continue
                seen_names.add(key)

                bbox = item.get("bbox", None)
                xyxy = bbox_ratio_xyxy_to_pixels(bbox, img_w, img_h) if isinstance(bbox, list) else None
                if not xyxy:
                    continue

                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                elements.append(
                    DetectedElement(
                        name=name,
                        bbox=(x1 + screen_x_offset, y1, x2 + screen_x_offset, y2),
                        center=(cx + screen_x_offset, cy),
                        interactive=interactive,
                        element_type=element_type,
                        source=item.get("source"),
                        score=self._score(name, interactive, element_type),
                    )
                )
            except Exception:
                continue

        return elements

    @staticmethod
    def _score(name: str, interactive: bool, element_type: str) -> int:
        lowered = name.lower()
        score = 0

        if interactive:
            score += 50

        strong_keywords = [
            "open", "new", "delete", "rename", "copy", "paste", "cut", "share",
            "next", "close", "search", "sort", "view",
        ]
        if any(keyword in lowered for keyword in strong_keywords):
            score += 20

        if "m0,0" in lowered or "l9,0" in lowered:
            score -= 50

        name_len = len(lowered)
        if 2 <= name_len <= 18:
            score += 10
        elif name_len <= 30:
            score += 5
        else:
            score -= 5

        if element_type.lower() == "icon":
            score += 5

        return score
